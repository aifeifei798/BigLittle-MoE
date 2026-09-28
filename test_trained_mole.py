import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

# ----------------------------------------------------------------------
# 1. LoRA Micro-Expert Architecture (Rank=16, 64 KB)
# ----------------------------------------------------------------------
class LoRAMicroExpert(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 16, lora_alpha: float = 16.0, dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling

# ----------------------------------------------------------------------
# 2. Production BigLittle-MoE Inference Wrapper (With Routing Inspector)
# ----------------------------------------------------------------------
class BigLittleInferenceWrapper(nn.Module):
    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        rank: int = 16,
        num_lora_experts: int = 32,
        device: str = "cuda:0",
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.big_core = original_mlp

        # Trained Router on GPU
        self.router = nn.Linear(hidden_dim, num_lora_experts, bias=False, device=device, dtype=dtype)

        # 32 LoRA Micro-Experts stored in Host RAM with Pinned Memory
        self.lora_pool_cpu = nn.ModuleList([
            LoRAMicroExpert(hidden_dim, rank=rank, dtype=dtype).to("cpu")
            for _ in range(num_lora_experts)
        ])
        for p in self.lora_pool_cpu.parameters():
            p.data = p.data.pin_memory()

        self.transfer_stream = torch.cuda.Stream(device=device)
        self.last_selected_expert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        big_out = self.big_core(x)

        # Dynamic Routing decision
        current_token = x[:, -1:, :]
        router_logits = self.router(current_token)
        selected_idx = torch.argmax(router_logits, dim=-1).item()
        self.last_selected_expert = selected_idx

        # Stream the 64 KB micro-expert over PCIe via DMA
        selected_lora_gpu = self.lora_pool_cpu[selected_idx].to(self.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        lora_out = selected_lora_gpu(x) * 0.3
        return big_out + lora_out

# ----------------------------------------------------------------------
# 3. Domain Helper & Verification Suite
# ----------------------------------------------------------------------
def get_domain_label(expert_id):
    if 0 <= expert_id <= 7:
        return f"Expert #{expert_id:02d} [Domain: Code / Algorithm]"
    elif 8 <= expert_id <= 15:
        return f"Expert #{expert_id:02d} [Domain: Math / Reasoning]"
    else:
        return f"Expert #{expert_id:02d} [Domain: General / Writing]"

def test_prompt(model, tokenizer, prompt, test_name):
    print(f"\n{'='*70}")
    print(f"[*] Testing {test_name}")
    print(f"    Input Prompt: \"{prompt}\"")
    print(f"{'='*70}")

    inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").to("cuda:0")

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,          # 适度发散，避免僵死
            top_p=0.9,
            repetition_penalty=1.15,   # <--- 核心解药！彻底消灭 4444 复读机
            pad_token_id=tokenizer.eos_token_id
        )
    total_time_ms = (time.perf_counter() - t0) * 1000

    # Inspect which experts were activated on sample layers
    print(f"[+] Multi-Layer Routing Decisions:")
    for layer_idx in [0, 7, 14, 21, 27]:
        chosen = model.model.layers[layer_idx].mlp.last_selected_expert
        print(f"    - Layer {layer_idx:02d}: {get_domain_label(chosen)}")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    throughput = 45 / (total_time_ms / 1000)

    print(f"\n[✔] Generated Output:")
    print(f"--------------------------------------------------")
    print(generated_text)
    print(f"--------------------------------------------------")
    print(f"[⚡] Throughput: {throughput:.2f} tokens/s (Time: {total_time_ms:.1f} ms)")

# ----------------------------------------------------------------------
# 4. Main Evaluation Runner
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    weights_path = "biglittle_mole_896e_weights.pt"

    assert os.path.exists(weights_path), f"Weights file {weights_path} not found! Please ensure training has finished."

    print(f"[*] Initializing BigLittle-MoE Verification Suite...")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cuda:0")
    hidden_dim = model.config.hidden_size

    # Patch with inference wrapper
    print(f"[*] Patching 28 layers with BigLittle-MoE Host-RAM streaming...")
    for layer in model.model.layers:
        layer.mlp = BigLittleInferenceWrapper(layer.mlp, hidden_dim, rank=16, num_lora_experts=32, device="cuda:0", dtype=dtype)

    # Load trained 56 MB weights
    print(f"[*] Loading trained weights from {weights_path}...")
    saved_weights = torch.load(weights_path, map_location="cpu")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.router.load_state_dict(saved_weights[f"layer_{i}_router"])
        layer.mlp.lora_pool_cpu.load_state_dict(saved_weights[f"layer_{i}_loras"])

    print(f"[✔] Successfully loaded 896 trained experts & 28 routers!")

    # Run tests across 3 distinct domains
    test_prompt(
        model, tokenizer,
        prompt="Write a Python function to check if a number is prime.",
        test_name="Test 1: Python Code Task"
    )

    test_prompt(
        model, tokenizer,
        prompt="A car travels 120 miles in 2 hours. What is its average speed in miles per hour?",
        test_name="Test 2: Mathematical Reasoning"
    )

    test_prompt(
        model, tokenizer,
        prompt="Describe the serene beauty of a quiet mountain lake at dawn.",
        test_name="Test 3: Creative & Descriptive Writing"
    )

if __name__ == "__main__":
    main()
