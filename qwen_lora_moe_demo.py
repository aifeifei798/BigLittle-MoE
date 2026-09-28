import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# ----------------------------------------------------------------------
# 1. Ultra-Lightweight LoRA Micro-Expert (Rank=16, Footprint: ~64 KB)
# ----------------------------------------------------------------------
class LoRAMicroExpert(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 16, lora_alpha: float = 16.0, dtype=torch.bfloat16):
        super().__init__()
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        # Low-rank decomposition: W_A (H -> r), W_B (r -> H)
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, dtype=dtype)
        
        # Initialize A with Kaiming uniform, B with small normal (for immediate demonstration)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.normal_(self.lora_B.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard LoRA forward: Δy = (x @ W_A^T @ W_B^T) * scaling
        return self.lora_B(self.lora_A(x)) * self.scaling

# ----------------------------------------------------------------------
# 2. BigLittle-Qwen LoRA Wrapper:
#    GPU-Resident Native MLP (Big Core) + Host RAM Streamed LoRAs
# ----------------------------------------------------------------------
class BigLittleQwenLoRAWrapper(nn.Module):
    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        rank: int = 16,
        num_lora_experts: int = 32,   # 32 fine-grained LoRA experts per layer
        device: str = "cuda:0",
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # [Big Core]: Base Qwen MLP permanently pinned in GPU VRAM
        self.big_core = original_mlp

        # [Router]: Ultra-lightweight gating layer on GPU
        self.router = nn.Linear(hidden_dim, num_lora_experts, bias=False, device=device, dtype=dtype)

        # [LoRA Pool in Host RAM]: 32 tiny LoRA adapters (64 KB each) stored in DDR memory
        self.lora_pool_cpu = nn.ModuleList([
            LoRAMicroExpert(hidden_dim, rank=rank, dtype=dtype).to("cpu")
            for _ in range(num_lora_experts)
        ])
        for param in self.lora_pool_cpu.parameters():
            param.data = param.data.pin_memory()

        # Dedicated non-blocking CUDA stream
        self.transfer_stream = torch.cuda.Stream(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Big Core computes directly in VRAM (zero transfer)
        big_out = self.big_core(x)

        # 2. Router scores current token to choose the optimal LoRA adapter
        current_token = x[:, -1:, :]
        router_logits = self.router(current_token)
        selected_idx = torch.argmax(router_logits, dim=-1).item()

        # 3. Stream the selected 64 KB LoRA module from Host RAM via PCIe DMA
        selected_lora_cpu = self.lora_pool_cpu[selected_idx]
        selected_lora_gpu = selected_lora_cpu.to(self.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # 4. Compute dynamic LoRA perturbation & blend with Big Core output
        lora_out = selected_lora_gpu(x)
        return big_out + lora_out

# ----------------------------------------------------------------------
# 3. Model Loading & In-Place LoRA-MoE Patching
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    print(f"[*] Loading base model {model_id} to GPU...")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda:0"
    )

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)
    num_loras_per_layer = 32
    total_loras = num_layers * num_loras_per_layer

    print(f"[+] Base model loaded: {num_layers} layers, hidden_dim = {hidden_dim}")
    baseline_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"[+] Baseline Qwen VRAM Allocated: {baseline_vram:.2f} MB")

    # Patch every layer with BigLittle-LoRA MoE
    print(f"[*] Patching layers with Mixture-of-LoRA (MoLE) architecture...")
    for layer in model.model.layers:
        layer.mlp = BigLittleQwenLoRAWrapper(
            original_mlp=layer.mlp,
            hidden_dim=hidden_dim,
            rank=16,
            num_lora_experts=num_loras_per_layer,
            device="cuda:0",
            dtype=dtype
        )

    patched_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_delta = patched_vram - baseline_vram
    
    # Calculate physical RAM consumption of all LoRAs
    # Each LoRA: 2 * (1024 * 16) params * 2 bytes = 65,536 bytes (~64 KB)
    ram_footprint_mb = (total_loras * 65536) / (1024 ** 2)

    print(f"[+] MoLE Patching Complete!")
    print(f"    - Total LoRA Micro-Experts: {total_loras} experts across {num_layers} layers")
    print(f"    - Host RAM Pool Footprint : ~{ram_footprint_mb:.2f} MB (Only ~64 KB per expert!)")
    print(f"    - Added GPU VRAM Delta    : {vram_delta:.2f} MB (Router heads only)")

    # ------------------------------------------------------------------
    # 4. End-to-End Generation Benchmark
    # ------------------------------------------------------------------
    prompt = "Artificial Intelligence will change the future because"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    print(f"\n[+] Running MoLE end-to-end text generation benchmark...")
    print(f"    Prompt: \"{prompt}\"")
    
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    total_time_ms = (time.perf_counter() - t0) * 1000

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    throughput = 40 / (total_time_ms / 1000)

    print(f"\n[✔] MoLE Generated Output:")
    print(f"--------------------------------------------------")
    print(generated_text)
    print(f"--------------------------------------------------")
    print(f"[⚡] Generated 40 tokens in {total_time_ms:.2f} ms (Throughput: {throughput:.2f} tokens/s)")

if __name__ == "__main__":
    main()
