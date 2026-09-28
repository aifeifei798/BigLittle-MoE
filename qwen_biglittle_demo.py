import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# ----------------------------------------------------------------------
# 1. Lightweight Micro-Expert Architecture
# ----------------------------------------------------------------------
class MicroExpert(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

# ----------------------------------------------------------------------
# 2. BigLittle-Qwen MLP Wrapper:
#    Wraps Qwen's native MLP as the GPU Big Core + dynamically streams Host RAM Little Cores
# ----------------------------------------------------------------------
class BigLittleQwenMLPWrapper(nn.Module):
    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        little_dim: int = 512,
        num_little_experts: int = 16,
        device: str = "cuda:0",
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # [Big Core]: Pretrained native Qwen MLP permanently pinned in GPU VRAM
        self.big_core = original_mlp

        # [Router]: Lightweight gating layer running natively on GPU
        self.router = nn.Linear(hidden_dim, num_little_experts, bias=False, device=device, dtype=dtype)

        # [Little Core Pool]: Stored in Host System RAM (DDR) using pinned memory for high-speed DMA
        self.little_cores_cpu = nn.ModuleList([
            MicroExpert(hidden_dim, little_dim, dtype=dtype).to("cpu")
            for _ in range(num_little_experts)
        ])
        for param in self.little_cores_cpu.parameters():
            param.data = param.data.pin_memory()

        # Dedicated CUDA stream for non-blocking asynchronous transfers
        self.transfer_stream = torch.cuda.Stream(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Big Core computes directly in GPU VRAM (zero PCIe overhead)
        big_out = self.big_core(x)

        # 2. Router scores current token to select top-1 micro-expert
        current_token = x[:, -1:, :]
        router_logits = self.router(current_token)
        selected_idx = torch.argmax(router_logits, dim=-1).item()

        # 3. Stream selected micro-expert from Host RAM via PCIe DMA
        selected_little_cpu = self.little_cores_cpu[selected_idx]
        selected_little_gpu = selected_little_cpu.to(self.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # 4. Compute micro-expert forward pass and blend outputs (0.1 residual scaling)
        little_out = selected_little_gpu(x) * 0.1
        
        return big_out + little_out

# ----------------------------------------------------------------------
# 3. Model Loading & In-Place Monkey-Patching
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    print(f"[*] Loading model {model_id} to GPU...")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda:0"
    )

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)
    print(f"[+] Base model loaded successfully: {num_layers} layers, hidden_dim = {hidden_dim}")

    # Track baseline GPU VRAM
    baseline_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"[+] Qwen Baseline VRAM Allocated: {baseline_vram:.2f} MB")

    # In-place architectural replacement: patch each layer's MLP into BigLittle-MoE
    print(f"[*] Patching layers into BigLittle-MoE architecture (attaching Host RAM micro-expert pool)...")
    for i, layer in enumerate(model.model.layers):
        layer.mlp = BigLittleQwenMLPWrapper(
            original_mlp=layer.mlp,
            hidden_dim=hidden_dim,
            little_dim=512,          # Fine-grained micro-expert intermediate dimension
            num_little_experts=16,   # 16 micro-experts per layer in Host RAM
            device="cuda:0",
            dtype=dtype
        )

    patched_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_delta = patched_vram - baseline_vram
    print(f"[+] Patching complete! VRAM Delta: {vram_delta:.2f} MB (All micro-experts resident in Host RAM)")

    # ------------------------------------------------------------------
    # 4. End-to-End Generation Benchmark
    # ------------------------------------------------------------------
    prompt = "Artificial Intelligence will change the future because"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    print(f"\n[+] Starting end-to-end text generation test...")
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

    print(f"\n[✔] Generated Output:")
    print(f"--------------------------------------------------")
    print(generated_text)
    print(f"--------------------------------------------------")
    print(f"[⚡] Generated 40 tokens in {total_time_ms:.2f} ms (Throughput: {throughput:.2f} tokens/s)")

if __name__ == "__main__":
    main()