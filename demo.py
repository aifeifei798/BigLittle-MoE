import torch
import torch.nn as nn
import time

# ----------------------------------------------------------------------
# 1. Base SwiGLU Expert Architecture
# ----------------------------------------------------------------------
class MLPExpert(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

# ----------------------------------------------------------------------
# 2. BigLittle-MoE Layer: GPU-Resident Big Core + Streamed CPU Little Cores
# ----------------------------------------------------------------------
class BigLittleMoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,
        big_dim: int = 8192,
        little_dim: int = 1024,
        num_little_experts: int = 16,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.device = device
        self.num_little_experts = num_little_experts

        print(f"[*] Initializing BigLittle-MoE Layer:")
        print(f"    - Big Core (GPU-Resident)    : intermediate_dim = {big_dim}")
        print(f"    - Little Cores (Host RAM Pool): intermediate_dim = {little_dim} (Count: {num_little_experts})")

        # [Component 1] Big Core: Permanently resident in GPU VRAM (Zero PCIe overhead)
        self.big_core = MLPExpert(hidden_dim, big_dim).to(self.device).half()

        # [Component 2] Router: GPU-native gating mechanism
        self.router = nn.Linear(hidden_dim, num_little_experts).to(self.device).half()

        # [Component 3] Little Core Pool: Kept in Host RAM with Pinned Memory for DMA transfer
        self.little_cores_cpu = nn.ModuleList([
            MLPExpert(hidden_dim, little_dim).to("cpu").half()
            for _ in range(num_little_experts)
        ])
        for param in self.little_cores_cpu.parameters():
            param.data = param.data.pin_memory()

        # Dedicated CUDA stream for non-blocking asynchronous streaming
        self.transfer_stream = torch.cuda.Stream(device=self.device)

    def forward(self, x: torch.Tensor):
        # 1. Big Core computes directly in VRAM
        big_out = self.big_core(x)

        # 2. Router scores and selects Top-1 Little Expert for the current token
        router_logits = self.router(x)
        selected_idx = torch.argmax(router_logits, dim=-1).item()

        # 3. Stream the selected Micro-Expert from Host RAM to GPU via PCIe DMA
        t0 = time.perf_counter()
        selected_little_cpu = self.little_cores_cpu[selected_idx]
        
        selected_little_gpu = selected_little_cpu.to(self.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        transfer_latency_ms = (time.perf_counter() - t0) * 1000

        # 4. Little Core computes on GPU
        little_out = selected_little_gpu(x)

        # 5. Aggregate: Foundational reasoning + Specialized knowledge injection
        final_out = big_out + little_out

        return final_out, selected_idx, transfer_latency_ms

# ----------------------------------------------------------------------
# 3. Verification & Benchmark
# ----------------------------------------------------------------------
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA device required for this benchmark."
    
    torch.cuda.empty_cache()
    baseline_vram = torch.cuda.memory_allocated() / (1024 ** 2)

    # Instantiate layer: Hidden=4096, BigCore=8192, 16 LittleCores=1024
    layer = BigLittleMoELayer(
        hidden_dim=4096,
        big_dim=8192,
        little_dim=1024,
        num_little_experts=16
    )

    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"\n[+] Memory Footprint:")
    print(f"    - GPU VRAM Allocated (Big Core Only): {allocated_vram - baseline_vram:.2f} MB")
    print(f"    - Host RAM Footprint (16 Little Cores): ~{16 * 25.2 / 8:.2f} MB")

    # Simulate token forward pass
    token_input = torch.randn(1, 4096, device="cuda:0", dtype=torch.float16)

    # Warmup pass
    _ = layer(token_input)

    print(f"\n[+] Running Inference & Dynamic Streaming Benchmark:")
    latencies = []
    for step in range(5):
        _, expert_idx, t_ms = layer(token_input)
        latencies.append(t_ms)
        print(f"    [Step {step+1}] Routed to Expert #{expert_idx:02d} | PCIe Transfer Latency: {t_ms:.3f} ms")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\n[✔] Average Micro-Expert Streaming Latency: {avg_latency:.3f} ms / token")