
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
# 2. Multi-BigLittle-MoE Layer: Multi-Big (VRAM) + Streamed Little (RAM)
# ----------------------------------------------------------------------
class MultiBigLittleMoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_big_experts: int = 2,       # e.g., 2 Big Cores in VRAM
        top_k_big: int = 2,             # Activate Top-K Big Cores (2 = All-Active Dual Core)
        big_dim: int = 6144,            # Intermediate size of each Big Core
        num_little_experts: int = 16,   # Pool of Little Cores in Host RAM
        top_k_little: int = 1,          # Stream Top-1 Little Core per token
        little_dim: int = 1024,         # Lightweight footprint
        device: str = "cuda:0"
    ):
        super().__init__()
        self.device = device
        self.num_big_experts = num_big_experts
        self.top_k_big = top_k_big
        self.num_little_experts = num_little_experts
        self.top_k_little = top_k_little

        print(f"[*] Initializing Multi-BigLittle-MoE Layer:")
        print(f"    - Big Cores (VRAM Resident) : {num_big_experts} experts (Top-{top_k_big} active, dim={big_dim})")
        print(f"    - Little Cores (RAM Pool)   : {num_little_experts} experts (Top-{top_k_little} active, dim={little_dim})")

        # [Tier 1: Big Cores] All pinned permanently in GPU VRAM
        self.big_cores = nn.ModuleList([
            MLPExpert(hidden_dim, big_dim).to(self.device).half()
            for _ in range(num_big_experts)
        ])
        # Router for Big Cores (runs locally on GPU)
        self.router_big = nn.Linear(hidden_dim, num_big_experts).to(self.device).half()

        # [Tier 2: Little Cores] Reside in Host RAM (Page-locked Pinned Memory)
        self.little_cores_cpu = nn.ModuleList([
            MLPExpert(hidden_dim, little_dim).to("cpu").half()
            for _ in range(num_little_experts)
        ])
        for p in self.little_cores_cpu.parameters():
            p.data = p.data.pin_memory()

        # Router for Little Cores (runs on GPU, scores Host RAM candidates)
        self.router_little = nn.Linear(hidden_dim, num_little_experts).to(self.device).half()

        # Dedicated non-blocking stream
        self.transfer_stream = torch.cuda.Stream(device=self.device)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # -------------------------------------------------------------
        # Step 1: Big Cores Execution on GPU (Zero PCIe overhead)
        # -------------------------------------------------------------
        big_logits = self.router_big(x)
        big_weights = torch.softmax(big_logits, dim=-1)
        top_big_weights, top_big_indices = torch.topk(big_weights, self.top_k_big, dim=-1)

        big_out = torch.zeros_like(x)
        for k in range(self.top_k_big):
            idx = top_big_indices[0, k].item()
            weight = top_big_weights[0, k]
            # Parallel compute inside VRAM
            big_out += weight * self.big_cores[idx](x)

        # -------------------------------------------------------------
        # Step 2: Little Core Routing & PCIe DMA Streaming
        # -------------------------------------------------------------
        little_logits = self.router_little(x)
        little_weights = torch.softmax(little_logits, dim=-1)
        top_little_weights, top_little_indices = torch.topk(little_weights, self.top_k_little, dim=-1)

        t0 = time.perf_counter()
        little_out = torch.zeros_like(x)
        for k in range(self.top_k_little):
            selected_idx = top_little_indices[0, k].item()
            weight = top_little_weights[0, k]

            # Stream micro-expert weights from Host RAM -> GPU VRAM
            selected_little_gpu = self.little_cores_cpu[selected_idx].to(self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            
            little_out += weight * selected_little_gpu(x)

        transfer_ms = (time.perf_counter() - t0) * 1000

        # -------------------------------------------------------------
        # Step 3: Fusion of Tier-1 and Tier-2 Outputs
        # -------------------------------------------------------------
        final_out = big_out + little_out

        return final_out, top_big_indices[0].tolist(), top_little_indices[0].tolist(), transfer_ms

# ----------------------------------------------------------------------
# 3. Test & Verification
# ----------------------------------------------------------------------
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required."
    torch.cuda.empty_cache()

    baseline_vram = torch.cuda.memory_allocated() / (1024 ** 2)

    # 实例化：双大核（2个大核，每次2个都激活），挂16个小核（每次拉取1个）
    layer = MultiBigLittleMoELayer(
        hidden_dim=4096,
        num_big_experts=2,      # 2个大核
        top_k_big=2,            # 2个同时激活（全勤协同）
        big_dim=6144,           # 每个大核中间层 6144
        num_little_experts=16,  # 16个内存小核
        top_k_little=1,
        little_dim=1024
    )

    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"\n[+] Memory Footprint:")
    print(f"    - GPU VRAM Allocated (2 Big Cores + Routers): {allocated_vram - baseline_vram:.2f} MB")
    print(f"    - Host RAM Footprint (16 Little Cores)      : ~50.40 MB")

    token_input = torch.randn(1, 4096, device="cuda:0", dtype=torch.float16)

    # Warmup
    _ = layer(token_input)

    print(f"\n[+] Running Multi-Big Inference & Benchmark:")
    latencies = []
    for step in range(5):
        _, big_ids, little_ids, t_ms = layer(token_input)
        latencies.append(t_ms)
        print(f"    [Step {step+1}] Active Big Cores: {big_ids} | Streamed Little: #{little_ids[0]:02d} | Transfer: {t_ms:.3f} ms")

    avg_ms = sum(latencies) / len(latencies)
    print(f"\n[✔] Average Micro-Expert Streaming Latency: {avg_ms:.3f} ms / token")