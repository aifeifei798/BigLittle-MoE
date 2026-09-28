"""Standalone latency micro-benchmark for the two-tier layer.

Merges the original ``demo.py`` (single big core) and ``demo_multi_big.py``
(dual big cores) into one script with a flag. No language model involved --
this measures raw PCIe streaming behaviour.

    python benchmarks/toy_layer.py --big-cores 1
    python benchmarks/toy_layer.py --big-cores 2
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


class MLPExpert(nn.Module):
    """SwiGLU block, the shape of a Qwen-style FFN."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype=torch.float16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class BigLittleLayer(nn.Module):
    """``num_big_cores`` experts in VRAM + ``num_little_experts`` streamed from RAM."""

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_big_cores: int = 1,
        big_dim: int = 8192,
        num_little_experts: int = 16,
        little_dim: int = 1024,
        top_k_little: int = 1,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.num_big_cores = num_big_cores
        self.num_little_experts = num_little_experts
        self.top_k_little = top_k_little
        self.device = device

        print("[*] Initialising BigLittle-MoE layer")
        print(f"    - Big cores (VRAM)   : {num_big_cores} x dim {big_dim}")
        print(f"    - Little cores (RAM) : {num_little_experts} x dim {little_dim} "
              f"(top-{top_k_little} per token)")

        self.big_cores = nn.ModuleList(
            MLPExpert(hidden_dim, big_dim, torch.float16).to(device)
            for _ in range(num_big_cores)
        )
        self.router_big = (
            nn.Linear(hidden_dim, num_big_cores, bias=False).to(device, torch.float16)
            if num_big_cores > 1 else None
        )
        self.little_cores_cpu = nn.ModuleList(
            MLPExpert(hidden_dim, little_dim, torch.float16).to("cpu")
            for _ in range(num_little_experts)
        )
        for p in self.little_cores_cpu.parameters():
            p.data = p.data.pin_memory()
        self.router_little = nn.Linear(
            hidden_dim, num_little_experts, bias=False
        ).to(device, torch.float16)
        self.transfer_stream = torch.cuda.Stream(device=device)

    def _stream(self, selected: list[int]) -> list[nn.Module]:
        """Queue every selected expert onto the transfer stream, sync once."""
        with torch.cuda.stream(self.transfer_stream):
            experts = []
            for idx in selected:
                gpu = self.little_cores_cpu[idx].to(self.device, non_blocking=True)
                for p in gpu.parameters():
                    p.record_stream(torch.cuda.current_stream())
                experts.append(gpu)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        return experts

    def forward(self, x: torch.Tensor):
        if self.num_big_cores == 1:
            big_out = self.big_cores[0](x)
            big_ids = [0]
        else:
            weights = torch.softmax(self.router_big(x[:, -1:, :]), dim=-1)
            top_w, top_i = torch.topk(weights, self.num_big_cores, dim=-1)
            big_out = torch.zeros_like(x)
            for k in range(self.num_big_cores):
                big_out = big_out + top_w[:, :, k] * self.big_cores[top_i[0, 0, k].item()](x)
            big_ids = top_i[0, 0].tolist()

        little_w = torch.softmax(self.router_little(x[:, -1:, :]), dim=-1)
        top_lw, top_li = torch.topk(little_w, self.top_k_little, dim=-1)

        t0 = time.perf_counter()
        little_out = torch.zeros_like(x)
        for k in range(self.top_k_little):
            gpu = self._stream([top_li[0, 0, k].item()])[0]
            little_out = little_out + top_lw[:, :, k] * gpu(x)
        transfer_ms = (time.perf_counter() - t0) * 1000

        return big_out + little_out, big_ids, top_li[0, 0].tolist(), transfer_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--big-cores", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=4096)
    ap.add_argument("--little-experts", type=int, default=16)
    ap.add_argument("--top-k-little", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required for this benchmark."
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**2

    layer = BigLittleLayer(
        hidden_dim=args.hidden_dim,
        num_big_cores=args.big_cores,
        num_little_experts=args.little_experts,
        top_k_little=args.top_k_little,
    )

    vram = torch.cuda.memory_allocated() / 1024**2 - baseline
    # 3 matrices of hidden x intermediate, fp16.
    per_little_mb = 3 * args.hidden_dim * 1024 * 2 / 1024**2
    print(f"\n[+] VRAM allocated (big cores + routers): {vram:.2f} MB")
    print(f"[+] Host RAM for {args.little_experts} little cores: "
          f"~{per_little_mb * args.little_experts:.2f} MB")

    # [batch, seq, hidden] to match the decoder-layer shape the wrapper expects.
    x = torch.randn(1, 1, args.hidden_dim, device="cuda:0", dtype=torch.float16)
    for _ in range(3):
        layer(x)
    torch.cuda.synchronize()

    print(f"\n[+] Streaming benchmark ({args.iters} iterations):")
    latencies = []
    for i in range(args.iters):
        _, big_ids, little_ids, t_ms = layer(x)
        latencies.append(t_ms)
        if i < 5:
            print(f"    [{i + 1:02d}] big={big_ids} little={little_ids} "
                  f"| transfer {t_ms:.3f} ms")

    lat = torch.tensor(latencies)
    print(f"\n[+] Stream latency: mean {lat.mean():.3f} ms | "
          f"p50 {lat.median():.3f} ms | p95 {lat.quantile(0.95):.3f} ms")


if __name__ == "__main__":
    main()
