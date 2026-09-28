"""Plumbing smoke test: patch a real LLM with *untrained* micro-experts.

Merges the original ``qwen_biglittle_demo.py`` (SwiGLU micro-experts),
``qwen_dual_big_demo.py`` (two big cores) and ``qwen_lora_moe_demo.py``
(rank-16 LoRA) into one script. The point is to verify the graft survives a
real forward pass and that the VRAM/RAM accounting is what we claim -- not to
produce meaningful text, since nothing is trained here.

    python benchmarks/qwen_plumbing.py --mode lora
    python benchmarks/qwen_plumbing.py --mode swiglu --big-cores 2
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn


from biglittle_moe.config import GenConfig, MoEConfig
from biglittle_moe.generate import generate
from biglittle_moe.model import load_base_model, load_tokenizer
from biglittle_moe.modules import BigLittleMoEWrapper

PROMPT = "Artificial intelligence will change the future because"


class SwiGLUMicroExpert(nn.Module):
    """Full (non-low-rank) micro-expert."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DualBigLittleWrapper(nn.Module):
    """Variant with two resident big cores instead of one."""

    def __init__(self, original_mlp, hidden_dim, num_little=16, little_dim=512,
                 device="cuda:0", dtype=torch.bfloat16, gamma=0.1):
        super().__init__()
        self.device, self.gamma = device, gamma
        self.big_core_1 = original_mlp
        self.big_core_2 = copy.deepcopy(original_mlp)
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)
        self.router_little = nn.Linear(hidden_dim, num_little, bias=False,
                                       device=device, dtype=dtype)
        self.little_cores_cpu = nn.ModuleList(
            SwiGLUMicroExpert(hidden_dim, little_dim, dtype).to("cpu")
            for _ in range(num_little)
        )
        for p in self.little_cores_cpu.parameters():
            p.data = p.data.pin_memory()
        self.transfer_stream = torch.cuda.Stream(device=device)

    def forward(self, x):
        w = torch.softmax(self.router_big(x[:, -1:, :]), dim=-1)
        big_out = w[:, :, 0:1] * self.big_core_1(x) + w[:, :, 1:2] * self.big_core_2(x)

        logits = self.router_little(x[:, -1:, :])
        idx = int(logits.argmax(dim=-1))
        with torch.cuda.stream(self.transfer_stream):
            gpu = self.little_cores_cpu[idx].to(self.device, non_blocking=True)
            for p in gpu.parameters():
                p.record_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        return big_out + self.gamma * gpu(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("lora", "swiglu", "dual-big"), default="lora")
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    cfg = MoEConfig(model_id=args.model_id, num_experts=args.num_experts, top_k=args.top_k)
    tokenizer = load_tokenizer(cfg.model_id)
    model = load_base_model(cfg)
    model.eval()

    hidden = model.config.hidden_size
    n_layers = len(model.model.layers)
    baseline_vram = torch.cuda.memory_allocated() / 1024**2
    print(f"[+] {n_layers} layers, hidden={hidden}, "
          f"baseline VRAM {baseline_vram:.2f} MB")

    print(f"[*] Patching in mode '{args.mode}' (untrained experts) ...")
    for layer in model.model.layers:
        if args.mode == "dual-big":
            layer.mlp = DualBigLittleWrapper(layer.mlp, hidden, device=cfg.device,
                                             dtype=cfg.compute_dtype)
        else:
            layer.mlp = BigLittleMoEWrapper(
                layer.mlp, hidden_dim=hidden,
                rank=cfg.rank, lora_alpha=cfg.lora_alpha,
                num_experts=cfg.num_experts, top_k=cfg.top_k,
                gamma=cfg.gamma if args.mode == "lora" else 0.1,
                streaming=True, device=cfg.device,
                param_dtype=cfg.param_dtype, compute_dtype=cfg.compute_dtype,
            )
            if args.mode == "swiglu":
                layer.mlp.lora_pool = nn.ModuleList(
                    SwiGLUMicroExpert(hidden, 512, cfg.compute_dtype).to("cpu")
                    for _ in range(cfg.num_experts)
                )
                for p in layer.mlp.lora_pool.parameters():
                    p.data = p.data.pin_memory()

    delta = torch.cuda.memory_allocated() / 1024**2 - baseline_vram
    total = n_layers * cfg.num_experts
    if args.mode == "lora":
        per_mb = 2 * cfg.rank * hidden * 2 / 1024**2
    else:
        per_mb = 3 * hidden * 512 * 2 / 1024**2
    print(f"[+] Patched {total} micro-experts")
    print(f"    - added VRAM      : {delta:.2f} MB")
    print(f"    - host RAM pool   : ~{per_mb * total:.2f} MB "
          f"({per_mb * 1024:.0f} KB each)")

    res = generate(
        model, tokenizer, PROMPT, device=cfg.device,
        gen_cfg=GenConfig(max_new_tokens=args.max_new_tokens),
    )
    print(f"\n[+] {res['tokens']} tokens in {res['elapsed_s'] * 1000:.0f} ms "
          f"-> {res['tokens_per_s']:.2f} tokens/s")
    print("[+] (Experts are randomly initialised; output quality is meaningless.)")
    print(f"    {res['text']!r}")


if __name__ == "__main__":
    main()
