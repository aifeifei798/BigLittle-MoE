"""Supervised training of the LoRA micro-expert pool.

The important difference from the original script: routing is *learned* rather
than hard-wired to the domain index. Every expert is evaluated densely and
receives gradient on every step, so the exported pool really does contain
``num_layers * num_experts`` trained experts rather than three per layer.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import DOMAINS, Experiment
from .data import MoLEDataset
from .model import build_experiment, save_expert_weights, trainable_parameters
from .modules import domain_target_distribution, kl_to_target, load_balancing_loss


def _expand_domain_ids(domain_ids: torch.Tensor, seqlen: int) -> torch.Tensor:
    """Repeat a per-sample domain id across the sequence dimension."""
    return domain_ids.unsqueeze(1).expand(-1, seqlen).reshape(-1)


def train(exp: Experiment | None = None) -> Path:
    exp = exp or Experiment()
    cfg, tcfg = exp.moe, exp.train

    torch.manual_seed(tcfg.seed)

    model, tokenizer, num_layers = build_experiment(exp, streaming=False)
    params = trainable_parameters(model)
    total_trainable = sum(p.numel() for p in params)
    print(f"[+] Trainable parameters: {total_trainable / 1e6:.2f} M")

    dataset = MoLEDataset(exp.data_path, tokenizer, max_length=tcfg.max_length)
    loader = DataLoader(
        dataset,
        batch_size=tcfg.micro_batch,
        shuffle=True,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    total_steps = max(1, len(loader) // tcfg.grad_accum)
    print(
        f"[*] {len(dataset):,} samples | micro-batch {tcfg.micro_batch} x "
        f"accum {tcfg.grad_accum} = {tcfg.effective_batch} | "
        f"{total_steps} optimizer steps"
    )

    optimizer = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    model.train()
    autocast = torch.autocast(device_type="cuda", dtype=cfg.compute_dtype)

    print("\n[+] Training ...")
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    accum = {"lm": 0.0, "route": 0.0, "aux": 0.0, "n": 0}
    done = False

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        attn = batch["attention_mask"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)
        domain_ids = batch["domain_id"].to(cfg.device, non_blocking=True)

        with autocast:
            outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            lm_loss = outputs.loss

            # Routing supervision: pull every token's distribution toward the
            # cluster that owns its domain, and keep usage spread out.
            seqlen = input_ids.shape[1]
            flat_domain = _expand_domain_ids(domain_ids, seqlen)
            target = domain_target_distribution(flat_domain, DOMAINS, cfg.num_experts)

            route_loss = torch.zeros((), device=cfg.device)
            aux_loss = torch.zeros((), device=cfg.device)
            for layer in model.model.layers:
                logits = layer.mlp.last_router_logits.reshape(-1, cfg.num_experts)
                probs = layer.mlp.last_router_probs.reshape(-1, cfg.num_experts)
                route_loss = route_loss + kl_to_target(logits, target)
                aux_loss = aux_loss + load_balancing_loss(probs)
            route_loss = route_loss / num_layers
            aux_loss = aux_loss / num_layers

            total = lm_loss + route_loss + tcfg.aux_loss_weight * aux_loss

        (total / tcfg.grad_accum).backward()

        for key, val in (("lm", lm_loss), ("route", route_loss), ("aux", aux_loss)):
            accum[key] += float(val.detach())
        accum["n"] += 1

        if (step + 1) % tcfg.grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            gstep = (step + 1) // tcfg.grad_accum
            if gstep % tcfg.log_every == 0 or gstep == total_steps:
                n = max(accum["n"], 1)
                elapsed = time.perf_counter() - start
                speed = ((step + 1) * tcfg.micro_batch) / elapsed
                print(
                    f"    [step {gstep:03d}/{total_steps}] "
                    f"lm {accum['lm'] / n:.4f} | route {accum['route'] / n:.4f} | "
                    f"aux {accum['aux'] / n:.4f} | {speed:.1f} samples/s | "
                    f"{elapsed:.0f}s"
                )
                accum = {"lm": 0.0, "route": 0.0, "aux": 0.0, "n": 0}

        if (step + 1) // tcfg.grad_accum >= total_steps:
            done = True
            break

    minutes = (time.perf_counter() - start) / 60
    print(f"\n[+] Training finished in {minutes:.2f} min (complete={done})")

    save_expert_weights(model, exp.weights_path, dtype=cfg.compute_dtype)
    size_mb = exp.weights_path.stat().st_size / 1024**2
    print(f"[+] Saved {exp.total_experts} trained experts to "
          f"{exp.weights_path.name} ({size_mb:.2f} MB)")
    _report_expert_norms(model, num_layers)
    return exp.weights_path


def _report_expert_norms(model, num_layers: int, sample_layers=(0, 14, 27)) -> None:
    """Sanity check: every expert should now carry non-zero weight."""
    print("\n[*] Per-expert L2 norm of lora_B (expect all > 0):")
    for li in sample_layers:
        pool = model.model.layers[li].mlp.lora_pool
        norms = [float(e.lora_B.weight.detach().float().norm()) for e in pool]
        zeros = sum(1 for v in norms if v == 0.0)
        print(f"    layer {li:02d}: dead={zeros}/{len(norms)}  "
              f"min={min(norms):.3e}  max={max(norms):.3e}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train the LoRA micro-expert pool")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--aux-loss-weight", type=float, default=None)
    args = ap.parse_args()

    exp = Experiment()
    if args.data is not None:
        exp.data_path = args.data
    if args.out is not None:
        exp.weights_path = args.out
    for attr in ("lr", "micro_batch", "grad_accum", "max_length", "aux_loss_weight"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(exp.train, attr, value)

    if not exp.data_path.exists():
        raise SystemExit(
            f"Dataset '{exp.data_path.name}' not found. "
            f"Run `python -m biglittle_moe.prepare` first."
        )
    train(exp)


if __name__ == "__main__":
    main()
