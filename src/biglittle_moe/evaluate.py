"""Domain benchmark suite over the trained expert pool.

Replaces ``test_trained_mole.py`` (Top-1) and ``test_trained_mole_8.py``
(Top-8) with a single script whose top-k is a flag.
"""

from __future__ import annotations


import torch

from .config import Experiment, GenConfig
from .domains import collect_model_counter, render_dashboard
from .generate import generate
from .model import build_experiment, load_expert_weights

PROMPTS = (
    ("Code", "Write a Python function to check if a number is prime."),
    ("Math", "A car travels 120 miles in 2 hours. What is its average speed "
             "in miles per hour?"),
    ("Writing", "Describe the serene beauty of a quiet mountain lake at dawn."),
)


def evaluate(top_k: int | None = None, max_new_tokens: int = 200) -> dict:
    exp = Experiment()
    if top_k is not None:
        exp.moe.top_k = top_k
    if not exp.weights_path.exists():
        raise SystemExit(f"Checkpoint '{exp.weights_path.name}' not found.")

    model, tokenizer, _ = build_experiment(exp, streaming=True)
    load_expert_weights(model, exp.weights_path)
    model.eval()

    gen_cfg = GenConfig(max_new_tokens=max_new_tokens)
    results = []

    for expected, prompt in PROMPTS:
        for layer in model.model.layers:
            layer.mlp.reset_stats()

        res = generate(
            model, tokenizer, prompt,
            gen_cfg=gen_cfg,
            device=exp.moe.device,
            show_routing=True,
        )
        res["expected"] = expected
        results.append(res)

        print("=" * 75)
        print(f"[{expected}] \"{prompt}\"")
        print("=" * 75)
        print(f"    {res['tokens']} tokens in {res['elapsed_s'] * 1000:.0f} ms "
              f"-> {res['tokens_per_s']:.2f} tokens/s")
        print("-" * 75)
        print(res["text"])
        print("-" * 75)
        for li, summary in res["routing"].items():
            print(f"    layer {li:02d}: {summary}")
        print(render_dashboard(collect_model_counter(model)))
        print()

    _report_summary(results, exp.moe.top_k)
    return {"results": results}


def _report_summary(results: list[dict], top_k: int) -> None:
    print("=" * 75)
    print(f"Summary (top_k={top_k})")
    print("=" * 75)
    for r in results:
        print(f"    {r['expected']:<8} {r['tokens_per_s']:6.2f} tok/s  "
              f"{r['tokens']:4d} tokens  {r['elapsed_s'] * 1000:7.0f} ms")
    mean = sum(r["tokens_per_s"] for r in results) / len(results)
    print(f"    {'mean':<8} {mean:6.2f} tok/s")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="BigLittle-MoE domain benchmark")
    ap.add_argument("--top-k", type=int, default=None,
                    help="active experts per layer (default: from config, 8)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(0)
    evaluate(top_k=args.top_k, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
