"""Interactive streaming terminal for the trained expert pool."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Thread

from .config import Experiment
from .domains import collect_model_counter, describe_clusters, render_dashboard
from .generate import make_streamer
from .model import build_experiment, eos_token_ids, load_expert_weights

SYSTEM_PROMPT = "You are a helpful, precise, and thoughtful assistant."
HISTORY_TURNS = 6


def chat(exp: Experiment | None = None) -> None:
    exp = exp or Experiment()
    if not exp.weights_path.exists():
        raise SystemExit(
            f"Checkpoint '{exp.weights_path.name}' not found. "
            f"Run `python -m biglittle_moe.train` first."
        )

    print("=" * 70)
    print("\U0001f680 BigLittle-MoE interactive terminal (Top-%d streaming)"
          % exp.moe.top_k)
    print("=" * 70)

    model, tokenizer, _ = build_experiment(exp, streaming=True)
    print(f"[*] Mounting {exp.total_experts} micro-experts from "
          f"{exp.weights_path.name} ...")
    load_expert_weights(model, exp.weights_path)
    model.eval()

    print(f"[+] Expert clusters: {describe_clusters()}")
    print("\n[+] Ready. Commands: `clear` (reset context), `exit`/`quit`\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    eos_ids = eos_token_ids(tokenizer)

    while True:
        try:
            user_input = input("\n\U0001f464 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            return

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Session terminated.")
            return
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("\U0001f9f9 Context reset.")
            continue

        for layer in model.model.layers:
            layer.mlp.reset_stats()

        messages.append({"role": "user", "content": user_input})
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(exp.moe.device)
        prompt_len = inputs["input_ids"].shape[1]

        streamer = make_streamer(tokenizer)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=exp.gen.max_new_tokens,
            do_sample=exp.gen.do_sample,
            temperature=exp.gen.temperature,
            top_p=exp.gen.top_p,
            repetition_penalty=exp.gen.repetition_penalty,
            eos_token_id=eos_ids,
        )

        print("\n\U0001f916 Assistant: ", end="", flush=True)
        thread = Thread(target=model.generate, kwargs=kwargs, daemon=True)
        t0 = time.perf_counter()
        thread.start()

        chunks: list[str] = []
        try:
            for chunk in streamer:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        thread.join(timeout=30.0)
        elapsed = time.perf_counter() - t0

        answer = "".join(chunks)
        # Count against the streamed text so an interrupted turn reports
        # honestly instead of being discarded.
        n_new = len(tokenizer.encode(answer, add_special_tokens=False))
        print(f"\n\n\u26a1 {n_new / elapsed:.1f} tokens/s "
              f"({n_new} tokens in {elapsed * 1000:.0f} ms, prompt {prompt_len} tok)")

        print(render_dashboard(collect_model_counter(model)))

        messages.append({"role": "assistant", "content": answer})
        if len(messages) > HISTORY_TURNS + 1:
            messages = [messages[0]] + messages[-(HISTORY_TURNS):]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Interactive BigLittle-MoE chat terminal")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    exp = Experiment()
    if args.weights is not None:
        exp.weights_path = args.weights
    if args.top_k is not None:
        exp.moe.top_k = args.top_k
    if args.max_new_tokens is not None:
        exp.gen.max_new_tokens = args.max_new_tokens

    chat(exp)


if __name__ == "__main__":
    main()
