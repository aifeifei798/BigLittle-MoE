"""Shared generation helper with accurate token accounting.

The original scripts reported ``45 / elapsed`` regardless of how many tokens
were actually produced. Here the count always comes from the output tensor.
"""

from __future__ import annotations

import time

import torch
from transformers import TextIteratorStreamer

from .config import GenConfig
from .model import eos_token_ids


def count_new_tokens(outputs, prompt_len: int) -> int:
    return int(outputs.shape[1]) - int(prompt_len)


def generate(
    model,
    tokenizer,
    prompt: str,
    gen_cfg: GenConfig | None = None,
    device: str = "cuda:0",
    show_routing: bool = False,
    sample_layers=(0, 7, 14, 21, 27),
) -> dict:
    """Run a single completion and return text plus real throughput."""
    gen_cfg = gen_cfg or GenConfig()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            eos_token_id=eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - t0

    n_new = count_new_tokens(outputs, prompt_len)
    text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    routing = {}
    if show_routing:
        from .domains import format_expert_ids

        for li in sample_layers:
            if li < len(model.model.layers):
                routing[li] = format_expert_ids(model.model.layers[li].mlp.last_selected)

    return {
        "text": text,
        "tokens": n_new,
        "elapsed_s": elapsed,
        "tokens_per_s": n_new / elapsed if elapsed > 0 else 0.0,
        "routing": routing,
    }


def make_streamer(tokenizer) -> TextIteratorStreamer:
    return TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
