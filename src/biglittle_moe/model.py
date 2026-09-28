"""Base-model loading, in-place grafting, and expert checkpoint I/O."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment, MoEConfig
from .modules import BigLittleMoEWrapper


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def eos_token_ids(tokenizer) -> list[int]:
    """Base EOS plus ChatML ``<|im_end|>`` when the vocab defines it."""
    ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id and im_end not in ids:
        ids.append(im_end)
    return ids


def load_base_model(cfg: MoEConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=cfg.compute_dtype,
        device_map=cfg.device,
    )
    for p in model.parameters():
        p.requires_grad = False
    return model


def patch_model(model, cfg: MoEConfig, streaming: bool) -> None:
    """Replace every decoder layer's MLP with the two-tier wrapper."""
    hidden_dim = model.config.hidden_size
    for layer in model.model.layers:
        layer.mlp = BigLittleMoEWrapper(
            layer.mlp,
            hidden_dim=hidden_dim,
            rank=cfg.rank,
            lora_alpha=cfg.lora_alpha,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            gamma=cfg.gamma,
            streaming=streaming,
            device=cfg.device,
            param_dtype=cfg.param_dtype,
            compute_dtype=cfg.compute_dtype,
        )


def build_experiment(
    exp: Experiment | None = None,
    streaming: bool = False,
    verbose: bool = True,
):
    """Load + patch + return ``(model, tokenizer, num_layers)``."""
    exp = exp or Experiment()
    cfg = exp.moe

    if verbose:
        print(f"[*] Loading base model {cfg.model_id} ...")
    t0 = time.perf_counter()
    tokenizer = load_tokenizer(cfg.model_id)
    model = load_base_model(cfg)

    num_layers = len(model.model.layers)
    if verbose:
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[+] Loaded in {time.perf_counter() - t0:.1f}s | "
              f"{num_layers} layers, hidden={model.config.hidden_size}, "
              f"VRAM {vram:.1f} MB")

    patch_model(model, cfg, streaming=streaming)
    if verbose:
        _report_pool(model, num_layers, streaming, cfg)
    return model, tokenizer, num_layers


def _report_pool(model, num_layers: int, streaming: bool, cfg: MoEConfig) -> None:
    total = num_layers * cfg.num_experts
    if streaming:
        mb = total * (2 * cfg.rank * model.config.hidden_size) * 2 / 1024**2
        print(f"[+] Expert pool: {total} micro-experts in pinned host RAM ~{mb:.2f} MB")
    else:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[+] Expert pool: {total} micro-experts in VRAM | "
              f"trainable {n_train / 1e6:.2f} M params")


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_expert_weights(model, path: Path, dtype: torch.dtype | None = None) -> None:
    """Export routers + expert pool.

    Training keeps fp32 master weights; the export is cast down to ``dtype``
    (bf16 by default) because inference runs in bf16 and it halves the file.
    """
    def cast(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu().to(dtype) if dtype is not None else t.detach().cpu()

    state: dict[str, dict] = {}
    for i, layer in enumerate(model.model.layers):
        state[f"layer_{i}_router"] = {
            k: cast(v) for k, v in layer.mlp.router.state_dict().items()
        }
        state[f"layer_{i}_loras"] = {
            k: cast(v) for k, v in layer.mlp.lora_pool.state_dict().items()
        }
    torch.save(state, path)


def load_expert_weights(model, path: Path, strict: bool = True) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.router.load_state_dict(saved[f"layer_{i}_router"], strict=strict)
        layer.mlp.lora_pool.load_state_dict(saved[f"layer_{i}_loras"], strict=strict)


def trainable_parameters(model) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for layer in model.model.layers:
        params.extend(layer.mlp.router.parameters())
        params.extend(layer.mlp.lora_pool.parameters())
    for p in params:
        p.requires_grad = True
    return params
