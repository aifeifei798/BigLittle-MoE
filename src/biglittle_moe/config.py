"""Single source of truth for every tunable in BigLittle-MoE.

Every magic number that used to be hard-coded across the ten top-level scripts
lives here exactly once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Domain layout
# --------------------------------------------------------------------------
# Each layer owns `NUM_EXPERTS` micro-experts, partitioned into contiguous
# domain clusters. During training every expert in the target cluster receives
# gradient; at inference the router picks the top-k of the whole pool.

DOMAINS: tuple[tuple[str, int, int], ...] = (
    ("Code", 0, 8),
    ("Math", 8, 16),
    ("Writing", 16, 32),
)

DOMAIN_ID: dict[str, int] = {name: i for i, (name, _, _) in enumerate(DOMAINS)}
DOMAIN_NAMES: tuple[str, ...] = tuple(name for name, _, _ in DOMAINS)


def domain_of_expert(expert_id: int) -> str:
    """Map a flat expert index to its domain label."""
    for name, lo, hi in DOMAINS:
        if lo <= expert_id < hi:
            return name
    raise ValueError(f"expert id {expert_id} outside configured clusters {DOMAINS}")


def cluster_indices(domain: str) -> list[int]:
    """Expert indices belonging to ``domain``."""
    for name, lo, hi in DOMAINS:
        if name == domain:
            return list(range(lo, hi))
    raise KeyError(f"unknown domain {domain!r}; expected one of {DOMAIN_NAMES}")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "mole_train_data.jsonl"
WEIGHTS_PATH = PROJECT_ROOT / "biglittle_moe_896e_weights.pt"


# --------------------------------------------------------------------------
# Model / architecture
# --------------------------------------------------------------------------
@dataclass
class MoEConfig:
    """Architecture hyper-parameters for the two-tier MoE."""

    model_id: str = "Qwen/Qwen3-0.6B"
    rank: int = 16
    lora_alpha: float = 16.0
    num_experts: int = 32
    top_k: int = 8
    #: Residual blending coefficient gamma for the streamed expert ensemble.
    gamma: float = 0.3
    #: Master-weight dtype. Compute runs under bf16 autocast.
    param_dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda:0"

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.rank

    def to_dict(self) -> dict:
        d = asdict(self)
        d["param_dtype"] = str(self.param_dtype)
        d["compute_dtype"] = str(self.compute_dtype)
        return d


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters for expert training."""

    max_length: int = 512
    micro_batch: int = 4
    grad_accum: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.01
    #: Weight of the Switch-Transformer style load-balancing auxiliary loss.
    aux_loss_weight: float = 0.01
    log_every: int = 25
    num_workers: int = 4
    seed: int = 1234

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum


@dataclass
class GenConfig:
    """Default sampling parameters for inference."""

    max_new_tokens: int = 600
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    do_sample: bool = True


#: Number of decoder layers in the reference base model (Qwen3-0.6B).
#: Overridable so the package can be pointed at another backbone.
NUM_LAYERS: int = int(os.environ.get("BIGLITTLE_NUM_LAYERS", "28"))


@dataclass
class Experiment:
    """Bundle used by the CLI entry points."""

    moe: MoEConfig = field(default_factory=MoEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    data_path: Path = DATA_PATH
    weights_path: Path = WEIGHTS_PATH
    num_layers: int = NUM_LAYERS

    @property
    def total_experts(self) -> int:
        return self.num_layers * self.moe.num_experts
