"""Rank-16 LoRA micro-expert: the unit that lives in host RAM.

At hidden_dim=1024 a single expert is (16*1024 + 1024*16) * 2 bytes = 64 KB, so
896 of them fit in ~56 MB of DDR without touching VRAM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRAMicroExpert(nn.Module):
    """A single low-rank residual branch, ``lora_B(lora_A(x)) * scaling``."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 16,
        lora_alpha: float = 16.0,
        dtype: torch.dtype = torch.float32,
        zero_init_b: bool = True,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, dtype=dtype)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        if zero_init_b:
            # Standard LoRA init: the branch starts as an exact no-op so the
            # grafted model reproduces the base model bit-for-bit at step 0.
            nn.init.zeros_(self.lora_B.weight)
        else:
            nn.init.normal_(self.lora_B.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling
