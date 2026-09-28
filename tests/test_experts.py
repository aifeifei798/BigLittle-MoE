import pytest
import torch

from biglittle_moe.config import DOMAINS
from biglittle_moe.experts import LoRAMicroExpert
from biglittle_moe.modules import (
    domain_target_distribution,
    kl_to_target,
    load_balancing_loss,
)


def test_expert_is_identity_at_init():
    """Zero-init lora_B makes a fresh grafted model reproduce the base model."""
    torch.manual_seed(0)
    expert = LoRAMicroExpert(64, rank=8)
    x = torch.randn(4, 64)
    assert torch.allclose(expert(x), torch.zeros_like(x))


def test_expert_footprint():
    expert = LoRAMicroExpert(1024, rank=16, dtype=torch.bfloat16)
    nbytes = sum(p.numel() * p.element_size() for p in expert.parameters())
    assert nbytes == 65_536


def test_expert_non_zero_after_b_update():
    torch.manual_seed(0)
    expert = LoRAMicroExpert(32, rank=4, zero_init_b=False)
    x = torch.randn(2, 32)
    assert not torch.allclose(expert(x), torch.zeros_like(x))


def test_domain_target_is_uniform_over_cluster():
    ids = torch.tensor([0, 1, 2])
    target = domain_target_distribution(ids, DOMAINS, num_experts=32)
    assert target.shape == (3, 32)
    assert torch.allclose(target.sum(-1), torch.ones(3))
    # Code row: mass only on experts 0..7, evenly.
    assert torch.allclose(target[0, 0:8], torch.full((8,), 1 / 8))
    assert target[0, 8:].sum() == 0
    assert torch.allclose(target[2, 16:32], torch.full((16,), 1 / 16))


def test_kl_to_target_zero_when_aligned():
    target = torch.tensor([[0.5, 0.5]])
    assert kl_to_target(torch.log(target), target).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_positive_when_misaligned():
    target = torch.tensor([[1.0, 0.0]])
    assert kl_to_target(torch.zeros(1, 2), target).item() > 0


def test_load_balancing_loss_minimal_when_uniform():
    probs = torch.full((64, 8), 1 / 8)
    assert load_balancing_loss(probs).item() == pytest.approx(1.0, abs=1e-5)


def test_load_balancing_loss_high_when_collapsed():
    """The point of the aux term: penalise a router that always picks one expert."""
    collapsed = torch.zeros(64, 8)
    collapsed[:, 0] = 1.0
    uniform = torch.full((64, 8), 1 / 8)
    assert load_balancing_loss(collapsed).item() > 7.0
    assert load_balancing_loss(collapsed).item() > load_balancing_loss(uniform).item()
