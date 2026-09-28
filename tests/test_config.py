import pytest

from biglittle_moe.config import (
    DOMAINS,
    MoEConfig,
    cluster_indices,
    domain_of_expert,
)


def test_domains_partition_the_pool():
    """Every expert index belongs to exactly one cluster, no gaps or overlaps."""
    covered = []
    for name, lo, hi in DOMAINS:
        covered.extend(range(lo, hi))
    assert covered == list(range(32))
    assert len(covered) == len(set(covered))


def test_domain_of_expert_boundaries():
    assert domain_of_expert(0) == "Code"
    assert domain_of_expert(7) == "Code"
    assert domain_of_expert(8) == "Math"
    assert domain_of_expert(15) == "Math"
    assert domain_of_expert(16) == "Writing"
    assert domain_of_expert(31) == "Writing"
    with pytest.raises(ValueError):
        domain_of_expert(32)


def test_cluster_indices():
    assert cluster_indices("Code") == list(range(0, 8))
    assert cluster_indices("Writing") == list(range(16, 32))
    with pytest.raises(KeyError):
        cluster_indices("Nope")


def test_scaling():
    cfg = MoEConfig(rank=16, lora_alpha=32.0)
    assert cfg.scaling == 2.0


def test_expert_headline_arithmetic():
    """32 experts x 28 layers = 896; each rank-16 expert is ~64 KB at H=1024."""
    cfg = MoEConfig()
    assert 28 * cfg.num_experts == 896
    per_expert_bytes = 2 * cfg.rank * 1024 * 2  # two matrices, bf16
    assert per_expert_bytes == 65_536
    assert 896 * per_expert_bytes / 1024**2 == pytest.approx(56.0, abs=0.1)
