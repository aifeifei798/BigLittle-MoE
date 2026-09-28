"""Expert telemetry: per-domain activation breakdown."""

from __future__ import annotations

from collections import Counter

from .config import DOMAINS, DOMAIN_NAMES, domain_of_expert

_ICONS = {"Code": "\U0001f4bb", "Math": "\U0001f9ee", "Writing": "\u270d\ufe0f"}


def aggregate_counter(counter: Counter[int]) -> dict[str, int]:
    """Fold a flat ``expert_id -> count`` mapping into per-domain totals."""
    totals = {name: 0 for name in DOMAIN_NAMES}
    for eid, count in counter.items():
        totals[domain_of_expert(eid)] += count
    return totals


def collect_model_counter(model) -> Counter[int]:
    """Sum the per-layer expert counters of a patched model."""
    total: Counter[int] = Counter()
    for layer in model.model.layers:
        total.update(layer.mlp.expert_counter)
    return total


def format_expert_ids(expert_ids: list[int]) -> str:
    """``[#03, #07, #12] -> (Code: 2, Math: 1)``"""
    if not expert_ids:
        return "(none)"
    per_domain: Counter[str] = Counter(domain_of_expert(e) for e in expert_ids)
    summary = ", ".join(f"{k}: {v}" for k, v in per_domain.items() if v)
    ids = ", ".join(f"#{e:02d}" for e in expert_ids)
    return f"[{ids}] -> ({summary})"


def render_dashboard(counter: Counter[int], bar_width: int = 20) -> str:
    """Render the multi-expert allocation bar chart."""
    totals = aggregate_counter(counter)
    grand = sum(totals.values())
    if grand == 0:
        return ""

    lines = ["\n" + "\u2500" * 70,
             "\U0001f4ca [Neural Activity Breakdown / Multi-Expert Allocation]:"]
    for name in DOMAIN_NAMES:
        pct = 100.0 * totals[name] / grand
        bar = "\u2588" * int(pct // 5)
        icon = _ICONS.get(name, "\u2022")
        lines.append(
            f"   {icon} {name + ' /':<18} {pct:5.1f}% "
            f"[{bar:<{bar_width}}] ({totals[name]:,} calls)"
        )
    lines.append("\u2500" * 70)
    return "\n".join(lines)


def describe_clusters() -> str:
    """One-line human summary of the expert -> domain layout."""
    return ", ".join(f"{name} #{lo:02d}-#{hi - 1:02d}" for name, lo, hi in DOMAINS)
