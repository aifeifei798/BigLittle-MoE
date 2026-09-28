"""The two-tier MoE layer: a pinned GPU big core plus a host-RAM expert pool.

Two forward modes share one implementation:

``streaming=True`` (inference)
    Experts live in page-locked host RAM. Each step routes on the newest
    activation, queues the top-k experts onto a dedicated CUDA stream while the
    big core is still computing, then does a single sync before use.

``streaming=False`` (training)
    Experts live in VRAM and are evaluated *densely* -- every expert receives
    gradient on every step. Routing is a differentiable softmax rather than a
    hard index, which is what makes the exported Top-8 ensemble meaningful.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn

from .experts import LoRAMicroExpert


class BigLittleMoEWrapper(nn.Module):
    """Wraps a dense MLP with a router and a pool of LoRA micro-experts."""

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        rank: int = 16,
        lora_alpha: float = 16.0,
        num_experts: int = 32,
        top_k: int = 8,
        gamma: float = 0.3,
        streaming: bool = False,
        device: str = "cuda:0",
        param_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gamma = gamma
        self.scaling = lora_alpha / rank
        self.streaming = streaming
        self.device = device
        self.compute_dtype = compute_dtype
        self._param_dtype = param_dtype

        # Tier 1: pretrained dense backbone, permanently resident in VRAM.
        self.big_core = original_mlp

        # Router head runs on the GPU in both modes. Training keeps it in fp32
        # (it is optimised); inference casts to compute dtype so the matmul
        # matches the bf16 activations coming out of the big core.
        self.router = nn.Linear(
            hidden_dim,
            num_experts,
            bias=False,
            device=device,
            dtype=compute_dtype if streaming else param_dtype,
        )
        nn.init.normal_(self.router.weight, std=0.02)

        if streaming:
            # Tier 2: experts live in page-locked host RAM so the DMA engine can
            # fetch them without CPU involvement.
            pool = nn.ModuleList(
                LoRAMicroExpert(
                    hidden_dim,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    dtype=compute_dtype,
                ).to("cpu")
                for _ in range(num_experts)
            )
            for p in pool.parameters():
                p.data = p.data.pin_memory()
        else:
            pool = nn.ModuleList(
                LoRAMicroExpert(
                    hidden_dim,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    dtype=param_dtype,
                ).to(device)
                for _ in range(num_experts)
            )

        self.lora_pool = pool
        self.transfer_stream = torch.cuda.Stream(device=device) if streaming else None

        # --- telemetry -------------------------------------------------
        self.expert_counter: Counter[int] = Counter()
        self.last_selected: list[int] = []
        self.last_router_probs: torch.Tensor | None = None
        self.last_router_logits: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def reset_stats(self) -> None:
        self.expert_counter.clear()

    def num_active_experts(self) -> int:
        return len(self.last_selected)

    def set_streaming(self, streaming: bool) -> None:
        """Move the expert pool between VRAM (training) and pinned RAM (inference)."""
        if streaming == self.streaming:
            return
        pool_dtype = self.compute_dtype if streaming else self._param_dtype
        self.lora_pool.to(device="cpu" if streaming else self.device, dtype=pool_dtype)
        self.router.to(dtype=pool_dtype)
        if streaming:
            for p in self.lora_pool.parameters():
                p.data = p.data.pin_memory()
            if self.transfer_stream is None:
                self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.streaming = streaming

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.streaming:
            return self._forward_streaming(x)
        return self._forward_dense(x)

    # ------------------------------------------------------------------
    # Training path: dense, differentiable, every expert gets gradient.
    # ------------------------------------------------------------------
    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        big_out = self.big_core(x)

        bsz, seqlen, hidden = x.shape
        flat = x.reshape(-1, hidden)

        router_logits = self.router(flat).view(bsz, seqlen, self.num_experts)
        probs = torch.softmax(router_logits, dim=-1)
        self.last_router_logits = router_logits
        self.last_router_probs = probs

        # Stacking keeps this differentiable w.r.t. the ModuleList parameters
        # while costing only a couple of MB of copies.
        w_a = torch.stack([e.lora_A.weight for e in self.lora_pool], dim=0)  # [E, r, H]
        w_b = torch.stack([e.lora_B.weight for e in self.lora_pool], dim=0)  # [E, H, r]

        # Project every token into every expert's rank-r space: [N, E, r].
        h = torch.einsum("nh,erh->ner", flat, w_a)
        # Fold the LoRA scaling and the routing gate in *before* expanding back
        # to hidden size, so the [N, E, H] tensor is never materialised.
        h = h * (probs.reshape(-1, self.num_experts, 1) * self.scaling)

        out = h.new_zeros((flat.shape[0], hidden))
        for e in range(self.num_experts):
            out = out + h[:, e, :] @ w_b[e].t()

        return big_out + self.gamma * out.view(bsz, seqlen, hidden)

    # ------------------------------------------------------------------
    # Inference path: top-k routing + real asynchronous host->device DMA.
    # ------------------------------------------------------------------
    def _forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        # Tier 1 is issued first and runs on the default stream; everything in
        # _fetch_experts below is issued on transfer_stream and therefore
        # overlaps with it.
        big_out = self.big_core(x)

        router_logits = self.router(x[:, -1:, :])
        topk_scores, topk_indices = torch.topk(
            router_logits, k=self.top_k, dim=-1
        )
        topk_probs = torch.softmax(topk_scores, dim=-1)

        selected = topk_indices[0, -1].tolist()
        weights = topk_probs[0, -1]
        self.last_selected = selected
        for eid in selected:
            self.expert_counter[eid] += 1

        experts = self._fetch_experts(selected)
        current = x[:, -1:, :]
        lora_out = torch.zeros_like(current)
        for weight, expert in zip(weights, experts):
            lora_out = lora_out + weight * expert(current)

        return big_out + self.gamma * lora_out

    def _fetch_experts(self, selected: list[int]) -> list[nn.Module]:
        """Queue all top-k experts onto the transfer stream, then sync once.

        Doing one batched wait instead of one wait per expert is what lets the
        DMA engine keep the link busy; the copies themselves overlap with the
        big-core matmuls still in flight on the default stream.
        """
        with torch.cuda.stream(self.transfer_stream):
            experts = []
            for eid in selected:
                gpu_expert = self.lora_pool[eid].to(self.device, non_blocking=True)
                # The buffers were allocated on transfer_stream but are consumed
                # on the default stream; without this the caching allocator may
                # recycle them underneath us.
                for p in gpu_expert.parameters():
                    p.record_stream(torch.cuda.current_stream())
                experts.append(gpu_expert)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        return experts


# ----------------------------------------------------------------------
# Routing losses used during training
# ----------------------------------------------------------------------
def domain_target_distribution(
    domain_ids: torch.Tensor, domains: tuple[tuple[str, int, int], ...], num_experts: int
) -> torch.Tensor:
    """Uniform distribution over the expert cluster owning each sample's domain.

    ``domain_ids`` is [B]; the result is [B, num_experts] and sums to 1 per row.
    """
    bsz = domain_ids.shape[0]
    target = torch.zeros(bsz, num_experts, device=domain_ids.device)
    for d_id, (_name, lo, hi) in enumerate(domains):
        mask = domain_ids == d_id
        if mask.any():
            width = hi - lo
            target[mask, lo:hi] = 1.0 / width
    return target


def kl_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(target || softmax(logits)), averaged over tokens."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return (target * (torch.log(target.clamp_min(1e-9)) - log_probs)).sum(-1).mean()


def load_balancing_loss(probs: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer auxiliary loss: E * sum_i f_i * P_i.

    ``probs`` is [N, E] router probabilities. ``f_i`` is the fraction of tokens
    whose argmax lands on expert i, ``P_i`` the mean router probability. The
    product is minimised (value 1.0) only when usage is uniform across all E.

    Note: because routing here is deliberately *domain-clustered*, a correctly
    trained router concentrates on ~8 of 32 experts and therefore sits near
    ``8 / 32 * 32 = 4.0``, not 1.0. This term guards against collapse onto a
    single expert inside a cluster; it is not a global-uniformity score.
    """
    num_experts = probs.shape[-1]
    frac_tokens = torch.zeros(
        num_experts, device=probs.device, dtype=probs.dtype
    ).scatter_add_(
        0,
        probs.argmax(dim=-1),
        torch.ones_like(probs[:, 0]),
    )
    frac_tokens = frac_tokens / probs.shape[0]
    mean_probs = probs.mean(dim=0)
    return num_experts * torch.sum(frac_tokens * mean_probs)
