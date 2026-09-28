import torch
import torch.nn as nn

from biglittle_moe.modules import BigLittleMoEWrapper


class ToyMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.proj(x)


def make_wrapper(num_experts=8, top_k=4, hidden=32, gamma=0.3, streaming=False):
    mlp = ToyMLP(hidden)
    return BigLittleMoEWrapper(
        mlp,
        hidden_dim=hidden,
        rank=4,
        lora_alpha=4.0,
        num_experts=num_experts,
        top_k=top_k,
        gamma=gamma,
        streaming=streaming,
        device="cpu",
        param_dtype=torch.float32,
        compute_dtype=torch.float32,
    )


# ----------------------------------------------------------------------
# The regression this project actually had: only 3 of 32 experts were
# reachable during training, so the rest stayed exactly zero and Top-8
# bought nothing over Top-1.
# ----------------------------------------------------------------------
def test_every_expert_receives_gradient_in_training_mode():
    w = make_wrapper(num_experts=8)
    x = torch.randn(2, 5, 32)
    w(x).sum().backward()

    dead = [
        e
        for e, expert in enumerate(w.lora_pool)
        if expert.lora_B.weight.grad is None
        or expert.lora_B.weight.grad.abs().sum() == 0
    ]
    assert dead == [], f"experts received no gradient: {dead}"


def test_every_router_column_receives_gradient():
    """The router is supervised by the explicit routing loss, not the LM loss.

    At step 0 every ``lora_B`` is zero, so the LM path cannot reach the router
    at all -- the gradient w.r.t. the gate is proportional to the expert output.
    Training therefore drives the router with ``kl_to_target`` directly. This
    checks the objective that is actually optimised.
    """
    from biglittle_moe.modules import kl_to_target

    w = make_wrapper(num_experts=8)
    x = torch.randn(2, 5, 32)
    w(x)

    logits = w.last_router_logits.reshape(-1, 8)
    # 8 experts configured, so the Code cluster is experts 0..1.
    target = torch.zeros(logits.shape[0], 8)
    target[:, 0:2] = 0.5

    kl_to_target(logits, target).backward()

    grad = w.router.weight.grad
    assert grad is not None
    assert grad.abs().sum(dim=0).min() > 0, "some router output units are dead"


def test_router_becomes_reachable_through_lm_loss_after_warmup():
    """Once lora_B is non-zero, the LM path also trains the router."""
    w = make_wrapper(num_experts=8)
    # Simulate post-first-step weights.
    for e in w.lora_pool:
        nn.init.normal_(e.lora_B.weight, std=0.05)
    w(torch.randn(2, 5, 32)).sum().backward()
    assert w.router.weight.grad.abs().sum(dim=0).min() > 0


# ----------------------------------------------------------------------
# Forward-pass correctness
# ----------------------------------------------------------------------
def test_dense_forward_equals_explicit_weighted_sum():
    """out == big_core(x) + gamma * sum_e softmax(router(x))_e * expert_e(x)."""
    torch.manual_seed(0)
    w = make_wrapper(num_experts=6, hidden=16)
    # Break the zero-init so the expert branches actually contribute.
    for e in w.lora_pool:
        nn.init.normal_(e.lora_B.weight, std=0.1)

    x = torch.randn(2, 3, 16)
    got = w(x)

    flat = x.reshape(-1, 16)
    probs = torch.softmax(w.router(flat), dim=-1)
    expected = torch.zeros_like(got)
    big = w.big_core(x)
    for b in range(2):
        for t in range(3):
            n = b * 3 + t
            acc = torch.zeros(16)
            for e, expert in enumerate(w.lora_pool):
                acc = acc + probs[n, e] * expert(flat[n : n + 1])[0]
            expected[b, t] = big[b, t] + w.gamma * acc

    assert torch.allclose(got, expected, atol=1e-5)


def test_gamma_zero_reduces_to_big_core():
    w = make_wrapper(gamma=0.0, hidden=16)
    for e in w.lora_pool:
        nn.init.normal_(e.lora_B.weight, std=0.1)
    x = torch.randn(2, 3, 16)
    assert torch.allclose(w(x), w.big_core(x), atol=1e-6)


def test_zero_init_experts_leave_model_unchanged():
    w = make_wrapper(gamma=1.0, hidden=16)
    x = torch.randn(2, 3, 16)
    assert torch.allclose(w(x), w.big_core(x), atol=1e-6)


def test_shape_preserved():
    w = make_wrapper(hidden=24)
    assert w(torch.randn(3, 7, 24)).shape == (3, 7, 24)


def test_top_k_clamped_to_pool_size():
    w = make_wrapper(num_experts=4, top_k=99)
    assert w.top_k == 4
