# BigLittle-MoE: Breaking the VRAM Wall with Hierarchical Big Core & Micro-Expert Streaming

[![GitHub](https://img.shields.io/badge/GitHub-BigLittle--MoE-181717?style=flat&logo=github&logoColor=white)](https://github.com/aifeifei798/BigLittle-MoE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BigLittle--MoE-yellow?style=flat)](https://huggingface.co/aifeifei798/BigLittle-MoE)
[![Framework](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat)](https://opensource.org/licenses/Apache-2.0)

> All numbers below were measured on an **RTX 5090 D** with the refactored
> codebase (`v0.2.0`). Where a claim could not be reproduced it is called out
> explicitly rather than quietly dropped.

---

## 1. Executive Summary

Deploying frontier-scale Mixture-of-Experts (MoE) models locally runs into the
**VRAM capacity wall**. Conventional offloading fails because swapping
multi-gigabyte expert weights over PCIe is far too slow to hide behind compute.

**BigLittle-MoE** borrows the asymmetric two-tier idea from CPU big.LITTLE
designs:

1. **Tier 1 -- GPU-resident big core.** The pretrained dense Qwen MLP, pinned in
   VRAM, anchoring syntax, reasoning and fluency at zero transfer cost.
2. **Tier 2 -- host-RAM micro-expert pool.** 896 rank-16 LoRA micro-experts
   (~64 KB each) living in page-locked DDR, streamed on demand.
3. **Top-8 collaborative routing.** Each layer dispatches the top-8
   micro-experts per token and blends them with a normalised weight.

### Measured results

| Metric | Value |
| :--- | :--- |
| Trained micro-experts | **896** (32 x 28 layers), all with non-zero weight |
| Host RAM pool (bf16) | **56.00 MB** (~64 KB / expert) |
| Checkpoint on disk | 58.30 MB |
| Added VRAM over baseline | **1.75 MB** (router heads only) |
| Top-8 throughput | **22.4 tokens/s** mean across 3 domains |
| Top-1 throughput | 36.5 tokens/s mean |
| Training time | **8.4 min** for 8,000 samples / 500 optimizer steps |
| Training throughput | 15.8 samples/s |
| Per-expert stream latency | 0.214 ms (p50 0.209, p95 0.238) |

---

## 2. Architecture

```text
                  ┌────────────────────────────────────────────────────────┐
                  │                       GPU VRAM                         │
                  │  ┌──────────────────────────────────────────────────┐  │
                  │  │          Attention Layer + LayerNorm             │  │
                  │  ├──────────────────────────────────────────────────┤  │
                  │  │   Tier-1: Big Core (pinned in VRAM)              │  │
                  │  │   [Pretrained Qwen dense FFN]                    │  │  Zero PCIe cost
                  │  ├──────────────────────────────────────────────────┤  │
                  │  │   Router Head (per-layer gating)                 │  │
                  │  └──────────┬──────────────────────────┬────────────┘  │
                  └─────────────┼──────────────────────────┼───────────────┘
                                │                          │
                  Intra-VRAM    │                          │  async DMA
                  co-execution  ▼                          ▼  (Top-8 in flight)
                        ┌──────────────┐          ┌────────────────────────────────────────┐
                        │ Dense Output │          │            Host System RAM             │
                        └──────┬───────┘          │  ┌────────┐ ┌────────┐     ┌────────┐  │
                                               │  │LoRA  01│ │LoRA  02│ ... │LoRA 896│  │
                               │                  │  └────────┘ └────────┘     └────────┘  │
                               ▼                  └────────────────────────────────────────┘
                     [ Final Layer Output ]
```

### Formulation

$$\mathbf{y} = \text{FFN}^{\text{Big}}(\mathbf{x}) + \gamma \sum_{i \in \text{Top-}8} \omega_i \cdot \text{Expert}_{i}^{\text{Little}}(\mathbf{x})$$

* $\omega_i = \text{Softmax}(\text{Top-}8(W_{\text{router}} x_{[-1,:]})_i)$
* $\text{Expert}_i(\mathbf{x}) = W_B^{(i)} W_A^{(i)} \mathbf{x} \cdot \frac{\alpha}{r}$
* $\gamma = 0.3$

---

## 3. Routing: what the previous version got wrong

An earlier revision hard-wired expert selection during training:

```python
if domain == 0:      expert = lora_pool[0]     # Code
elif domain == 1:    expert = lora_pool[8]     # Math
else:                expert = lora_pool[16]    # Writing
```

Only **3 of 32** experts per layer ever received gradient, and because `lora_B`
is zero-initialised the other 29 produced *exactly zero* forever. Top-8 routing
therefore bought nothing over Top-1, and the "896 trained experts" headline was
really 84.

The current implementation fixes this:

* **Dense training forward.** Every expert is evaluated for every token with a
  differentiable softmax gate, so all 32 experts per layer receive gradient.
  Verified by `tests/test_modules.py::test_every_expert_receives_gradient` and by
  the post-training norm report (`dead=0/32` on sampled layers).
* **Domain supervision without index hard-coding.** The routing target is a
  uniform distribution over the sample's 8-expert cluster, applied as a KL term.
* **Load balancing.** A Switch-Transformer auxiliary loss discourages collapse
  onto a single expert within a cluster.
* **Matched formats.** Training now renders text with the same ChatML template
  used at inference, and masks prompt tokens to `-100` so loss covers the
  response only.

Routing quality is now essentially perfect -- each domain activates its own
cluster and nothing else:

```text
   💻 Code /     99.8%   (code prompt)
   🧮 Math /    100.0%   (math prompt)
   ✍️ Writing / 100.0%   (writing prompt)
```

---

## 4. Data & training

8,000 domain-aligned instruction pairs:

| Cluster | Experts | Source | Samples |
| :--- | :--- | :--- | :--- |
| Code & Algorithms | #00-#07 | `iamtarun/python_code_instructions_18k_alpaca` | 2,000 |
| Math & Reasoning | #08-#15 | `openai/gsm8k` (calculator annotations stripped) | 2,000 |
| General Writing | #16-#31 | `HuggingFaceH4/no_robots` | 4,000 |

Training config: bf16 compute with **fp32 master weights**, seq len 512,
micro-batch 4 x grad-accum 4 (effective 16), AdamW `lr=1e-3`, `wd=0.01`,
aux-loss weight 0.01, grad-norm clipping at 1.0.

Loss curve from the shipped run:

```text
[step 025/500] lm 1.7135 | route 0.6419 | aux 1.4990 | 14.8 samples/s
[step 250/500] lm 1.6535 | route 0.0716 | aux 1.5614 | 15.6 samples/s
[step 500/500] lm 1.5656 | route 0.0452 | aux 1.5281 | 15.8 samples/s
```

> The `aux` value hovers near 1.5 rather than 1.0. That is expected here:
> routing is *deliberately* domain-clustered, so a healthy router concentrates on
> ~8 of 32 experts. The term guards against intra-cluster collapse; it is not a
> global-uniformity score.

---

## 5. Evaluation

```bash
python -m biglittle_moe.evaluate --top-k 8
```

Measured (200 new tokens per prompt):

| Domain | Top-8 | Top-1 | Latency (Top-8) |
| :--- | --- | --- | --- |
| Code | 21.25 tok/s | 33.34 tok/s | 9410 ms |
| Math | 22.98 tok/s | 37.84 tok/s | 8703 ms |
| Writing | 23.05 tok/s | 38.33 tok/s | 8677 ms |
| **mean** | **22.43 tok/s** | 36.50 tok/s | |

> **On the Top-8 headline.** The original README claimed Top-8 ran at 20.8-22.6
> tok/s and implied a benefit over Top-1. Both halves were wrong: the original
> throughputs were computed as `45 / elapsed` regardless of how many tokens were
> actually produced, and Top-8 could not have beaten Top-1 because the extra 7
> experts were zero. With honest counting, **Top-1 is genuinely faster than
> Top-8** (36.5 vs 22.4 tok/s) -- routing cost grows linearly in k. Top-8 remains
> the default because it is the setting that exercises collaborative routing, not
> because it is faster. Whether the extra capacity is worth ~14 tok/s is a real
> open question.

---

## 6. Why PCIe does not choke

1. **Compact payloads.** Each expert is $W_A \in \mathbb{R}^{16 \times 1024}$ and
   $W_B \in \mathbb{R}^{1024 \times 16}$ in bf16:
   $$(16 \cdot 1024 + 1024 \cdot 16) \cdot 2 = 65{,}536 \text{ bytes} \approx 64 \text{ KB}$$
   All 896 = 56.00 MB.
2. **Pinned DMA.** Experts are allocated with `pin_memory()`, so the DMA engine
   moves them without CPU involvement.
3. **Batched stream + one sync.** The top-k copies are queued together on a
   dedicated CUDA stream *while the big-core matmuls are still in flight* on the
   default stream, then synchronised once. The earlier code issued a copy and an
   immediate `wait_stream` per expert, which serialised the transfers and
   destroyed the overlap. Buffers are also passed through `record_stream` to
   stop the caching allocator recycling them early.

Measured end-to-end stream latency in `benchmarks/toy_layer.py` (4096 hidden,
2 big cores + 16 streamed SwiGLU experts):

```text
[+] Stream latency: mean 0.214 ms | p50 0.209 ms | p95 0.238 ms
```

---

## 7. Quickstart

Requires an NVIDIA GPU with ~4 GB free VRAM and Python 3.10+.

```bash
git clone https://github.com/aifeifei798/BigLittle-MoE.git
cd BigLittle-MoE

# uv (recommended)
uv venv && uv pip install -e ".[dev]"

# or plain pip
pip install -e ".[dev]"
```

Then:

```bash
python -m biglittle_moe.prepare      # download + assemble 8,000 samples
python -m biglittle_moe.train        # train 896 experts (~8.5 min on a 5090 D)
python -m biglittle_moe.chat         # interactive streaming chat
python -m biglittle_moe.evaluate     # domain benchmark
```

The original top-level script names still work and forward into the package:

```bash
python prepare_data.py
python train_mole.py
python chat_mole_8_stream_en.py
python test_trained_mole_8.py
```

Benchmarks (no training, no language model):

```bash
python benchmarks/toy_layer.py --big-cores 2
python benchmarks/qwen_plumbing.py --mode lora
```

---

## 8. Project layout

```text
src/biglittle_moe/
  config.py      single source of truth: model id, rank, top-k, gamma, domains
  experts.py     LoRAMicroExpert
  modules.py     BigLittleMoEWrapper -- dense (train) and streaming (infer) paths
  model.py       load / patch / checkpoint I/O
  domains.py     expert -> domain mapping and telemetry rendering
  data.py        corpus assembly + ChatML dataset with prompt masking
  train.py       training loop with routing + auxiliary losses
  generate.py    generation with accurate token accounting
  chat.py        interactive terminal
  evaluate.py    domain benchmark suite
  prepare.py     data download CLI
benchmarks/      toy_layer.py, qwen_plumbing.py
tests/           21 unit tests
```

Configuration is centralised in `config.py`; `test_trained_mole.py` and
`test_trained_mole_8.py` are now thin shims over one implementation with a
`--top-k` flag, since they differed only in that value.

---

## 9. Hardware notes

| Platform | Topology | Feasibility | Speed |
| :--- | :--- | :--- | :--- |
| RTX 4060 / 5090 | Discrete GPU + PCIe | Pinned host DMA | 22 tok/s (Top-8) / 36 tok/s (Top-1) |
| Laptops, 4 GB VRAM | Shared VRAM | Fits in < 1.3 GB | untested |
| Apple Silicon (M2-M4) | Unified memory | Zero-copy pointer swap | untested |

> The original table quoted 40-80 tok/s figures for Apple Silicon and mobile SoCs.
> **Those were projections, never measurements** -- this project has not been
> run on any of those platforms, and no such hardware was available. Treat them
> as untested.

---

## 10. Citation

```bibtex
@misc{biglittle_moe_2026,
  author = {aifeifei798 and Community Contributors},
  title = {BigLittle-MoE: Breaking the VRAM Wall with Hierarchical Big Core and Micro-Expert Streaming},
  year = {2026},
  publisher = {GitHub and Hugging Face},
  howpublished = {\url{https://github.com/aifeifei798/BigLittle-MoE}}
}
```
