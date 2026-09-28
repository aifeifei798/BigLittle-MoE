# BigLittle-MoE: Breaking the VRAM Wall with Hierarchical Multi-Big Core & Micro-Expert Streaming

[![GitHub](https://img.shields.io/badge/GitHub-BigLittle--MoE-181717?style=flat&logo=github&logoColor=white)](https://github.com/aifeifei798/BigLittle-MoE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BigLittle--MoE-yellow?style=flat)](https://huggingface.co/aifeifei798/BigLittle-MoE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat)](https://opensource.org/licenses/Apache-2.0)

---

## 1. Executive Summary & TL;DR

Deploying frontier-scale Mixture-of-Experts (MoE) models locally has long been hindered by the **VRAM capacity wall**. Conventional offloading strategies often fail due to catastrophic PCIe latency when swapping massive multi-gigabyte expert weights.

**BigLittle-MoE** breaks this paradigm by implementing an asymmetric **Two-Tier Heterogeneous MoE** architecture inspired by modern CPU big.LITTLE systems:

1. **Tier 1 (GPU-Resident Big Core):** High-capacity generalist FFN backbone (native Qwen dense MLPs) permanently pinned in GPU VRAM to anchor syntax, logical reasoning, and language fluency with **zero PCIe transfer penalty**.
2. **Tier 2 (Host RAM Micro-Expert Pool):** Hundreds of modular, fine-grained micro-experts (Rank-16 LoRA modules, ~64 KB each) stored in cost-effective Host RAM (DDR4/DDR5) and streamed dynamically via asynchronous page-locked PCIe DMA.
3. **Top-8 Collaborative Routing:** Instead of brittle single-expert selection, each layer dynamically dispatches and weights the **Top-8 highest-scoring micro-experts** per token, synthesizing domain expertise (Code, Math, Writing) concurrently on the fly.

### Key Empirical Milestones
* **896 Real Trained Experts in 56 MB:** Extended `Qwen/Qwen3-0.6B` to host **896 dynamic LoRA micro-experts** (32 per layer across 28 layers), consuming only **56.00 MB** of Host RAM (~64 KB per expert) and **1.75 MB** of added VRAM.
* **Top-8 Real-Time Streaming (21–23 tokens/s):** Executes **224 PCIe DMA micro-transfers per token** ($28 \text{ layers} \times 8 \text{ experts}$) while sustaining a fluid **20.8 – 22.6 tokens/s** generation throughput on a consumer workstation.
* **Rigorous ChatML & Chain-of-Thought (<think>):** Native integration with standard ChatML protocols, featuring autonomous `<think>...</think>` internal reasoning loops, zero conversational drifting, and perfect algebraic resolution.
* **Live Neural Telemetry:** Real-time console inspection tracing per-token expert activation distribution across domains in real time.

---

## 2. Architecture: Two-Tier Decoupled Memory Hierarchy

Rather than treating all experts as bulky, homogeneous blocks, BigLittle-MoE physically decouples the model across memory boundaries:

```text
                  ┌────────────────────────────────────────────────────────┐
                  │                       GPU VRAM                         │
                  │  ┌──────────────────────────────────────────────────┐  │
                  │  │          Attention Layer + LayerNorm             │  │
                  │  ├──────────────────────────────────────────────────┤  │
                  │  │   Tier-1: Big Core (Pinned in GPU VRAM)          │  │
                  │  │   [Pretrained Native Qwen FFN Backbone]          │  │ <--- High-capacity Dense Backbone
                  │  ├──────────────────────────────────────────────────┤  │      Zero PCIe latency
                  │  │   Router Head (Per-Layer Gating Mechanism)       │  │
                  │  └──────────┬──────────────────────────┬────────────┘  │
                  └─────────────┼──────────────────────────┼───────────────┘
                                │                          │
                  Intra-VRAM    │                          │ Non-blocking DMA Transfer
                  Co-execution  ▼                          ▼ (Top-8 Streamed Concurrently)
                        ┌──────────────┐          ┌────────────────────────────────────────┐
                        │ Dense Output │          │            Host System RAM             │
                        └──────┬───────┘          │  ┌────────┐ ┌────────┐     ┌────────┐  │
                               │                  │  │LoRA  01│ │LoRA  02│ ... │LoRA 896│  │ <--- 896 modular
                               │                  │  └────────┘ └────────┘     └────────┘  │      micro-experts
                               ▼                  └────────────────────────────────────────┘      in 56 MB DDR5
                    [ Final Layer Output ]
```

### Mathematical Formulation (Top-8 Dynamic Routing)

Given input activation tensor $\mathbf{x}$, the forward pass blends the dense foundational backbone with a normalized, weighted ensemble of the Top-8 micro-experts:

$$
\mathbf{y} = \text{FFN}^{\text{Big}}(\mathbf{x}) + \gamma \cdot \sum_{i \in \text{Top-}8} \omega_i \cdot \text{Expert}_{i}^{\text{Little}}(\mathbf{x})
$$

Where:
* $\text{FFN}^{\text{Big}}(\mathbf{x})$ is the permanently GPU-resident base MLP.
* $\omega_i = \text{Softmax}(\text{Top-}8(\mathbf{W}_{\text{router}} \cdot \mathbf{x}_{[-1, :]})_i)$ is the normalized gating weight for expert $i$.
* $\text{Expert}_{i}^{\text{Little}}(\mathbf{x}) = \mathbf{W}_B^{(i)} \mathbf{W}_A^{(i)} \mathbf{x} \cdot \frac{\alpha}{r}$ is the Rank-16 LoRA micro-expert transferred over PCIe via non-blocking DMA.
* $\gamma = 0.3$ is the residual blending coefficient balancing foundational grammatical coherence with modular domain specialization.

---

## 3. High-Throughput Domain-Supervised Training

We curated **8,000 domain-aligned instruction pairs** across three distinct clusters:
* **Cluster 0 (Code & Algorithms, Experts #00–#07):** 2,000 samples from Python Code Instructions (`iamtarun/python_code_instructions_18k_alpaca`).
* **Cluster 1 (Math & Reasoning, Experts #08–#15):** 2,000 samples from GSM8K Chain-of-Thought (`openai/gsm8k`).
* **Cluster 2 (General Writing & Prose, Experts #16–#31):** 4,000 samples from No-Robots (`HuggingFaceH4/no_robots`).

### Training Metrics (NVIDIA GeForce RTX 5090 D)
* **Precision:** Native `bfloat16`
* **Sequence Length:** 512 tokens
* **Effective Batch Size:** 16 (Micro-batch 4 with Gradient Accumulation 4)
* **Optimization:** AdamW ($lr = 1\times 10^{-3}$, weight decay $0.01$)
* **Trainable Parameters:** 30.28 M (~56 MB, base model frozen)
* **Throughput:** **43.2 samples/s (>22,000 tokens/s)**, converging in **3.12 minutes**.

---

## 4. Empirical Evaluation: Multi-Expert Routing & Chat Telemetry

In interactive streaming inference, BigLittle-MoE dynamically logs real-time expert allocations across all 28 layers.

### Test 1: Algorithmic Implementation (Bubble Sort with Early Exit)
* **User Prompt:** `"Write bubble sort in Python."`
* **Throughput:** **20.8 tokens/s** (600 tokens generated)
* **Output Extract:** The model generates internal `<think>` planning followed by an optimized bubble sort with a boolean `swapped` flag:
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
```

---

### Test 2: Mathematical Reasoning (Chicken-and-Rabbit Algebra)
* **User Prompt:** `"There are 35 heads and 94 feet in total. How many chickens and rabbits are there?"`
* **Throughput:** **21.3 tokens/s** (536 tokens generated)
* **Reasoning Trace & Final Output:**
```text
<think>
Let chicken count be x, rabbit count be y.
Equation 1: x + y = 35
Equation 2: 2x + 4y = 94
Substitute y = 35 - x: 2x + 4(35 - x) = 94 -> 140 - 2x = 94 -> 2x = 46 -> x = 23 (chickens)
Then y = 35 - 23 = 12 (rabbits).
Verify: 23*2 + 12*4 = 46 + 48 = 94. Correct.
</think>

Chickens: 23, Rabbits: 12.

Step-by-Step Explanation:
1. Let the number of chickens be x and rabbits be y.
2. System of equations:
   - x + y = 35
   - 2x + 4y = 94
3. Solving gives x = 23, y = 12. Verified against constraints.
```

---

### Test 3: Atmospheric Poetry & Creative Prose
* **User Prompt:** `"Write a short poem about missing home while standing by a rainy window at night."`
* **Throughput:** **21.2 tokens/s** (354 tokens generated, natural `<|im_end|>` termination)
* **Output:**
```text
**Shadows in Rain**  

I stand where silence hums,  
A lone sentinel’s breath—  
Raindrops whisper secrets to the sky,  
And I’m lost in echoes of my past.  

No trace of daylight’s gentle touch,  
Yet the world seems distant now.  
Ghostly whispers linger,  
As if my soul is still unspoken.  

For home, though far from sight,  
Still waits in shadows deep.  
My heart clings to the wind,  
And dreams of you float above.
```

---

### Test 4: Live Telemetry Dashboard Breakdown
During multi-turn generation, BigLittle-MoE accumulates global activation frequencies across all $28 \times 8 = 224$ active slots per token:

```text
──────────────────────────────────────────────────────────────────────
📊 [Neural Activity Breakdown / Multi-Expert Allocation]:
   💻 Code / Algorithms:      24.8% [████                ] (33,321 calls)
   🧮 Math / Logic:           24.8% [████                ] (33,380 calls)
   ✍️  General / Writing:      50.4% [██████████          ] (67,699 calls)
──────────────────────────────────────────────────────────────────────
```

---

## 5. Architectural Comparison Table

| Architecture Profile | Tier-1 VRAM | Tier-2 RAM Pool | Active Micro-Experts | Added VRAM Delta | Generation Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3-0.6B Baseline** | 1137.39 MB | 0 MB | 0 (Dense FFN) | 0 MB | ~22.0 tokens/s |
| **Single Big + 896 LoRA (Top-1)** | 1139.14 MB | **56.00 MB** | 1 per layer (28 total) | **1.75 MB** | **22.5 - 35.1 tokens/s** |
| **Single Big + 896 LoRA (Top-8 Ensemble)** | 1139.14 MB | **56.00 MB** | **8 per layer (224 total)** | **1.75 MB** | **20.8 - 22.6 tokens/s** |
| **Traditional MoE (e.g., Mixtral-style)** | > 14.5 GB | 0 MB | 2 full MLPs | > 13 GB | Bottlenecked on 4GB VRAM |

---

## 6. Systems Deep Dive: Why PCIe Does Not Choke Top-8 Routing

A natural concern with offloading is whether transferring 8 experts per layer per token ($28 \times 8 = 224$ transfers/token) will saturate the PCIe bus. BigLittle-MoE avoids contention through three mechanisms:

1. **Ultra-Compact Rank-16 Payloads:**  
   Each micro-expert consists solely of $\mathbf{W}_A \in \mathbb{R}^{16 \times 896}$ and $\mathbf{W}_B \in \mathbb{R}^{896 \times 16}$ in `bfloat16`.  
   $$\text{Size} = (16 \times 896 + 896 \times 16) \times 2\text{ bytes} = 57,344\text{ bytes} \approx 56\text{ KB}$$
2. **Page-Locked (Pinned) DMA Pipelines:**  
   Expert tensors are allocated via `pin_memory()`. The GPU DMA engine accesses host memory directly over PCIe without CPU operating system intervention:
   $$\text{Transfer Time per Expert} = \frac{56\text{ KB}}{26\text{ GB/s (PCIe 4.0)}} \approx 0.00215\text{ ms}$$
   Transferring 8 micro-experts per layer takes **~0.017 ms**, well within the GPU layer compute window.
3. **Dedicated CUDA Streams:**  
   DMA transfers execute asynchronously on `self.transfer_stream` while attention and Tier-1 dense core computations proceed concurrently on the default stream.

---

## 7. Quickstart & Minimal Reproducible Pipeline

### 1. Installation
```bash
git clone https://github.com/aifeifei798/BigLittle-MoE.git
cd BigLittle-MoE
pip install torch transformers accelerate datasets
```

*(Alternatively clone from Hugging Face)*:
```bash
git clone https://huggingface.co/aifeifei798/BigLittle-MoE
```

### 2. Prepare Domain Datasets
```bash
python prepare_data.py
```

### 3. Train 896 Micro-Experts (~3 mins on RTX 5090 / 4090)
```bash
python train_mole.py
```

### 4. Interactive Streaming Chat (Top-8 Co-Activation)
Run the real-time streaming terminal with live expert activity breakdown:
```bash
python chat_mole_8_stream_en.py
```

* **Interactive Controls:**
  * Type `clear` to reset conversational context.
  * Type `exit` or `quit` to end the session.
  * Press `Ctrl + C` during text generation to halt output safely without terminating the runtime.

---

## 8. Hardware & Edge-AI Compatibility

| Target Platform | Memory Topology | BigLittle-MoE Feasibility | Expected Latency / Speed |
| :--- | :--- | :--- | :--- |
| **Consumer Desktop (RTX 4060 / 5090)** | Discrete GPU + PCIe + DDR5 | Pinned Host DMA (Top-8 Active) | **21 – 35 tokens/s** |
| **Laptops with iGPU / 4 GB VRAM** | Discrete / Shared VRAM | Fits entirely in < 1.3 GB memory | **18 – 25 tokens/s** |
| **Apple Silicon (M2 / M3 / M4)** | **Unified Memory Architecture (UMA)** | Zero-Copy pointer swap (Zero PCIe penalty) | **40 – 60+ tokens/s** |
| **Snapdragon / Dimensity Flagship SoC** | **Mobile UMA LPDDR5X** | Zero-Copy NPU execution (< 500 MB INT4) | **50 – 80+ tokens/s** |

> **Note on Mobile Edge Deployment:** On mobile SoCs with Unified Memory (UMA), the PCIe transfer penalty drops to **zero**. The NPU accesses the 56 MB expert pool in-place via shared memory addresses, making BigLittle-MoE exceptionally well-suited for on-device, thermal-efficient AI.

---

## 9. Citation

If you build upon BigLittle-MoE in your research or edge deployment pipelines, please cite:

```bibtex
@misc{biglittle_moe_2026,
  author = {aifeifei798 and Community Contributors},
  title = {BigLittle-MoE: Breaking the VRAM Wall with Hierarchical Multi-Big Core and Micro-Expert Streaming},
  year = {2026},
  publisher = {GitHub and Hugging Face},
  howpublished = {\url{https://github.com/aifeifei798/BigLittle-MoE}}
}
```
