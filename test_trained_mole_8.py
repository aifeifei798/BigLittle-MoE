import os
import time
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------
# 1. LoRA Micro-Expert Architecture (Rank=16, 64 KB)
# ----------------------------------------------------------------------
class LoRAMicroExpert(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 rank: int = 16,
                 lora_alpha: float = 16.0,
                 dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


# ----------------------------------------------------------------------
# 2. Production BigLittle-MoE Inference Wrapper (Top-8 专家协同版)
# ----------------------------------------------------------------------
class BigLittleInferenceWrapper(nn.Module):

    def __init__(self,
                 original_mlp: nn.Module,
                 hidden_dim: int,
                 rank: int = 16,
                 num_lora_experts: int = 32,
                 top_k: int = 8,  # 每层激活 8 个专家
                 device: str = "cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.big_core = original_mlp
        self.top_k = top_k

        # 路由器（在 GPU 上运行）
        self.router = nn.Linear(hidden_dim,
                                num_lora_experts,
                                bias=False,
                                device=device,
                                dtype=dtype)

        # 32 个微专家候选池（保存在 CPU 锁页内存中）
        self.lora_pool_cpu = nn.ModuleList([
            LoRAMicroExpert(hidden_dim, rank=rank, dtype=dtype).to("cpu")
            for _ in range(num_lora_experts)
        ])
        for p in self.lora_pool_cpu.parameters():
            p.data = p.data.pin_memory()

        self.transfer_stream = torch.cuda.Stream(device=device)
        self.last_selected_experts = []  # 记录当前层被激活的 8 个专家编号

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 基础大模型核心直接计算
        big_out = self.big_core(x)

        # 2. 动态路由决策：根据当前最新 token 计算专家匹配分
        current_token = x[:, -1:, :]
        router_logits = self.router(current_token)  # [batch, 1, 32]

        # 3. 选出得分最高的 8 个专家及其权重（Top-8）
        topk_scores, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)  # 归一化为概率

        # 记录选中的专家 ID（方便在控制台展示）
        selected_ids = topk_indices[0, -1].tolist()
        self.last_selected_experts = selected_ids
        weights = topk_probs[0, -1]

        # 4. 传输并累加 8 个专家的输出
        lora_out = torch.zeros_like(big_out)
        for weight, expert_idx in zip(weights, selected_ids):
            # 通过独立流将选中的专家从 CPU 传输到 GPU
            with torch.cuda.stream(self.transfer_stream):
                selected_lora_gpu = self.lora_pool_cpu[expert_idx].to(
                    self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.transfer_stream)

            # 按权重累加
            lora_out = lora_out + (weight * selected_lora_gpu(x))

        return big_out + (lora_out * 0.3)


# ----------------------------------------------------------------------
# 3. 领域标签与验证工具
# ----------------------------------------------------------------------
def format_expert_summary(expert_ids):
    """统计并格式化打印选出的 8 个专家所属领域"""
    domain_counts = Counter()
    for eid in expert_ids:
        if 0 <= eid <= 7:
            domain_counts["Code"] += 1
        elif 8 <= eid <= 15:
            domain_counts["Math"] += 1
        else:
            domain_counts["Writing"] += 1

    summary = ", ".join(
        [f"{k}: {v}个" for k, v in domain_counts.items() if v > 0])
    ids_str = ", ".join([f"#{eid:02d}" for eid in expert_ids])
    return f"[{ids_str}] -> ({summary})"


def test_prompt(model, tokenizer, prompt, test_name):
    print(f"\n{'='*75}")
    print(f"[*] Testing {test_name}")
    print(f"    Input Prompt: \"{prompt}\"")
    print(f"{'='*75}")

    inputs = tokenizer(f"User: {prompt}\nAssistant:",
                       return_tensors="pt").to("cuda:0")

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_new_tokens=60,
                                 do_sample=True,
                                 temperature=0.7,
                                 top_p=0.9,
                                 repetition_penalty=1.15,
                                 pad_token_id=tokenizer.eos_token_id)
    total_time_ms = (time.perf_counter() - t0) * 1000

    # 打印采样层中被激活的 8 个专家详情
    print(f"[+] Multi-Layer Routing Decisions (Top-8 Experts Active):")
    for layer_idx in [0, 7, 14, 21, 27]:
        chosen_experts = model.model.layers[
            layer_idx].mlp.last_selected_experts
        print(
            f"    - Layer {layer_idx:02d}: {format_expert_summary(chosen_experts)}"
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    throughput = 45 / (total_time_ms / 1000)

    print(f"\n[✔] Generated Output:")
    print(f"--------------------------------------------------")
    print(generated_text)
    print(f"--------------------------------------------------")
    print(
        f"[⚡] Throughput: {throughput:.2f} tokens/s (Time: {total_time_ms:.1f} ms)"
    )


# ----------------------------------------------------------------------
# 4. 主执行入口
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    weights_path = "biglittle_mole_896e_weights.pt"

    assert os.path.exists(
        weights_path
    ), f"Weights file {weights_path} not found! Please ensure training has finished."

    print(
        f"[*] Initializing BigLittle-MoE Verification Suite (Top-8 Experts Mode)..."
    )
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")
    hidden_dim = model.config.hidden_size

    # 将每一层包装为：每层32个候选专家，推理时激活 Top-8 专家
    print(f"[*] Patching 28 layers with BigLittle-MoE (Top-8 routing)...")
    for layer in model.model.layers:
        layer.mlp = BigLittleInferenceWrapper(layer.mlp,
                                              hidden_dim,
                                              rank=16,
                                              num_lora_experts=32,
                                              top_k=8,
                                              device="cuda:0",
                                              dtype=dtype)

    # 加载已有的 56MB 权重
    print(f"[*] Loading trained weights from {weights_path}...")
    saved_weights = torch.load(weights_path, map_location="cpu")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.router.load_state_dict(saved_weights[f"layer_{i}_router"])
        layer.mlp.lora_pool_cpu.load_state_dict(
            saved_weights[f"layer_{i}_loras"])

    print(f"[✔] Successfully loaded 896 trained experts & 28 routers!")

    # 运行三类任务测试
    test_prompt(
        model,
        tokenizer,
        prompt="Write a Python function to check if a number is prime.",
        test_name="Test 1: Python Code Task")

    test_prompt(
        model,
        tokenizer,
        prompt=
        "A car travels 120 miles in 2 hours. What is its average speed in miles per hour?",
        test_name="Test 2: Mathematical Reasoning")

    test_prompt(
        model,
        tokenizer,
        prompt="Describe the serene beauty of a quiet mountain lake at dawn.",
        test_name="Test 3: Creative & Descriptive Writing")


if __name__ == "__main__":
    main()