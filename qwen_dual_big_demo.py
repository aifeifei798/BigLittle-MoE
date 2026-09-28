
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
import time

# ----------------------------------------------------------------------
# 1. 轻量级小核 (Micro-Expert) 结构
# ----------------------------------------------------------------------
class MicroExpert(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

# ----------------------------------------------------------------------
# 2. Qwen 双大核 + 内存微专家包装器 (Dual Big Core Wrapper)
# ----------------------------------------------------------------------
class DualBigLittleQwenMLPWrapper(nn.Module):
    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        little_dim: int = 512,
        num_little_experts: int = 16,
        device: str = "cuda:0",
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # ==============================================================
        # Tier 1: 显存常驻双大核 (VRAM Dual Big Cores)
        # ==============================================================
        # [大核 1]：Qwen 原始预训练好的通用 MLP (守住语言常识)
        self.big_core_1 = original_mlp

        # [大核 2]：克隆的原厂大核，用于承载高维专项知识 (同样常驻 GPU 显存)
        self.big_core_2 = copy.deepcopy(original_mlp)

        # [大核门控]：在 GPU 上对 2 个大核做动态加权
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)

        # ==============================================================
        # Tier 2: 内存微专家池 (Host RAM Micro-Expert Pool)
        # ==============================================================
        self.router_little = nn.Linear(hidden_dim, num_little_experts, bias=False, device=device, dtype=dtype)
        
        self.little_cores_cpu = nn.ModuleList([
            MicroExpert(hidden_dim, little_dim, dtype=dtype).to("cpu")
            for _ in range(num_little_experts)
        ])
        for param in self.little_cores_cpu.parameters():
            param.data = param.data.pin_memory()

        self.transfer_stream = torch.cuda.Stream(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_token = x[:, -1:, :]

        # --------------------------------------------------------------
        # 1. 双大核显存内并行协同运算 (Zero PCIe Overhead)
        # --------------------------------------------------------------
        big_logits = self.router_big(current_token)
        big_weights = torch.softmax(big_logits, dim=-1)  # [batch, 1, 2]
        
        w1 = big_weights[:, :, 0:1]
        w2 = big_weights[:, :, 1:2]

        # 双大核在显存里全速并行，加权合流
        big_out = (w1 * self.big_core_1(x)) + (w2 * self.big_core_2(x))

        # --------------------------------------------------------------
        # 2. 内存微专家挑选与 PCIe 动态流式拉取 (DMA Streaming)
        # --------------------------------------------------------------
        little_logits = self.router_little(current_token)
        selected_idx = torch.argmax(little_logits, dim=-1).item()

        selected_little_cpu = self.little_cores_cpu[selected_idx]
        selected_little_gpu = selected_little_cpu.to(self.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # 小核特化微调补充 (系数 0.1)
        little_out = selected_little_gpu(x) * 0.1

        # 最终汇总
        return big_out + little_out

# ----------------------------------------------------------------------
# 3. 加载 Qwen 并打补丁
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    print(f"[*] Loading base model {model_id} to GPU...")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda:0"
    )

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)
    print(f"[+] Base model loaded: {num_layers} layers, hidden_dim = {hidden_dim}")

    baseline_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"[+] Single-Core Qwen Baseline VRAM: {baseline_vram:.2f} MB")

    # 逐层替换为【双大核 + 内存微专家】架构
    print(f"[*] Upgrading layers to DUAL BIG CORES (VRAM) + Micro-Experts (Host RAM)...")
    for i, layer in enumerate(model.model.layers):
        layer.mlp = DualBigLittleQwenMLPWrapper(
            original_mlp=layer.mlp,
            hidden_dim=hidden_dim,
            little_dim=512,          # 每个小核维度 512
            num_little_experts=16,   # 每层 16 个内存小核
            device="cuda:0",
            dtype=dtype
        )

    dual_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_delta = dual_vram - baseline_vram
    print(f"[+] Dual Big Core Upgrade Complete!")
    print(f"    - Total VRAM Allocated : {dual_vram:.2f} MB")
    print(f"    - Added VRAM Delta     : {vram_delta:.2f} MB (28 Second Big Cores + Routers)")
    print(f"    - Micro-Experts in RAM : 448 experts (Consuming 0 MB VRAM)")

    # ------------------------------------------------------------------
    # 4. 执行真实端到端推理生成
    # ------------------------------------------------------------------
    prompt = "Artificial Intelligence will change the future because"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    print(f"\n[+] Running Dual-Big-Core text generation benchmark...")
    print(f"    Prompt: \"{prompt}\"")
    
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    total_time_ms = (time.perf_counter() - t0) * 1000

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    throughput = 40 / (total_time_ms / 1000)

    print(f"\n[✔] Dual-Big-Core Generated Output:")
    print(f"--------------------------------------------------")
    print(generated_text)
    print(f"--------------------------------------------------")
    print(f"[⚡] Generated 40 tokens in {total_time_ms:.2f} ms (Throughput: {throughput:.2f} tokens/s)")

if __name__ == "__main__":
    main()