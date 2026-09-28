from collections import Counter
import os
from threading import Thread
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


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
# 2. Production BigLittle-MoE Inference Wrapper (Top-8 Routed)
# ----------------------------------------------------------------------
class BigLittleInferenceWrapper(nn.Module):

    def __init__(self,
                 original_mlp: nn.Module,
                 hidden_dim: int,
                 rank: int = 16,
                 num_lora_experts: int = 32,
                 top_k: int = 8,
                 device: str = "cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.big_core = original_mlp
        self.top_k = top_k

        # GPU-based router
        self.router = nn.Linear(hidden_dim,
                                num_lora_experts,
                                bias=False,
                                device=device,
                                dtype=dtype)

        # 32 Micro-experts stored in host pinned RAM
        self.lora_pool_cpu = nn.ModuleList([
            LoRAMicroExpert(hidden_dim, rank=rank, dtype=dtype).to("cpu")
            for _ in range(num_lora_experts)
        ])
        for p in self.lora_pool_cpu.parameters():
            p.data = p.data.pin_memory()

        self.transfer_stream = torch.cuda.Stream(device=device)
        self.token_expert_counter = Counter()

    def reset_stats(self):
        self.token_expert_counter.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        big_out = self.big_core(x)

        current_token = x[:, -1:, :]
        router_logits = self.router(current_token)

        # Dynamic Top-k routing
        topk_scores, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        selected_ids = topk_indices[0, -1].tolist()
        weights = topk_probs[0, -1]

        for eid in selected_ids:
            self.token_expert_counter[eid] += 1

        # Stream & accumulate outputs from the 8 micro-experts
        lora_out = torch.zeros_like(big_out)
        for weight, expert_idx in zip(weights, selected_ids):
            with torch.cuda.stream(self.transfer_stream):
                expert_gpu = self.lora_pool_cpu[expert_idx].to(
                    self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            lora_out = lora_out + (weight * expert_gpu(x))

        return big_out + (lora_out * 0.3)


# ----------------------------------------------------------------------
# 3. Live Expert Activity Telemetry Dashboard
# ----------------------------------------------------------------------
def show_expert_dashboard(model):
    total_counter = Counter()
    for layer in model.model.layers:
        total_counter.update(layer.mlp.token_expert_counter)

    code_calls = sum(total_counter[i] for i in range(0, 8))
    math_calls = sum(total_counter[i] for i in range(8, 16))
    writing_calls = sum(total_counter[i] for i in range(16, 32))
    all_calls = code_calls + math_calls + writing_calls

    if all_calls == 0:
        return

    c_pct = (code_calls / all_calls) * 100
    m_pct = (math_calls / all_calls) * 100
    w_pct = (writing_calls / all_calls) * 100

    print("\n" + "─" * 70)
    print("📊 [Neural Activity Breakdown / Multi-Expert Allocation]:")
    print(
        f"   💻 Code / Algorithms:     {c_pct:5.1f}% [{'█' * int(c_pct // 5):<20}] ({code_calls:,} calls)"
    )
    print(
        f"   🧮 Math / Logic:          {m_pct:5.1f}% [{'█' * int(m_pct // 5):<20}] ({math_calls:,} calls)"
    )
    print(
        f"   ✍️  General / Writing:     {w_pct:5.1f}% [{'█' * int(w_pct // 5):<20}] ({writing_calls:,} calls)"
    )
    print("─" * 70)


# ----------------------------------------------------------------------
# 4. Interactive Chat Loop (Streaming via TextIteratorStreamer)
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    weights_path = "biglittle_mole_896e_weights.pt"

    assert os.path.exists(
        weights_path
    ), f"Weights file '{weights_path}' not found! Please check file path."

    print("=" * 70)
    print("🚀 Initializing BigLittle-MoE Interactive Terminal (Streaming Mode)...")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure proper EOS tokens (<|im_end|> and base EOS)
    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")
    hidden_dim = model.config.hidden_size

    # Wrap all 28 MLP layers
    for layer in model.model.layers:
        layer.mlp = BigLittleInferenceWrapper(layer.mlp,
                                              hidden_dim,
                                              rank=16,
                                              num_lora_experts=32,
                                              top_k=8,
                                              device="cuda:0",
                                              dtype=dtype)

    # Load pre-trained weights
    print(
        f"[*] Mounting 896 micro-experts & 28 routing controllers from '{weights_path}'..."
    )
    saved_weights = torch.load(weights_path, map_location="cpu")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.router.load_state_dict(saved_weights[f"layer_{i}_router"])
        layer.mlp.lora_pool_cpu.load_state_dict(
            saved_weights[f"layer_{i}_loras"])

    print("\n✅ System ready! You can start asking questions.")
    print("👉 Quick commands:")
    print("   - Type 'clear' to reset dialogue history")
    print("   - Type 'exit' or 'quit' to terminate\n")

    # Standard ChatML conversation history
    messages = [{
        "role":
        "system",
        "content":
        "You are a helpful, precise, and thoughtful assistant."
    }]

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Session terminated. Goodbye!")
            break
        if user_input.lower() == "clear":
            messages = [{
                "role":
                "system",
                "content":
                "You are a helpful, precise, and thoughtful assistant."
            }]
            print("🧹 Context memory wiped clean.")
            continue

        # Reset per-token expert counters
        for layer in model.model.layers:
            layer.mlp.reset_stats()

        # Append user message
        messages.append({"role": "user", "content": user_input})

        # Render ChatML template
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        # Set up real-time streamer
        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=600,  # Ample room for CoT + full responses
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=eos_token_ids)

        print("\n🤖 Assistant: ", end="", flush=True)

        # Launch generation in background thread for non-blocking token streaming
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        t0 = time.perf_counter()
        thread.start()

        accumulated_text = ""
        try:
            for text_chunk in streamer:
                print(text_chunk, end="", flush=True)
                accumulated_text += text_chunk
        except KeyboardInterrupt:
            print("\n[Generation interrupted by user]")

        thread.join()
        elapsed_sec = time.perf_counter() - t0

        # Calculate exact token throughput
        gen_token_ids = tokenizer.encode(accumulated_text,
                                         add_special_tokens=False)
        gen_tokens_count = len(gen_token_ids)
        speed = gen_tokens_count / elapsed_sec if elapsed_sec > 0 else 0

        print(
            f"\n\n⚡ Speed: {speed:.1f} tokens/s (Generated {gen_tokens_count} tokens in {elapsed_sec*1000:.0f} ms)"
        )

        # Render expert telemetry dashboard
        show_expert_dashboard(model)

        # Save assistant turn to history
        messages.append({"role": "assistant", "content": accumulated_text})

        # Keep context within sliding window of 3 recent turns
        if len(messages) > 7:
            messages = [messages[0]] + messages[-6:]


if __name__ == "__main__":
    main()
