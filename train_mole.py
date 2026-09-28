import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

# ----------------------------------------------------------------------
# 1. High-Throughput Dataset Loader
# ----------------------------------------------------------------------
class MoLEDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        domain_id = item["domain_id"]
        
        text = f"User: {item['prompt']}\nAssistant: {item['response']}"
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "domain_id": torch.tensor(domain_id, dtype=torch.long)
        }

# ----------------------------------------------------------------------
# 2. Trainable MoLE Layer Definition (Clean Autograd without In-place assignment)
# ----------------------------------------------------------------------
class TrainableLoRAMicroExpert(nn.Module):
    def __init__(self, hidden_dim, rank=16, lora_alpha=16.0, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class TrainableBigLittleLoRAWrapper(nn.Module):
    def __init__(self, original_mlp, hidden_dim, rank=16, num_experts=32, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.big_core = original_mlp
        self.num_experts = num_experts
        self.current_domain_id = None
        self.last_router_logits = None

        self.router = nn.Linear(hidden_dim, num_experts, bias=False, device=device, dtype=dtype)
        self.lora_pool = nn.ModuleList([
            TrainableLoRAMicroExpert(hidden_dim, rank=rank, device=device, dtype=dtype)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        with torch.no_grad():
            big_out = self.big_core(x)

        router_logits = self.router(x)
        self.last_router_logits = router_logits

        # Supervised Domain Routing via clean concatenation
        if self.current_domain_id is not None:
            batch_size = x.size(0)
            lora_outs = []
            for b in range(batch_size):
                d_id = self.current_domain_id[b].item()
                if d_id == 0:
                    exp_idx = 0    # Code cluster
                elif d_id == 1:
                    exp_idx = 8    # Math cluster
                else:
                    exp_idx = 16   # General Writing cluster
                lora_outs.append(self.lora_pool[exp_idx](x[b:b+1]))
            lora_out = torch.cat(lora_outs, dim=0)
        else:
            weights = torch.softmax(router_logits, dim=-1)
            top1_idx = torch.argmax(weights, dim=-1)
            lora_out = self.lora_pool[top1_idx[0, -1].item()](x)

        return big_out + lora_out

# ----------------------------------------------------------------------
# 3. Memory-Safe Training Loop (Batch 4 + Grad Accum 4 = Effective Batch 16)
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    print(f"[*] Initializing Memory-Optimized Pipeline on NVIDIA RTX 5090 D...")
    print(f"[*] Base model: {model_id}")

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cuda:0")

    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)

    print(f"[*] Grafting 32 LoRA micro-experts onto all {num_layers} layers (896 total)...")
    for layer in model.model.layers:
        layer.mlp = TrainableBigLittleLoRAWrapper(
            layer.mlp,
            hidden_dim,
            rank=16,
            num_experts=32,
            device="cuda:0",
            dtype=dtype
        )

    trainable_params = []
    for layer in model.model.layers:
        trainable_params.extend([p for p in layer.mlp.router.parameters() if p.requires_grad])
        trainable_params.extend([p for p in layer.mlp.lora_pool.parameters() if p.requires_grad])

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"[+] Isolation complete! Trainable params: {total_trainable / 1e6:.2f} M (~56 MB)")

    # Safe batching: Physical Batch = 4, Accumulate = 4 -> Effective Batch = 16
    MICRO_BATCH = 4
    GRAD_ACCUM_STEPS = 4
    MAX_LENGTH = 512
    print(f"[*] Configuration: Micro Batch = {MICRO_BATCH}, Accum Steps = {GRAD_ACCUM_STEPS} (Effective Batch = 16), Max Length = {MAX_LENGTH}")

    dataset = MoLEDataset("mole_train_data.jsonl", tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
    criterion_router = nn.CrossEntropyLoss()

    total_steps = len(dataloader) // GRAD_ACCUM_STEPS
    print(f"\n[+] Starting Training (Total Optimizer Updates: {total_steps})...")
    model.train()
    
    start_time = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda:0", non_blocking=True)
        labels = batch["labels"].to("cuda:0", non_blocking=True)
        domain_id = batch["domain_id"].to("cuda:0", non_blocking=True)

        for layer in model.model.layers:
            layer.mlp.current_domain_id = domain_id

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss

        # Router alignment loss
        router_loss = 0.0
        target_cluster = torch.where(domain_id == 0, 0, torch.where(domain_id == 1, 8, 16))
        for layer in model.model.layers:
            logits = layer.mlp.last_router_logits[:, 0, :]
            router_loss += criterion_router(logits, target_cluster)

        # Scale loss for gradient accumulation
        total_loss = (lm_loss + 0.1 * (router_loss / num_layers)) / GRAD_ACCUM_STEPS
        total_loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

            global_step = (step + 1) // GRAD_ACCUM_STEPS
            if global_step % 25 == 0 or global_step == total_steps:
                elapsed = time.time() - start_time
                current_loss = total_loss.item() * GRAD_ACCUM_STEPS
                speed = ((step + 1) * MICRO_BATCH) / elapsed
                print(f"    [Step {global_step:03d}/{total_steps}] Loss: {current_loss:.4f} | Speed: {speed:.1f} samples/s | Elapsed: {elapsed:.1f}s")

    print(f"\n[✔] Training Complete on RTX 5090 D! Total duration: {(time.time() - start_time)/60:.2f} minutes")

    save_path = "biglittle_mole_896e_weights.pt"
    print(f"[*] Saving 896 trained micro-experts to {save_path}...")
    
    state_to_save = {}
    for i, layer in enumerate(model.model.layers):
        state_to_save[f"layer_{i}_router"] = layer.mlp.router.state_dict()
        state_to_save[f"layer_{i}_loras"] = layer.mlp.lora_pool.state_dict()

    torch.save(state_to_save, save_path)
    print(f"[✔] Successfully exported {save_path}! (~56 MB)")

if __name__ == "__main__":
    main()