import json
from datasets import load_dataset

print("[*] 正在从 Hugging Face 自动拉取并装配 MoLE 训练语料...")

# 1. 代码领域 (抽 2000 条 -> 对应 01~04 号 LoRA)
print("    - 正在加载代码数据集 (iamtarun/python_code_instructions_18k_alpaca)...")
ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")

# 2. 数学领域 (抽 2000 条 -> 对应 05~08 号 LoRA)
print("    - 正在加载数学思维链数据集 (openai/gsm8k)...")
ds_math = load_dataset("openai/gsm8k", "main", split="train[:2000]")

# 3. 通用写作与对话 (抽 4000 条 -> 对应 17~24 号 LoRA)
print("    - 正在加载高质量指令集 (HuggingFaceH4/no_robots)...")
ds_general = load_dataset("HuggingFaceH4/no_robots", split="train[:4000]")

unified_data = []

# 装配代码数据 (Domain 0: 对应 01~04 专家)
for item in ds_code:
    prompt = item["instruction"] + ("\n" + item["input"] if item["input"] else "")
    unified_data.append({
        "domain_id": 0,
        "domain_name": "Code",
        "prompt": prompt,
        "response": item["output"]
    })

# 装配数学数据 (Domain 1: 对应 05~08 专家)
for item in ds_math:
    unified_data.append({
        "domain_id": 1,
        "domain_name": "Math",
        "prompt": item["question"],
        "response": item["answer"]
    })

# 装配通用写作数据 (Domain 2: 对应 17~24 专家)
for item in ds_general:
    messages = item["messages"]
    if len(messages) >= 2:
        unified_data.append({
            "domain_id": 2,
            "domain_name": "General_Writing",
            "prompt": messages[0]["content"],
            "response": messages[1]["content"]
        })

output_file = "mole_train_data.jsonl"
print(f"[*] 正在打包写入 {output_file}，共 {len(unified_data)} 条专业对齐数据...")
with open(output_file, "w", encoding="utf-8") as f:
    for entry in unified_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"[✔] 数据装配大功告成！文件大小仅约 15 MB，随时可以拿来开火训练！")
