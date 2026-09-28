"""Domain corpus assembly and the supervised dataset.

Two things matter here and both were wrong in the original script:

* Training text is rendered with the *same* ChatML template the chat client
  uses at inference time.
* Loss is computed on the response only -- prompt tokens are masked to -100
  instead of being modelled as targets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Corpus definition: (domain_id, hf_dataset, config, split, n, adapter)
CORPORA = (
    (0, "iamtarun/python_code_instructions_18k_alpaca", None, "train[:2000]",
     lambda r: (r["instruction"] + ("\n" + r["input"] if r.get("input") else ""), r["output"])),
    (1, "openai/gsm8k", "main", "train[:2000]",
     lambda r: (r["question"], _clean_gsm8k(r["answer"]))),
    (2, "HuggingFaceH4/no_robots", None, "train[:4000]",
     lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])),
)

_CALC = re.compile(r"<<[^>]*>>")


def _clean_gsm8k(answer: str) -> str:
    """Strip calculator annotations and normalise the final-answer marker."""
    text = _CALC.sub("", answer).strip()
    if "####" in text:
        reasoning, _, final = text.rpartition("####")
        text = f"{reasoning.strip()}\nThe final answer is {final.strip()}."
    return text


def build_corpus(output_path: Path) -> int:
    """Download the three domain clusters and write a unified JSONL file."""
    from datasets import load_dataset

    print("[*] Pulling domain corpora from Hugging Face ...")
    records: list[dict] = []
    for domain_id, name, cfg, split, adapt in CORPORA:
        print(f"    - {name} ({split})")
        ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
        for row in ds:
            prompt, response = adapt(row)
            if not prompt or not response:
                continue
            records.append(
                {
                    "domain_id": domain_id,
                    "domain_name": ("Code", "Math", "Writing")[domain_id],
                    "prompt": prompt,
                    "response": response,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / 1024**2
    print(f"[+] Wrote {len(records):,} samples to {output_path.name} ({size_mb:.1f} MB)")
    return len(records)


class MoLEDataset(Dataset):
    """ChatML-rendered instruction pairs with prompt tokens masked out."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def _render(self, prompt: str, response: str) -> tuple[list[int], int]:
        """Return (token ids, number of leading prompt tokens to mask)."""
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        return full_ids[: self.max_length], min(len(prompt_ids), self.max_length)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        input_ids, n_prompt = self._render(item["prompt"], item["response"])

        ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros(self.max_length, dtype=torch.long)
        labels = torch.full((self.max_length,), -100, dtype=torch.long)

        n = len(input_ids)
        ids[:n] = torch.tensor(input_ids, dtype=torch.long)
        mask[:n] = 1
        labels[n_prompt:n] = torch.tensor(input_ids[n_prompt:], dtype=torch.long)

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "domain_id": torch.tensor(item["domain_id"], dtype=torch.long),
        }
