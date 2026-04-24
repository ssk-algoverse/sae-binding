import sys
import os
import json
import gc
import torch
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from _gemma_config import get_preset


def read_relations_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def encode(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)


def get_subset(tokenizer, text_examples, start_idx, end_idx):
    subset = []
    labels = []
    for rec in text_examples[start_idx:end_idx]:
        subset.append(encode(tokenizer, rec["input"]))
        labels.append(encode(tokenizer, rec["label"]).item())
    subset = torch.stack(subset, dim=0)
    return subset, labels


def run_inference(model, dataset, label_tokens, device, batch_size=4):
    correct = 0
    total = 0
    for idx in tqdm(range(0, len(dataset), batch_size), desc="examples"):
        inputs = dataset[idx:idx + batch_size].to(device)
        labels = torch.tensor(label_tokens[idx:idx + batch_size]).to(device)
        with torch.no_grad():
            logits = model(input_ids=inputs).logits
        preds = logits[:, -1, :].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(inputs)
        del inputs, labels, logits, preds
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def main():
    preset_name = os.environ.get("GEMMA_PRESET", "gemma-2-2b")
    preset = get_preset(preset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running eval for preset: {preset_name} on {device}")

    ft_ckpt = preset.get("ft_checkpoint")
    if not (ft_ckpt and os.path.isdir(ft_ckpt)):
        raise RuntimeError(
            f"Preset '{preset_name}' has no usable ft_checkpoint "
            f"(got {ft_ckpt!r}). Phase 1 sanity-checks the FT model."
        )
    print(f"Loading FT checkpoint from {ft_ckpt} (HF only, no HookedTransformer)...")
    tokenizer = AutoTokenizer.from_pretrained(ft_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ft_ckpt,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset_path = "gemma/gemma_toy_dataset_train.jsonl"
    print(f"Loading dataset from {dataset_path}...")
    text_examples = read_relations_jsonl(dataset_path)

    # Use indices 8000+ as the held-out eval split
    dataset, labels = get_subset(tokenizer, text_examples, 8000, len(text_examples))

    accuracy = run_inference(model, dataset, labels, device)
    if accuracy < 0.90:
        print(f"WARNING: accuracy {accuracy:.4f} < 0.90 — FT may have failed; do not proceed with experiments.")
    else:
        print(f"OK: accuracy {accuracy:.4f} >= 0.90 — proceed with experiments.")
    return accuracy


if __name__ == "__main__":
    main()
