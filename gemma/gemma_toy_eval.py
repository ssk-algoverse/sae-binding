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


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def get_subset(tokenizer, text_examples, start_idx, end_idx):
    """Tokenize inputs (variable length, kept as lists) and first-token labels.

    We take the FIRST token of each label: that's the next-token prediction
    target after the prompt. Under Gemma-2's tokenizer all labels happened to
    be single-token, but Gemma-3 sometimes splits them — first-token match is
    the intended semantic.
    """
    input_ids_list = []
    label_first_tokens = []
    for rec in text_examples[start_idx:end_idx]:
        input_ids_list.append(encode(tokenizer, rec["input"]))
        label_ids = encode(tokenizer, rec["label"])
        label_first_tokens.append(label_ids[0])
    return input_ids_list, label_first_tokens


def left_pad_batch(input_ids_list, pad_id):
    max_len = max(len(x) for x in input_ids_list)
    batch = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        batch[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attn[i, max_len - len(ids):] = 1
    return batch, attn


def run_inference(model, tokenizer, input_ids_list, label_tokens, device, batch_size=4):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    correct = 0
    total = 0
    for idx in tqdm(range(0, len(input_ids_list), batch_size), desc="examples"):
        chunk = input_ids_list[idx:idx + batch_size]
        inputs, attn = left_pad_batch(chunk, pad_id)
        inputs = inputs.to(device)
        attn = attn.to(device)
        labels = torch.tensor(label_tokens[idx:idx + batch_size]).to(device)
        with torch.no_grad():
            logits = model(input_ids=inputs, attention_mask=attn).logits
        preds = logits[:, -1, :].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(chunk)
        del inputs, attn, labels, logits, preds
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
    input_ids_list, labels = get_subset(tokenizer, text_examples, 8000, len(text_examples))

    accuracy = run_inference(model, tokenizer, input_ids_list, labels, device)
    if accuracy < 0.90:
        print(f"WARNING: accuracy {accuracy:.4f} < 0.90 — FT may have failed; do not proceed with experiments.")
    else:
        print(f"OK: accuracy {accuracy:.4f} >= 0.90 — proceed with experiments.")
    return accuracy


if __name__ == "__main__":
    main()
