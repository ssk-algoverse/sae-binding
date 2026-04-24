import os
import sys
import json
import torch
import gc
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
from transformer_lens import HookedTransformer

# ── CACHE SETUP ───────────────────────────────────────────────────────────
WORKSPACE_CACHE = "/workspace/cache"
os.makedirs(WORKSPACE_CACHE, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(WORKSPACE_CACHE, "huggingface")
os.environ["PIP_CACHE_DIR"] = os.path.join(WORKSPACE_CACHE, "pip")
os.environ["TORCH_HOME"] = os.path.join(WORKSPACE_CACHE, "torch")

# ── CONFIG ────────────────────────────────────────────────────────────────
# You can change these or pass them via environment variables
TL_MODEL_NAME = os.environ.get("TL_MODEL_NAME", "gemma-2-2b")
HF_MODEL_ID   = os.environ.get("HF_MODEL_ID", "google/gemma-2-2b")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "./gemma2_ft_toy")
DATASET_PATH  = os.environ.get("DATASET_PATH", "gemma/gemma_toy_dataset_train.jsonl")
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", 8))
GRAD_ACCUM    = int(os.environ.get("GRAD_ACCUM", 1))

# For Gemma 3 migration, you would set:
# TL_MODEL_NAME="gemma-3-1b-pt"
# HF_MODEL_ID="google/gemma-3-1b-pt"
# OUTPUT_DIR="./gemma3_1b_ft_toy"

def read_relations_jsonl(path: str):
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def get_subset(text_examples, model, start_idx, end_idx):
    subset = []
    labels = []
    for rec in text_examples[start_idx: end_idx]:
        subset.append(model.to_tokens(rec["input"], prepend_bos=False).squeeze())
        labels.append(model.to_tokens(rec["label"], prepend_bos=False).flatten().tolist())
    return subset, labels

def get_text_trainset(model, dataset, labels):
    train_dataset_text = []
    for idx in range(len(dataset)):
        ex = dataset[idx]
        label = labels[idx]
        ex_text = model.to_string(ex)
        label_text = model.to_string(label)
        train_dataset_text.append({"prompt": ex_text, "label": label_text})
    return train_dataset_text

class FixedDataset(Dataset):
    def __init__(self, examples, tok):
        self.recs = []
        for ex in examples:
            prompt_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
            label_ids  = tok(ex["label"], add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + label_ids
            labels    = [-100] * len(prompt_ids) + label_ids

            self.recs.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]

def dynamic_collator(batch):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        inp = ex["input_ids"] + [0] * pad_len
        lab = ex["labels"] + [-100] * pad_len
        mask = [1] * len(ex["input_ids"]) + [0] * pad_len
        input_ids.append(inp)
        labels.append(lab)
        attention_mask.append(mask)
        
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                print(f"Step {state.global_step}: loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"Step {state.global_step}: EVAL loss={logs['eval_loss']:.4f}")

def main():
    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN environment variable not set. Loading gated models may fail.")

    print(f"Loading {TL_MODEL_NAME} for data processing...")
    model = HookedTransformer.from_pretrained(TL_MODEL_NAME)
    tok = model.tokenizer

    path = DATASET_PATH
    if not os.path.exists(path):
        # Try relative to script if root path fails
        path_alt = os.path.join(os.path.dirname(__file__), os.path.basename(path))
        if os.path.exists(path_alt):
            path = path_alt
        else:
            print(f"ERROR: Dataset not found at {path}")
            sys.exit(1)

    print(f"Reading dataset from {path}...")
    text_examples = read_relations_jsonl(path)
    
    # Process dataset
    print("Processing dataset subset...")
    dataset, labels = get_subset(text_examples, model, 0, 8000)
    train_dataset_text = get_text_trainset(model, dataset, labels)
    
    # Save memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    train_ds = FixedDataset(train_dataset_text[:-max(1, len(train_dataset_text)//10)], tok)
    val_ds   = FixedDataset(train_dataset_text[-max(1, len(train_dataset_text)//10):], tok)

    print(f"Loading {HF_MODEL_ID} for fine-tuning...")
    auto_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_only_model=True,
        logging_steps=10,          # More frequent logs
        logging_first_step=True,   # Log the very first step
        report_to="wandb" if os.environ.get("WANDB_PROJECT") else "none",
        run_name=os.environ.get("WANDB_RUN_NAME"),
        log_level="info",          # Surface internal status
    )

    trainer = Trainer(
        model=auto_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=dynamic_collator,
        callbacks=[PrinterCallback()],
    )

    print("Starting training...")
    trainer.train()
    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Checkpoint saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
