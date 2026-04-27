import json
import torch
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    with open("gemma/gemma_toy_dataset_train.jsonl", "r") as f:
        lines = [json.loads(next(f)) for _ in range(5)]
        
    for i, line in enumerate(lines):
        inp = line["input"]
        label = line["label"]
        print(f"\n--- Example {i} ---")
        print(f"Input: {inp}")
        print(f"Label: {label}")
        
        tokens = tokenizer.encode(inp)
        print("Tokens:")
        for idx, t in enumerate(tokens):
            print(f"  {idx}: {t} -> '{tokenizer.decode([t])}'")

if __name__ == "__main__":
    main()
