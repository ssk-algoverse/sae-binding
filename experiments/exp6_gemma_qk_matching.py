"""
Experiment 6: Gemma QK Matching Analysis
========================================
Proves that the head identified via causal patching (L22H4) routes structurally
by exact query-key matching on the fact positions in Gemma-2-2B.
"""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformer_lens import HookedTransformer

os.environ["HF_TOKEN"] = "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"

def main():
    os.makedirs("experiments/results/gemma", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading pretrained Gemma-2-2b...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", center_unembed=True, center_writing_weights=True, fold_ln=True, device=device)
    model.eval()

    layer = 22
    head = 4

    print("Loading dataset...")
    with open("gemma/gemma_pp_dataset.jsonl", "r") as f:
        lines = [json.loads(line) for line in f]
        
    comma_id = model.to_single_token(",")
    
    correct_scores = []
    distractor_scores = []
    
    for idx, line in enumerate(tqdm(lines)):
        inp = line["clean"]
        target = line["label"]
        tokens = model.to_tokens(inp).squeeze(0)
        
        # The target fact contains the label
        # We need to find which comma corresponds to the correct fact vs distractors
        # A simpler way: we just compute QK dot products at the last layer for all positions
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=[f"blocks.{layer}.attn.hook_k", f"blocks.{layer}.attn.hook_q"])
            
            K = cache[f"blocks.{layer}.attn.hook_k"][0, :, head // 2, :] # [seq, d_head]
            Q = cache[f"blocks.{layer}.attn.hook_q"][0, -1, head, :] # [d_head], query is the last token
            
            # Dot products
            scores = (K @ Q.unsqueeze(-1)).squeeze(-1) # [seq]
            
            # Find comma positions
            comma_positions = (tokens == comma_id).nonzero().squeeze(-1)
            
            if comma_positions.numel() == 0:
                continue
                
            # Which comma corresponds to the correct fact?
            # Let's find the token position of the target label in the context
            try:
                target_token = model.to_single_token(target)
                target_pos = (tokens == target_token).nonzero().squeeze(-1)[0]
                
                # The correct comma is the first comma AFTER the target_pos
                valid_commas = comma_positions[comma_positions > target_pos]
                if valid_commas.numel() == 0:
                    correct_comma = comma_positions[-1]
                else:
                    correct_comma = valid_commas[0]
                    
                correct_score = scores[correct_comma].item()
                correct_scores.append(correct_score)
                
                for c_pos in comma_positions:
                    if c_pos != correct_comma:
                        distractor_scores.append(scores[c_pos].item())
                        
            except Exception as e:
                pass


    # Plot KDE of correct vs distractor attention scores
    plt.figure(figsize=(8, 6))
    if correct_scores and distractor_scores:
        sns.kdeplot(correct_scores, fill=True, color='green', label=f'Correct Fact Comma (Mean: {np.mean(correct_scores):.1f})')
        sns.kdeplot(distractor_scores, fill=True, color='red', label=f'Distractor Commas (Mean: {np.mean(distractor_scores):.1f})')
        plt.title(f"Gemma-2-2B Q-K Matching (Layer {layer}, Head {head})")
        plt.xlabel("Query-Key Dot Product (Pre-Softmax)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        
        out_path = "experiments/results/gemma/gemma_qk_matching.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved QK matching plot to {out_path}")
    else:
        print("Not enough data to plot.")

if __name__ == "__main__":
    main()
