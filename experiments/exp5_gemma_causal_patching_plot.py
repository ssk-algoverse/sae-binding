"""
Plot pre-computed causal patching results for Gemma
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("experiments/results/gemma", exist_ok=True)
    
    # Load per-head logit differences
    per_head = torch.load("gemma/per_head_logit_diffs.pt", map_location="cpu")
    print(f"Loaded per-head shape: {per_head.shape}") # Expected: [26, 8]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        per_head.detach().numpy(), 
        cmap="coolwarm", 
        center=0,
        cbar_kws={'label': 'Logit Difference'}
    )
    plt.title("Gemma-2-2B Causal Patching: Per-Head Effect")
    plt.xlabel("Head Index")
    plt.ylabel("Layer")
    plt.tight_layout()
    
    out_path = "experiments/results/gemma/gemma_causal_patching.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plotting to {out_path}")
    
if __name__ == "__main__":
    main()
