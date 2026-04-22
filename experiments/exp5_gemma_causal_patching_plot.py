"""
Plot pre-computed causal patching results for Gemma.

⚠️  NOTICE — KEEP IN SYNC WITH gemma/pp_toy_dataset.ipynb
─────────────────────────────────────────────────────────
This script only renders `gemma/per_head_logit_diffs.pt`. That tensor is
produced by `gemma/pp_toy_dataset.ipynb`, which loads the FT checkpoint
(`gemma/gemma2_ft_toy/checkpoint-900/`) into the gemma-2-2b architecture
template and runs path patching on the toy E1/T/E2 dataset. The notebook
is the source of truth for:

  • which model weights the head-selection ran on (FT, not base),
  • which prompts the head-selection ran on (held-out split logic),
  • how `per_head_logit_diffs` is computed (mean ablation? noise injection?
    metric: logit-diff vs KL?),
  • the layer/head grid being scanned.

If you change anything here that depends on those choices (e.g. the layer
range, the metric, the held-out vs full set assumption), update the
notebook in lock-step and re-export `per_head_logit_diffs.pt`. Likewise,
edits to the notebook that change the tensor's shape or semantics MUST
land alongside an update to this plotter.

The old root-level `PathPatchingGemma.ipynb` used base gemma-2-2b on the
prakash boxes dataset and was *not* the source of the L22H4 claim — it has
been removed. Don't resurrect it without aligning model + dataset with the
rest of the Gemma experiments.
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
