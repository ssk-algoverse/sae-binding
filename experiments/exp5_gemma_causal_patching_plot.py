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
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gemma_config import get_preset

def main():
    preset = get_preset()
    preset_name = preset["_name"]
    os.makedirs("experiments/results/gemma", exist_ok=True)

    # Per-preset .pt path: gemma-2-2b uses the canonical name; others use a
    # suffix so multiple runs coexist without overwriting each other.
    if preset_name == "gemma-2-2b":
        pt_path = "gemma/per_head_logit_diffs.pt"
    else:
        pt_path = f"gemma/per_head_logit_diffs_{preset_name}.pt"

    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"Expected {pt_path}. Re-run gemma/pp_toy_dataset.py for preset "
            f"'{preset_name}' and save the tensor there."
        )

    per_head = torch.load(pt_path, map_location="cpu")
    print(f"Loaded per-head shape: {per_head.shape}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        per_head.detach().float().numpy(),
        cmap="coolwarm",
        center=0,
        cbar_kws={'label': 'Logit Difference'}
    )
    plt.title(f"{preset_name} Causal Patching: Per-Head Effect")
    plt.xlabel("Head Index")
    plt.ylabel("Layer")
    plt.tight_layout()

    out_path = f"experiments/results/gemma/{preset_name}_causal_patching.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
