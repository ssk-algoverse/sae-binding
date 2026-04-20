"""
Experiment 6: Gemma QK Matching Analysis (with held-out + null controls)
========================================================================
Tests whether the selected circuit head's query-key dot products preferentially
match the comma following the target fact, vs three nulls:

  • DISTRACTOR_COMMA — same head, different comma in the same prompt
  • RANDOM_HEAD     — uniformly sampled (layer in preset range, head != target) on same comma
                      (controls for "any deep head scores the right comma high")
  • RANDOM_POSITION — same head, randomly chosen comma in the same prompt
                      (controls for "head just scores all commas the same")

The dataset is split 50/50 at the prompt level on a fixed seed; we report
on the HOLDOUT half. The first half is reserved for the path-patching head
selection (PathPatchingGemma.ipynb).

Model and target (layer, head) are selected via the ``GEMMA_PRESET`` env var
(see experiments/_gemma_config.py). The preset must have ``target_layer``,
``target_head``, and ``random_head_layer_range`` filled in — these come from
re-running PathPatchingGemma.ipynb on the chosen model.
"""
import os
import sys
import json
import random as _random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gemma_config import get_preset, require, model_arch, kv_head_for


SEED = 0
HOLDOUT_FRAC = 0.50  # second half held out from the head-selection step


def set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_qk_scores(cache, layer, q_head, group_size, query_pos=-1):
    """Return per-position dot products K @ Q for a given (layer, head) at query position."""
    kv_h = kv_head_for(q_head, group_size)
    K = cache[f"blocks.{layer}.attn.hook_k"][0, :, kv_h, :]              # [seq, d_head]
    Q = cache[f"blocks.{layer}.attn.hook_q"][0, query_pos, q_head, :]    # [d_head]
    return (K @ Q.unsqueeze(-1)).squeeze(-1)                             # [seq]


def main():
    os.makedirs("experiments/results/gemma", exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preset = get_preset()
    require(preset, "target_layer", "target_head", "random_head_layer_range")
    target_layer = preset["target_layer"]
    target_head = preset["target_head"]
    random_head_layer_range = preset["random_head_layer_range"]

    print(f"Loading pretrained {preset['model_name']}...")
    model = HookedTransformer.from_pretrained(
        preset["model_name"], center_unembed=True, center_writing_weights=True, fold_ln=True, device=device
    )
    model.eval()
    arch = model_arch(model)
    n_q_heads = arch["n_q_heads"]
    group_size = arch["group_size"]
    print(f"  preset={preset['_name']}  target=L{target_layer}H{target_head}  "
          f"n_q_heads={n_q_heads}  n_kv_heads={arch['n_kv_heads']}  group_size={group_size}")

    print("Loading dataset...")
    with open("gemma/gemma_pp_dataset.jsonl", "r") as f:
        lines = [json.loads(line) for line in f]

    # Held-out split at the prompt level
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(lines))
    n_holdout = int(round(len(lines) * HOLDOUT_FRAC))
    holdout_idx = set(perm[:n_holdout].tolist())
    holdout_lines = [lines[i] for i in sorted(holdout_idx)]
    print(f"  Total prompts: {len(lines)}  Holdout: {len(holdout_lines)}  "
          f"(seed={SEED})")

    comma_id = model.to_single_token(",")

    correct_scores = []        # L22H4 on the correct comma
    distractor_scores = []     # L22H4 on other commas in same prompt
    random_pos_scores = []     # L22H4 on a randomly chosen comma per prompt
    random_head_scores = []    # random (L,H) on the correct comma

    # All Q-head indices in the layer range, excluding the chosen head at the chosen layer
    candidate_heads = [
        (L, H)
        for L in range(*random_head_layer_range)
        for H in range(n_q_heads)
        if not (L == target_layer and H == target_head)
    ]

    # Run forward pass once per prompt and cache K/Q for all needed layers
    needed_layers = sorted({target_layer, *(L for L, _ in candidate_heads)})
    needed_names = [f"blocks.{l}.attn.hook_k" for l in needed_layers] + \
                   [f"blocks.{l}.attn.hook_q" for l in needed_layers]

    for line in tqdm(holdout_lines):
        inp = line["clean"]
        target = line["label"]
        tokens = model.to_tokens(inp).squeeze(0)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=needed_names)

            comma_positions = (tokens == comma_id).nonzero().squeeze(-1)
            if comma_positions.numel() == 0:
                continue

            try:
                target_token = model.to_single_token(target)
                target_pos = (tokens == target_token).nonzero().squeeze(-1)[0]
                valid = comma_positions[comma_positions > target_pos]
                correct_comma = valid[0] if valid.numel() > 0 else comma_positions[-1]
            except Exception:
                continue  # don't silently fold bad rows into the null

            # Target head scores
            scores = get_qk_scores(cache, target_layer, target_head, group_size)
            correct_scores.append(scores[correct_comma].item())
            for c in comma_positions:
                if c != correct_comma:
                    distractor_scores.append(scores[c].item())

            # Random-position null: pick any comma uniformly (independent of target)
            rand_c = comma_positions[rng.integers(0, comma_positions.numel())]
            random_pos_scores.append(scores[rand_c].item())

            # Random-head null: random other head, score on correct comma
            L_r, H_r = candidate_heads[rng.integers(0, len(candidate_heads))]
            scores_r = get_qk_scores(cache, L_r, H_r, group_size)
            random_head_scores.append(scores_r[correct_comma].item())

    print(f"\nN correct={len(correct_scores)}  distractor={len(distractor_scores)}  "
          f"random_pos={len(random_pos_scores)}  random_head={len(random_head_scores)}")
    print(f"means: correct={np.mean(correct_scores):.2f}  distractor={np.mean(distractor_scores):.2f}  "
          f"random_pos={np.mean(random_pos_scores):.2f}  random_head={np.mean(random_head_scores):.2f}")

    # Plot 4-way KDE
    plt.figure(figsize=(9, 6))
    if correct_scores and distractor_scores:
        sns.kdeplot(correct_scores, fill=True, color="#2ecc71",
                    label=f"L{target_layer}H{target_head} • correct comma (μ={np.mean(correct_scores):.1f})")
        sns.kdeplot(distractor_scores, fill=True, color="#e74c3c",
                    label=f"L{target_layer}H{target_head} • distractor commas (μ={np.mean(distractor_scores):.1f})")
        sns.kdeplot(random_pos_scores, fill=True, color="#95a5a6",
                    label=f"L{target_layer}H{target_head} • random-position null (μ={np.mean(random_pos_scores):.1f})")
        sns.kdeplot(random_head_scores, fill=True, color="#3498db",
                    label=f"random head • correct comma (μ={np.mean(random_head_scores):.1f})")
        plt.title(f"{preset['model_name']} Q-K Matching (held-out half, n={len(holdout_lines)} prompts)")
        plt.xlabel("Query-Key Dot Product (pre-softmax)")
        plt.ylabel("Density")
        plt.legend(fontsize=9)
        plt.tight_layout()
        out_path = f"experiments/results/gemma/{preset['_name']}_qk_matching.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved QK matching plot to {out_path}")
    else:
        print("Not enough data to plot.")


if __name__ == "__main__":
    main()
