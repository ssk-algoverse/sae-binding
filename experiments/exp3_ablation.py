"""
Experiment 3: Mean Ablation — Necessity Proof
==============================================
Ablates L0H0 (address head) and L0H1 (payload head) individually at SEP positions.

Expected results:
  - Ablating L0H0 → model loses routing ability (retrieves wrong fact, accuracy drops)
  - Ablating L0H1 → model routes correctly but loses payload (predicts random entity)
  - Ablating both → model completely fails

This proves necessity, complementing the causal patching which shows sufficiency.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ─── Constants ──────────────────────────────────────────────────────────────
MODEL_REPO_ID = "sebastianhoenig/2L2H_Final"
MODEL_FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"

E = 100
T = 10
SEP = E + T
Q = E + T + 1
PAD = E + T + 2
D_VOCAB = E + T + 3
N_LAYERS = 2
HEADS = 2
D_MODEL = 256
N_CTX = 64

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Reproducibility ────────────────────────────────────────────────────────
import random as _random

SEED = 0


def set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_facts_in_prompt(tokens):
    """Return list of (e1, t, e2) facts present in the prompt (by position)."""
    tokens_np = tokens.cpu().numpy()
    sep_positions = np.where(tokens_np == SEP)[0]
    facts = []
    for sp in sep_positions:
        sp = int(sp)
        if sp < 3:
            continue
        e1 = int(tokens_np[sp - 3])
        t_val = int(tokens_np[sp - 2])
        e2 = int(tokens_np[sp - 1])
        if e1 < E and (E <= t_val < E + T) and e2 < E:
            facts.append((e1, t_val - E, e2))
    return facts


def build_model():
    d_head = D_MODEL // HEADS
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS, n_heads=HEADS, d_model=D_MODEL, d_head=d_head,
        n_ctx=N_CTX, d_vocab=D_VOCAB, d_vocab_out=E,
        attn_only=True, normalization_type="LN", positional_embedding_type="rotary",
    )
    return HookedTransformer(cfg)


def load_model(device):
    weights_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    model = build_model().to(device)
    pretrained = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(pretrained["model"])
    model.cfg.use_attn_result = True
    model.eval()
    return model


class EntityBindingDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = torch.tensor(row["tokens"], dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tokens, label


def load_data():
    dataset = load_dataset("sojup/entity_binding", split="test")
    return EntityBindingDataset(dataset.to_pandas())


# ─── Compute mean activations for ablation ──────────────────────────────────
@torch.no_grad()
def compute_mean_activations(model, dataset, device, max_examples=1000):
    """Compute mean activations at SEP positions for each head in L0."""
    all_hookz_h0 = []
    all_hookz_h1 = []

    n = min(len(dataset), max_examples)
    for idx in tqdm(range(n), desc="Computing mean activations"):
        tokens, label = dataset[idx]
        tokens = tokens.to(device)

        sep_positions = (tokens == SEP).nonzero(as_tuple=False).flatten().tolist()
        if not sep_positions:
            continue

        _, cache = model.run_with_cache(tokens.unsqueeze(0))
        hookz = cache["blocks.0.attn.hook_z"]  # [1, seq, heads, d_head]

        for sp in sep_positions:
            all_hookz_h0.append(hookz[0, sp, 0, :].cpu())
            all_hookz_h1.append(hookz[0, sp, 1, :].cpu())

    mean_h0 = torch.stack(all_hookz_h0).mean(dim=0)
    mean_h1 = torch.stack(all_hookz_h1).mean(dim=0)
    return mean_h0.to(device), mean_h1.to(device)


# ─── Ablation hooks ────────────────────────────────────────────────────────
def make_sep_only_ablation_hook(heads_to_ablate, mean_activations, input_tokens, sep_token_id=SEP):
    """Create a hook that replaces specified head outputs with their mean ONLY at SEP positions."""
    sep_positions = (input_tokens == sep_token_id).nonzero(as_tuple=False)[:, -1].tolist()

    def hook_fn(act, hook):
        act = act.clone()
        for head_idx in heads_to_ablate:
            for sp in sep_positions:
                act[:, sp, head_idx, :] = mean_activations[head_idx]
        return act

    return hook_fn


def make_zero_ablation_hook(heads_to_ablate, input_tokens, sep_token_id=SEP):
    """Zero-ablate specified heads at SEP positions."""
    sep_positions = (input_tokens == sep_token_id).nonzero(as_tuple=False)[:, -1].tolist()

    def hook_fn(act, hook):
        act = act.clone()
        for head_idx in heads_to_ablate:
            for sp in sep_positions:
                act[:, sp, head_idx, :] = 0.0
        return act

    return hook_fn


def make_swap_hook(input_tokens, sep_token_id=SEP):
    """Swap L0H0 and L0H1 outputs at SEP positions.

    Tests role specialization: if H0 = Address and H1 = Payload, swapping
    should produce a *specific* error pattern (e.g., predict the E2 of a
    fact whose role was played by the swapped vector), not just degraded acc.
    """
    sep_positions = (input_tokens == sep_token_id).nonzero(as_tuple=False)[:, -1].tolist()

    def hook_fn(act, hook):
        act = act.clone()
        for sp in sep_positions:
            tmp = act[:, sp, 0, :].clone()
            act[:, sp, 0, :] = act[:, sp, 1, :]
            act[:, sp, 1, :] = tmp
        return act

    return hook_fn


# ─── Evaluation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_ablation(model, dataset, device, mean_h0, mean_h1, max_examples=2000):
    """Run all ablation conditions and report accuracy + error categories.

    Error categories (per prediction):
      correct                — pred == label
      wrong_E2_in_prompt     — pred is some E2 that DOES appear in the prompt
                               (model routed to a wrong fact, evidence of
                                preserved-but-mis-routed addressing)
      wrong_E2_not_in_prompt — pred is an E2 that does NOT appear in the prompt
                               (model hallucinated; loss of payload coherence)
    """
    mean_acts = {0: mean_h0, 1: mean_h1}

    conditions = {
        "clean": ("none", None),
        "ablate_L0H0_mean (address)": ("mean", [0]),
        "ablate_L0H1_mean (payload)": ("mean", [1]),
        "ablate_both_mean": ("mean", [0, 1]),
        "ablate_L0H0_zero (address)": ("zero", [0]),
        "ablate_L0H1_zero (payload)": ("zero", [1]),
        "ablate_both_zero": ("zero", [0, 1]),
        "swap_L0H0_L0H1": ("swap", None),
    }

    results = {}
    n = min(len(dataset), max_examples)

    for cond_name, (kind, heads) in conditions.items():
        correct = 0
        total = 0
        logit_diffs = []
        cat_counts = {"correct": 0, "wrong_E2_in_prompt": 0, "wrong_E2_not_in_prompt": 0}

        for idx in tqdm(range(n), desc=cond_name):
            tokens, label = dataset[idx]
            tokens = tokens.to(device)
            label_val = int(label.item())

            if kind == "none":
                logits = model(tokens.unsqueeze(0))
            elif kind == "zero":
                hook_fn = make_zero_ablation_hook(heads, tokens)
                logits = model.run_with_hooks(
                    tokens.unsqueeze(0),
                    fwd_hooks=[("blocks.0.attn.hook_z", hook_fn)],
                )
            elif kind == "mean":
                hook_fn = make_sep_only_ablation_hook(heads, mean_acts, tokens)
                logits = model.run_with_hooks(
                    tokens.unsqueeze(0),
                    fwd_hooks=[("blocks.0.attn.hook_z", hook_fn)],
                )
            elif kind == "swap":
                hook_fn = make_swap_hook(tokens)
                logits = model.run_with_hooks(
                    tokens.unsqueeze(0),
                    fwd_hooks=[("blocks.0.attn.hook_z", hook_fn)],
                )

            final_logits = logits[0, -1, :].detach()
            pred = int(final_logits.argmax().item())
            correct += int(pred == label_val)
            total += 1

            # Confusion category
            if pred == label_val:
                cat_counts["correct"] += 1
            else:
                facts = extract_facts_in_prompt(tokens)
                e2s_in_prompt = {f[2] for f in facts}
                if pred in e2s_in_prompt:
                    cat_counts["wrong_E2_in_prompt"] += 1
                else:
                    cat_counts["wrong_E2_not_in_prompt"] += 1

            masked = final_logits.clone()
            masked[label_val] = float("-inf")
            best_wrong = masked.max().item()
            logit_diffs.append(final_logits[label_val].item() - best_wrong)

        acc = correct / total
        mean_ld = np.mean(logit_diffs)
        cat_pcts = {k: v / total for k, v in cat_counts.items()}
        results[cond_name] = {
            "accuracy": acc, "mean_logit_diff": mean_ld, "n": total,
            "cat_counts": cat_counts, "cat_pcts": cat_pcts,
        }
        print(
            f"  {cond_name:35s} → acc={acc:.4f}  Δlogit={mean_ld:6.2f}  "
            f"[correct={cat_pcts['correct']:.3f} wrong_in_prompt={cat_pcts['wrong_E2_in_prompt']:.3f} "
            f"wrong_oop={cat_pcts['wrong_E2_not_in_prompt']:.3f}]"
        )

    return results


def plot_ablation_results(results, filename):
    """Bar chart of ablation accuracies."""
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    # Assign colors
    colors = []
    for n in names:
        if "clean" in n:
            colors.append("#2ecc71")
        elif "L0H0" in n:
            colors.append("#3498db")
        elif "L0H1" in n:
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Accuracy")
    ax.set_title("Mean Ablation: Necessity of L0H0 (Address) and L0H1 (Payload)", fontweight="bold")
    ax.set_xlim(0, 1.05)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_logit_diffs(results, filename):
    """Bar chart of mean logit differences across conditions."""
    names = list(results.keys())
    lds = [results[n]["mean_logit_diff"] for n in names]

    colors = []
    for n in names:
        if "clean" in n:
            colors.append("#2ecc71")
        elif "L0H0" in n:
            colors.append("#3498db")
        elif "L0H1" in n:
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), lds, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean Logit Difference (correct - best wrong)")
    ax.set_title("Mean Ablation: Impact on Logit Difference", fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    for bar, ld in zip(bars, lds):
        x_pos = bar.get_width() + 0.5 if bar.get_width() >= 0 else bar.get_width() - 3
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{ld:.1f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed={SEED}")

    print("Loading model...")
    model = load_model(device)

    print("Loading dataset...")
    dataset = load_data()

    print("Computing mean activations for ablation...")
    mean_h0, mean_h1 = compute_mean_activations(model, dataset, device, max_examples=1000)

    print("\n=== Ablation Results ===")
    results = evaluate_ablation(model, dataset, device, mean_h0, mean_h1, max_examples=2000)

    # Plots
    plot_ablation_results(results, OUTPUT_DIR / "ablation_accuracy.png")
    plot_logit_diffs(results, OUTPUT_DIR / "ablation_logit_diffs.png")

    # Print summary table with confusion breakdown
    print("\n" + "=" * 110)
    print(f"{'Condition':35s} {'Accuracy':>10s} {'Logit Diff':>12s} "
          f"{'correct':>10s} {'wrong_in':>10s} {'wrong_oop':>10s}")
    print("-" * 110)
    for name, res in results.items():
        c = res["cat_pcts"]
        print(f"{name:35s} {res['accuracy']:10.4f} {res['mean_logit_diff']:12.2f} "
              f"{c['correct']:10.4f} {c['wrong_E2_in_prompt']:10.4f} {c['wrong_E2_not_in_prompt']:10.4f}")
    print("=" * 110)

    print("\n✅ Experiment 3 complete. Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
