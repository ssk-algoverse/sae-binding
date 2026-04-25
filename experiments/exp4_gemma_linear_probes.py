"""
Experiment 4: Linear Probes on Gemma
=================================================
Trains linear probes on residual stream activations (`blocks.L.hook_resid_post`)
across all layers in a Gemma model to decode E1, T, E2, and (E1,T) from fact separator positions.

Model is selected via the ``GEMMA_PRESET`` env var (see experiments/_gemma_config.py).
Defaults to gemma-2-2b for backward compatibility.
"""

import os
import sys
import json
import random as _random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gemma_config import get_preset, model_arch, load_model

OUTPUT_DIR = "experiments/results/gemma"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 0
N_BOOT_SEEDS = 5
HOLDOUT_FRAC = 0.30  # fraction of *prompts* held out for final eval
MAX_PROMPTS = 300    # raised from 150 so the 30% holdout has enough class coverage


def set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_facts_from_tokens(tokens_np, comma_id, period_id):
    """
    Extracts facts from a tokenised prompt.
    Returns a list of dicts with fact metadata and the index of the separator token.
    """
    facts = []
    # Find positions of ',' and '.'
    sep_positions = np.where((tokens_np == comma_id) | (tokens_np == period_id))[0]
    
    for sp in sep_positions:
        # A fact is exactly E1, T, E2 before the separator.
        if sp < 3: continue
        e1 = int(tokens_np[sp - 3])
        t_rel = int(tokens_np[sp - 2])
        e2 = int(tokens_np[sp - 1])
        
        # Valid facts won't contain the Who token, and will just be normal word pieces.
        facts.append({
            "sep_pos": sp,
            "e1": e1,
            "t": t_rel,
            "e2": e2,
            "e1_t": f"{e1}_{t_rel}" # String label for combined class
        })
    return facts

def collect_activations(model, n_layers, max_prompts=MAX_PROMPTS):
    """Collect resid_post activations at SEP positions, tracking which prompt each
    fact came from so we can split at the prompt level (no leakage)."""
    device = model.cfg.device
    print(f"Loading up to {max_prompts} prompts...")

    with open("gemma/gemma_toy_dataset_train.jsonl", "r") as f:
        lines = [json.loads(next(f)) for _ in range(max_prompts)]

    comma_id = model.to_single_token(",")
    period_id = model.to_single_token(".")

    layer_acts = {l: [] for l in range(n_layers)}
    metadata = {"e1": [], "t": [], "e2": [], "e1_t": [], "prompt_id": []}

    print("Running forward passes and caching resid_post...")
    for prompt_id, line in enumerate(tqdm(lines)):
        inp = line["input"]
        tokens = model.to_tokens(inp).squeeze(0)
        facts = extract_facts_from_tokens(tokens.cpu().numpy(), comma_id, period_id)

        if not facts: continue

        filter_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
        _, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=filter_names)

        for fact in facts:
            sp = fact["sep_pos"]
            for l in range(n_layers):
                act = cache[f"blocks.{l}.hook_resid_post"][0, sp, :].float().cpu().numpy()
                layer_acts[l].append(act)

            metadata["e1"].append(fact["e1"])
            metadata["t"].append(fact["t"])
            metadata["e2"].append(fact["e2"])
            metadata["e1_t"].append(fact["e1_t"])
            metadata["prompt_id"].append(prompt_id)

    for l in range(n_layers):
        layer_acts[l] = np.stack(layer_acts[l])
    for k in metadata:
        metadata[k] = np.array(metadata[k])

    return layer_acts, metadata


def split_train_holdout(prompt_ids, holdout_frac=HOLDOUT_FRAC, seed=SEED):
    """Split at the *prompt* level so all facts from a given prompt land in the same fold."""
    rng = np.random.default_rng(seed)
    unique_prompts = np.unique(prompt_ids)
    rng.shuffle(unique_prompts)
    n_holdout = int(round(len(unique_prompts) * holdout_frac))
    holdout_prompts = set(unique_prompts[:n_holdout].tolist())
    train_mask = np.array([p not in holdout_prompts for p in prompt_ids])
    return train_mask


def train_linear_probes(layer_acts, labels, prompt_ids, target_name, n_layers,
                        seeds=tuple(range(N_BOOT_SEEDS))):
    """For each layer:
       - 3-fold stratified-CV accuracy on the TRAIN split (selection metric)
       - Held-out accuracy on the HOLDOUT split (reporting metric)
    Bootstrapped across `seeds` (each seed = a different prompt-level train/holdout split).
    Returns dict with arrays of shape [n_seeds, n_layers]."""
    cv_all = np.zeros((len(seeds), n_layers))
    holdout_all = np.zeros((len(seeds), n_layers))

    for s_i, seed in enumerate(seeds):
        train_mask = split_train_holdout(prompt_ids, seed=seed)
        # Filter rare classes globally so train/holdout cover the same label set
        unique, counts = np.unique(labels[train_mask], return_counts=True)
        valid_classes = unique[counts >= 3]
        keep = np.isin(labels, valid_classes)

        tr_mask = train_mask & keep
        ho_mask = (~train_mask) & keep

        if s_i == 0:
            n_tr, n_ho = int(tr_mask.sum()), int(ho_mask.sum())
            print(f"  [{target_name}] seed{seed}: train={n_tr} facts, holdout={n_ho} facts, "
                  f"{len(valid_classes)} classes (≥3 in train)")

        for l in range(n_layers):
            X_tr, y_tr = layer_acts[l][tr_mask], labels[tr_mask]
            X_ho, y_ho = layer_acts[l][ho_mask], labels[ho_mask]

            # CV on train (selection metric)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            accs = []
            for tr_i, te_i in skf.split(X_tr, y_tr):
                clf = RidgeClassifier(alpha=1.0)
                clf.fit(X_tr[tr_i], y_tr[tr_i])
                accs.append(clf.score(X_tr[te_i], y_tr[te_i]))
            cv_all[s_i, l] = float(np.mean(accs))

            # Held-out (reporting metric): train on ALL train, eval on holdout
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_tr, y_tr)
            # filter holdout to label set seen in training (Ridge can't predict unseen classes)
            seen = np.isin(y_ho, np.unique(y_tr))
            if seen.sum() == 0:
                holdout_all[s_i, l] = float("nan")
            else:
                holdout_all[s_i, l] = clf.score(X_ho[seen], y_ho[seen])

    return {"cv": cv_all, "holdout": holdout_all}

def plot_probe_accuracies(acc_dict, filename, n_layers, model_name, kind="holdout"):
    """Plot per-layer probe accuracy with shaded ±std band across seeds."""
    plt.figure(figsize=(10, 6))
    colors = {"e1": "#3498db", "t": "#2ecc71", "e2": "#e74c3c", "e1_t": "#9b59b6"}
    labels_clean = {"e1": "Entity 1 (Payload)", "t": "Relation", "e2": "Entity 2",
                    "e1_t": "Composed Address (E1, T)"}

    xs = np.arange(n_layers)
    for target in ["e1", "t", "e2", "e1_t"]:
        arr = acc_dict[target][kind]  # [n_seeds, n_layers]
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        plt.plot(xs, mean, marker='o', label=labels_clean[target], color=colors[target], linewidth=2)
        plt.fill_between(xs, mean - std, mean + std, color=colors[target], alpha=0.2)

    plt.xlabel("Layer")
    plt.ylabel(f"Probe Accuracy ({kind})")
    plt.title(f"{model_name} Probes — {kind} (mean ± std across {N_BOOT_SEEDS} prompt-level splits)", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.xticks(range(0, n_layers, max(1, n_layers // 13)))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = get_preset()
    model = load_model(preset, device)
    arch = model_arch(model)
    n_layers = arch["n_layers"]
    print(f"  preset={preset['_name']}  n_layers={n_layers}  n_q_heads={arch['n_q_heads']}  n_kv_heads={arch['n_kv_heads']}")

    layer_acts, metadata = collect_activations(model, n_layers=n_layers, max_prompts=MAX_PROMPTS)

    results = {}
    for target in ["e1", "t", "e2", "e1_t"]:
        print(f"\n=== Probing target: {target} ===")
        out = train_linear_probes(layer_acts, metadata[target], metadata["prompt_id"], target, n_layers=n_layers)
        results[target] = out

        cv_mean = np.nanmean(out["cv"], axis=0)
        ho_mean = np.nanmean(out["holdout"], axis=0)
        ho_std = np.nanstd(out["holdout"], axis=0)
        best_cv_layer = int(np.argmax(cv_mean))
        print(f"  best layer by CV: L{best_cv_layer} (cv={cv_mean[best_cv_layer]:.3f}) "
              f"→ HOLDOUT acc at that layer = {ho_mean[best_cv_layer]:.3f} ± {ho_std[best_cv_layer]:.3f}")

    tag = preset["_name"]
    plot_probe_accuracies(results, os.path.join(OUTPUT_DIR, f"{tag}_probe_accuracies_holdout.png"),
                          n_layers=n_layers, model_name=preset["model_name"], kind="holdout")
    plot_probe_accuracies(results, os.path.join(OUTPUT_DIR, f"{tag}_probe_accuracies_cv.png"),
                          n_layers=n_layers, model_name=preset["model_name"], kind="cv")

    # Save raw arrays for later analysis (shape per target: [n_seeds, n_layers])
    save_payload = {}
    for tgt, out in results.items():
        save_payload[f"{tgt}_cv"] = out["cv"]
        save_payload[f"{tgt}_holdout"] = out["holdout"]
    np.savez(os.path.join(OUTPUT_DIR, f"{tag}_probe_results.npz"), **save_payload)
    
if __name__ == "__main__":
    main()
