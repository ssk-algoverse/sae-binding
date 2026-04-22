"""
Experiment 7: Gemma SAE Feature Recovery (with controls + sweep)
================================================================
Tests whether Gemma-Scope SAEs suffer from the 'dark matter' failure mode.

For each (SAE width, L0) config in CONFIGS, compares Ridge probe accuracy
for E2 (Payload) and (E1,T) (Composed Address) across:

  raw X                — full residual activation
  X_recon              — SAE reconstruction
  z                    — SAE sparse latents (the actual interpretability target)
  PCA-rank-matched     — top-r PCA reconstruction with r ≈ mean SAE L0
  RandProj-rank-matched— random orthonormal rank-r projection of X

If z >> X_recon, the (E1,T) feature lives in the SAE dictionary but is
linearly destroyed by reconstruction. If both are << raw, the failure
is attributable to the SAE architecture, not just rank loss
(check vs PCA / RandProj at the same rank).
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
from sae_lens import SAE
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gemma_config import get_preset, require, resolve_sae_configs, load_model

SEED = 0
MAX_PROMPTS = 1500
N_BOOT_SEEDS = 5

# Model + target layer + SAE configs come from GEMMA_PRESET (see _gemma_config.py).
# To add a new SAE (e.g. width_65k/canonical) edit the preset's sae_configs list.


def set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sae(release, sae_id, device):
    print(f"  Loading SAE: {release} / {sae_id}")
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


def collect_raw_activations(model, layer, max_prompts):
    device = model.cfg.device
    with open("gemma/gemma_toy_dataset_train.jsonl", "r") as f:
        lines = [json.loads(next(f)) for _ in range(max_prompts)]

    comma_id = model.to_single_token(",")
    period_id = model.to_single_token(".")
    filter_name = f"blocks.{layer}.hook_resid_post"

    raw_acts = []
    metadata = {"e2": [], "e1_t": []}

    print(f"Collecting Layer {layer} activations from {max_prompts} prompts...")
    for line in tqdm(lines):
        inp = line["input"]
        tokens = model.to_tokens(inp).squeeze(0)
        tokens_np = tokens.cpu().numpy()
        sep_positions = np.where((tokens_np == comma_id) | (tokens_np == period_id))[0]

        facts = []
        for sp in sep_positions:
            if sp < 3:
                continue
            e1 = int(tokens_np[sp - 3])
            t_rel = int(tokens_np[sp - 2])
            e2 = int(tokens_np[sp - 1])
            facts.append((sp, e1, t_rel, e2))

        if not facts:
            continue

        _, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=[filter_name])
        acts = cache[filter_name][0]

        for sp, e1, t_rel, e2 in facts:
            raw_acts.append(acts[sp].cpu().numpy())
            metadata["e2"].append(e2)
            metadata["e1_t"].append(f"{e1}_{t_rel}")

    raw_acts = np.stack(raw_acts)
    for k in metadata:
        metadata[k] = np.array(metadata[k])
    return raw_acts, metadata


@torch.no_grad()
def encode_decode(sae, X_np, device, batch=256):
    """Return (X_recon, z) for the whole array X_np."""
    X = torch.from_numpy(X_np).float().to(device)
    Xr_chunks, Z_chunks = [], []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch]
        z = sae.encode(xb)
        x_hat = sae.decode(z)
        Xr_chunks.append(x_hat.cpu().numpy())
        Z_chunks.append(z.cpu().numpy())
    return np.concatenate(Xr_chunks, 0), np.concatenate(Z_chunks, 0)


def pca_truncate(X, rank, seed=0):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt_k = Vt[:rank]
    proj = Xc @ Vt_k.T
    return proj @ Vt_k + X.mean(0, keepdims=True)


def random_projection_truncate(X, rank, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    R = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(R)
    proj = X @ Q
    return proj @ Q.T


def probe(X, y, seed=0, n_splits=3, min_count=5):
    unique, counts = np.unique(y, return_counts=True)
    valid = unique[counts >= min_count]
    mask = np.isin(y, valid)
    Xf, yf = X[mask], y[mask]
    if len(np.unique(yf)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Xf, yf):
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(Xf[tr], yf[tr])
        accs.append(clf.score(Xf[te], yf[te]))
    return float(np.mean(accs))


def probe_bootstrap(X, y, seeds=tuple(range(N_BOOT_SEEDS))):
    accs = [probe(X, y, seed=s) for s in seeds]
    accs = np.array(accs)
    return float(np.nanmean(accs)), float(np.nanstd(accs))


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preset = get_preset()
    require(preset, "target_layer")
    layer = preset["target_layer"]
    configs = resolve_sae_configs(preset, layer=layer)

    model = load_model(preset, device)
    print(f"  preset={preset['_name']}  target_layer=L{layer}  n_configs={len(configs)}")

    raw_acts, metadata = collect_raw_activations(model, layer=layer, max_prompts=MAX_PROMPTS)
    print(f"Total facts collected: {len(raw_acts)}")

    targets = ["e2", "e1_t"]
    target_labels = {"e2": "Payload (E2)", "e1_t": "Composed (E1,T)"}

    # Raw probe (computed once, independent of SAE config)
    raw_probe = {t: probe_bootstrap(raw_acts, metadata[t]) for t in targets}
    print("\n[raw X]")
    for t in targets:
        m, s = raw_probe[t]
        print(f"  {target_labels[t]:<22} acc = {m:.3f} ± {s:.3f}")

    all_results = []  # list of dicts per config

    for cfg_label, release, sae_id in configs:
        print(f"\n===== Config: {cfg_label} =====")
        try:
            sae = load_sae(release, sae_id, device)
        except Exception as e:
            print(f"  Failed to load SAE: {e}\n  Skipping config.")
            continue

        X_recon, Z = encode_decode(sae, raw_acts, device)
        l0 = float((Z > 0).sum(axis=1).mean())
        recon_var = 1 - ((X_recon - raw_acts) ** 2).sum() / ((raw_acts - raw_acts.mean(0)) ** 2).sum()
        rank = max(8, int(round(l0)))
        print(f"  mean L0 = {l0:.1f}  explained-var = {recon_var:.3f}  → control rank = {rank}")

        X_pca = pca_truncate(raw_acts, rank=rank, seed=SEED)
        X_rp = random_projection_truncate(raw_acts, rank=rank, seed=SEED)

        rows = {}
        sources = [
            ("X_recon (SAE)", X_recon),
            ("z (SAE latents)", Z),
            (f"PCA rank={rank}", X_pca),
            (f"RandProj rank={rank}", X_rp),
        ]
        for src_name, M in sources:
            row = {}
            for t in targets:
                m, s = probe_bootstrap(M, metadata[t])
                row[t] = (m, s)
                print(f"  [{src_name:<22} | {target_labels[t]:<22}] {m:.3f} ± {s:.3f}")
            rows[src_name] = row

        all_results.append({
            "label": cfg_label, "rank": rank, "l0": l0,
            "explained_var": recon_var, "rows": rows,
        })

    # Plot: one grouped-bar subplot per config, raw shown as horizontal reference
    if not all_results:
        print("No SAE configs loaded; nothing to plot.")
        return

    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 5), squeeze=False)
    for ax, cfg in zip(axes[0], all_results):
        sources = list(cfg["rows"].keys())
        x = np.arange(len(sources))
        w = 0.38
        e2_means = [cfg["rows"][s]["e2"][0] for s in sources]
        e2_stds = [cfg["rows"][s]["e2"][1] for s in sources]
        e1t_means = [cfg["rows"][s]["e1_t"][0] for s in sources]
        e1t_stds = [cfg["rows"][s]["e1_t"][1] for s in sources]

        ax.bar(x - w / 2, e2_means, w, yerr=e2_stds, capsize=4, label="E2 (payload)", color="#e74c3c")
        ax.bar(x + w / 2, e1t_means, w, yerr=e1t_stds, capsize=4, label="(E1,T) composed", color="#9b59b6")

        ax.axhline(raw_probe["e2"][0], color="#e74c3c", linestyle="--", alpha=0.5,
                   label=f"raw E2 ({raw_probe['e2'][0]:.2f})")
        ax.axhline(raw_probe["e1_t"][0], color="#9b59b6", linestyle="--", alpha=0.5,
                   label=f"raw (E1,T) ({raw_probe['e1_t'][0]:.2f})")

        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probe accuracy")
        ax.set_title(f"{cfg['label']}\nL0={cfg['l0']:.1f}  EV={cfg['explained_var']:.2f}", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"{preset['model_name']} L{layer}: SAE recovery vs rank-matched controls (seeds={N_BOOT_SEEDS})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = f"experiments/results/gemma/{preset['_name']}_sae_recovery.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
