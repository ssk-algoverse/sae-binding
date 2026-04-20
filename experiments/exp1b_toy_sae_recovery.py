"""
Experiment 1b: Train an SAE on Toy 2L2H Activations and Measure Recovery
========================================================================
Companion to exp1 (geometry analysis) and exp7 (Gemma SAE recovery).

The original plan inferred "SAEs cannot recover (E1,T) from this geometry"
from the cosine histograms in exp1. That inference is contested without an
actual SAE on the toy. This script:

  1. Trains a small L1-sparse autoencoder on `blocks.0.hook_resid_post` of
     the 2L2H toy model.
  2. Compares Ridge-classifier recovery of (E1,T) and E2 from
       a) raw residual activations  X
       b) SAE-reconstructed         X_recon
       c) SAE sparse latents        z (the actual interpretability target)
       d) PCA-truncated control     of the same effective rank as the SAE
       e) Random-projection control of the same rank
  3. Reports CV accuracies + bootstrap CI.

A small SAE on the toy is a deliberate stress test: if even the controlled
toy setting shows that the (E1,T) feature lives in z (the sparse code) but
disappears from X_recon, that's the actual "dark matter" claim — separate
from the geometric inference.
"""

import os
import random as _random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ─── Constants ──────────────────────────────────────────────────────────────
MODEL_REPO_ID = "sebastianhoenig/2L2H_Final"
MODEL_FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"

E = 100
T = 10
SEP = E + T
D_VOCAB = E + T + 3
D_MODEL = 256
N_CTX = 64

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0


def set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Toy model loader ───────────────────────────────────────────────────────
def build_model():
    cfg = HookedTransformerConfig(
        n_layers=2, n_heads=2, d_model=D_MODEL, d_head=D_MODEL // 2,
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
        return tokens


# ─── Activation collection ──────────────────────────────────────────────────
def extract_facts(tokens):
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
            facts.append({"sep_pos": sp, "e1": e1, "t": t_val - E, "e2": e2})
    return facts


@torch.no_grad()
def collect_resid_post(model, dataset, device, max_examples=2500):
    vecs, e1s, ts, e2s, e1ts = [], [], [], [], []
    n = min(len(dataset), max_examples)
    for idx in tqdm(range(n), desc="Collect resid_post"):
        tokens = dataset[idx].to(device)
        facts = extract_facts(tokens)
        if not facts:
            continue
        _, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=["blocks.0.hook_resid_post"])
        rp = cache["blocks.0.hook_resid_post"]
        for f in facts:
            vecs.append(rp[0, f["sep_pos"], :].cpu())
            e1s.append(f["e1"])
            ts.append(f["t"])
            e2s.append(f["e2"])
            e1ts.append(f["e1"] * T + f["t"])
    return torch.stack(vecs), np.array(e1s), np.array(ts), np.array(e2s), np.array(e1ts)


# ─── Tiny L1 SAE ────────────────────────────────────────────────────────────
class SAE(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) * (1.0 / np.sqrt(d_in)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_in) * (1.0 / np.sqrt(d_hidden)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Tied init: dec rows ≈ enc cols
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def train_sae(X, d_hidden=1024, l1=3e-3, steps=4000, batch_size=512, lr=1e-3,
              device="cpu", seed=0):
    """Train a small L1 SAE on activations X (np.ndarray or tensor)."""
    set_seed(seed)
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    X = X.to(device)
    n, d = X.shape
    sae = SAE(d, d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses, l0s = [], []
    for step in tqdm(range(steps), desc=f"Train SAE d_hid={d_hidden}"):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb = X[idx]
        x_hat, z = sae(xb)
        recon = ((x_hat - xb) ** 2).mean()
        sparsity = z.abs().mean()
        loss = recon + l1 * sparsity
        opt.zero_grad()
        loss.backward()
        # Decoder norm constraint (standard for L1 SAEs)
        with torch.no_grad():
            sae.W_dec.data /= sae.W_dec.data.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                l0_now = (z > 0).float().sum(dim=-1).mean().item()
                losses.append(loss.item())
                l0s.append(l0_now)
    return sae, losses, l0s


@torch.no_grad()
def encode_decode(sae, X, device):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    X = X.to(device)
    x_hat, z = sae(X)
    return x_hat.cpu().numpy(), z.cpu().numpy()


# ─── Probes ─────────────────────────────────────────────────────────────────
def probe(X, y, target_name, n_splits=5, seed=0, min_count=3):
    unique, counts = np.unique(y, return_counts=True)
    valid = unique[counts >= min_count]
    mask = np.isin(y, valid)
    Xf, yf = X[mask], y[mask]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Xf, yf):
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(Xf[tr], yf[tr])
        accs.append(clf.score(Xf[te], yf[te]))
    return float(np.mean(accs)), float(np.std(accs)), len(valid), len(yf)


def bootstrap_seed_probe(X, y, target_name, seeds=(0, 1, 2, 3, 4), **kw):
    accs = [probe(X, y, target_name, seed=s, **kw)[0] for s in seeds]
    return float(np.mean(accs)), float(np.std(accs))


# ─── Controls ───────────────────────────────────────────────────────────────
def pca_truncate(X, rank, seed=0):
    """Project X to its top-`rank` PCA directions and reconstruct."""
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt_k = Vt[:rank]
    proj = Xc @ Vt_k.T
    recon = proj @ Vt_k + X.mean(axis=0, keepdims=True)
    return recon


def random_projection_truncate(X, rank, seed=0):
    """Project X through a random orthonormal rank-`rank` subspace and reconstruct."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    R = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(R)
    proj = X @ Q
    recon = proj @ Q.T
    return recon


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed={SEED}")

    print("Loading model + dataset...")
    model = load_model(device)
    dataset_hf = load_dataset("sojup/entity_binding", split="test")
    dataset = EntityBindingDataset(dataset_hf.to_pandas())

    X_t, e1, t, e2, e1t = collect_resid_post(model, dataset, device, max_examples=2500)
    X = X_t.numpy()
    print(f"  Collected {len(X)} fact activations  (d_model={X.shape[1]})")

    print("\nTraining toy SAE (d_hidden=1024, L1=3e-3, 4000 steps)...")
    sae, losses, l0s = train_sae(
        X_t, d_hidden=1024, l1=3e-3, steps=4000, batch_size=512, lr=1e-3,
        device=device, seed=SEED,
    )
    X_recon, Z = encode_decode(sae, X_t, device)
    final_l0 = float((Z > 0).sum(axis=1).mean())
    recon_var = float(1 - ((X_recon - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
    print(f"  SAE trained. Final mean L0={final_l0:.1f}  Explained-var={recon_var:.3f}")

    # Rank for controls: match SAE's effective rank (round L0)
    ctrl_rank = max(8, int(round(final_l0)))
    print(f"  Building controls with rank={ctrl_rank}")
    X_pca = pca_truncate(X, rank=ctrl_rank, seed=SEED)
    X_rp = random_projection_truncate(X, rank=ctrl_rank, seed=SEED)

    print("\n=== Probe accuracies (5-fold CV × 5 seeds) ===")
    print(f"{'Source':<22}{'(E1,T) acc':>16}{'E2 acc':>16}")
    rows = []
    for source_name, M in [
        ("raw X", X),
        ("X_recon (SAE)", X_recon),
        ("z (SAE latent)", Z),
        (f"X_pca (rank {ctrl_rank})", X_pca),
        (f"X_randproj (rank {ctrl_rank})", X_rp),
    ]:
        a_e1t, s_e1t = bootstrap_seed_probe(M, e1t, "(E1,T)")
        a_e2, s_e2 = bootstrap_seed_probe(M, e2, "E2")
        print(f"{source_name:<22}{a_e1t:>10.4f}±{s_e1t:.3f}{a_e2:>10.4f}±{s_e2:.3f}")
        rows.append((source_name, a_e1t, s_e1t, a_e2, s_e2))

    # Plot
    names = [r[0] for r in rows]
    e1t_means = [r[1] for r in rows]
    e1t_stds = [r[2] for r in rows]
    e2_means = [r[3] for r in rows]
    e2_stds = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, e1t_means, w, yerr=e1t_stds, capsize=4, label="(E1,T) composed", color="#9b59b6")
    ax.bar(x + w / 2, e2_means, w, yerr=e2_stds, capsize=4, label="E2 (payload)", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Probe accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Toy 2L2H: Probe accuracy on raw vs SAE-recon vs sparse latents vs rank-matched controls", fontsize=10, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / "toy_sae_recovery.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
