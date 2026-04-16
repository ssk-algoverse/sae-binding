"""
Experiment 2: QK and OV Weight Matrix Analysis
================================================
Performs SVD/eigendecomposition of:
  - W_Q @ W_K^T for Layer 1 heads (retrieval layer)
  - W_O @ W_V for Layer 1 heads

Goal: Show that the QK circuit is structurally hardcoded to match on the address subspace,
and the OV circuit acts as identity on the payload subspace.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

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


def extract_fact_metadata(tokens):
    tokens_np = tokens.cpu().numpy()
    sep_positions = np.where(tokens_np == SEP)[0]
    facts = []
    for sep_pos in sep_positions:
        sep_pos = int(sep_pos)
        if sep_pos < 3:
            continue
        e1 = int(tokens_np[sep_pos - 3])
        t_val = int(tokens_np[sep_pos - 2])
        e2 = int(tokens_np[sep_pos - 1])
        if e1 < E and (E <= t_val < E + T) and e2 < E:
            facts.append({
                "sep_pos": sep_pos,
                "e1": e1,
                "t": t_val - E,
                "e2": e2,
                "e1_t": e1 * T + (t_val - E),
            })
    return facts


def analyze_qk_circuit(model, device):
    """Analyze W_Q @ W_K^T for Layer 1 heads."""
    print("\n" + "=" * 60)
    print("QK Circuit Analysis (Layer 1)")
    print("=" * 60)

    for head_idx in range(HEADS):
        W_Q = model.W_Q[1, head_idx]  # [d_model, d_head]
        W_K = model.W_K[1, head_idx]  # [d_model, d_head]

        # Full QK matrix: d_model x d_model
        QK = W_Q @ W_K.T  # [d_model, d_model]

        # Symmetric part for eigenanalysis
        QK_sym = (QK + QK.T) / 2

        eigenvalues, eigenvectors = torch.linalg.eigh(QK_sym.float().detach().cpu())
        eigenvalues = eigenvalues.detach().numpy()

        print(f"\n  Head L1H{head_idx}:")
        print(f"    QK matrix shape: {QK.shape}")
        print(f"    Top 10 eigenvalues:  {eigenvalues[-10:][::-1]}")
        print(f"    Bottom 10 eigenvalues: {eigenvalues[:10]}")
        print(f"    Effective rank (evals > 1% of max): "
              f"{(np.abs(eigenvalues) > 0.01 * np.abs(eigenvalues).max()).sum()}")

        # SVD of QK
        U, S, Vh = torch.linalg.svd(QK.float().detach().cpu())
        print(f"    Top 10 singular values: {S[:10].detach().numpy()}")
        print(f"    Explained variance (top-10): {(S[:10]**2).sum() / (S**2).sum():.4f}")

    return


def analyze_ov_circuit(model, device):
    """Analyze W_O @ W_V for Layer 1 heads — should act as identity on payload."""
    print("\n" + "=" * 60)
    print("OV Circuit Analysis (Layer 1)")
    print("=" * 60)

    for head_idx in range(HEADS):
        W_V = model.W_V[1, head_idx]  # [d_model, d_head]
        W_O = model.W_O[1, head_idx]  # [d_head, d_model]

        # OV matrix: d_model x d_model
        OV = W_V @ W_O  # [d_model, d_model]

        # How close is this to identity?
        identity = torch.eye(D_MODEL, device=OV.device)
        frobenius_diff = torch.norm(OV.detach() - identity).item()
        frobenius_ov = torch.norm(OV.detach()).item()

        # SVD
        U, S, Vh = torch.linalg.svd(OV.float().detach().cpu())

        print(f"\n  Head L1H{head_idx}:")
        print(f"    OV matrix shape: {OV.shape}")
        print(f"    Frobenius norm: {frobenius_ov:.4f}")
        print(f"    ||OV - I||_F: {frobenius_diff:.4f}")
        print(f"    Top 10 singular values: {S[:10].detach().numpy()}")
        print(f"    Singular values near 1 (0.8-1.2): {((S > 0.8) & (S < 1.2)).sum().item()}")
        print(f"    Explained variance (top-10): {(S[:10]**2).sum() / (S**2).sum():.4f}")

    return


def compute_random_baseline(d_model=D_MODEL, k=20, n_trials=2000, seed=0):
    """Monte Carlo baseline: expected principal-angle cosines between two
    uniformly random k-dim subspaces in R^{d_model}. Used as the null for
    the subspace alignment probe below."""
    rng = np.random.default_rng(seed)
    mean_sigma, align = [], []
    for _ in range(n_trials):
        A, _ = np.linalg.qr(rng.standard_normal((d_model, k)))
        B, _ = np.linalg.qr(rng.standard_normal((d_model, k)))
        s = np.linalg.svd(A.T @ B, compute_uv=False)
        mean_sigma.append(s.mean())
        align.append((s ** 2).sum() / k)
    print(f"\n  Random baseline (d={d_model}, k={k}, n_trials={n_trials}):")
    print(f"    Mean overlap sigma_bar:     {np.mean(mean_sigma):.4f} "
          f"(std {np.std(mean_sigma):.4f})")
    print(f"    Alignment score sum(s^2)/k: {np.mean(align):.4f} "
          f"(std {np.std(align):.4f})  [analytical k/d = {k / d_model:.4f}]")
    return float(np.mean(mean_sigma)), float(np.mean(align))


def probe_subspace_alignment(model, device):
    """Check whether QK top eigenvectors align with the address subspace
    by computing overlap with actual class-mean address directions."""
    print("\n" + "=" * 60)
    print("Subspace Alignment: QK eigvecs vs Address Directions")
    print("=" * 60)

    # Collect address vectors from L0H0 at SEP positions
    dataset_hf = load_dataset("sojup/entity_binding", split="test")
    dataset = EntityBindingDataset(dataset_hf.to_pandas())

    address_vecs = {}  # (e1, t) -> list of vecs
    n = min(len(dataset), 2000)

    for idx in tqdm(range(n), desc="Collecting address vectors"):
        tokens, label = dataset[idx]
        tokens = tokens.to(device)
        facts = extract_fact_metadata(tokens)
        if not facts:
            continue

        _, cache = model.run_with_cache(tokens.unsqueeze(0))
        resid_post = cache["blocks.0.hook_resid_post"]  # composed address goes here

        for fact in facts:
            key = (fact["e1"], fact["t"])
            vec = resid_post[0, fact["sep_pos"], :].cpu()
            if key not in address_vecs:
                address_vecs[key] = []
            address_vecs[key].append(vec)

    # Compute class-mean address directions
    class_means = []
    for key in sorted(address_vecs.keys()):
        if len(address_vecs[key]) >= 3:
            mean_vec = torch.stack(address_vecs[key]).mean(dim=0)
            class_means.append(mean_vec)
    class_means = torch.stack(class_means)  # [n_classes, d_model]
    print(f"  Number of (E1,T) class means: {len(class_means)}")

    # PCA of class means to get address subspace
    class_means_centered = class_means - class_means.mean(dim=0)
    U_addr, S_addr, Vh_addr = torch.linalg.svd(class_means_centered.float(), full_matrices=False)

    # Get top-k eigenvectors of QK for L1H0
    for head_idx in range(HEADS):
        W_Q = model.W_Q[1, head_idx].detach()
        W_K = model.W_K[1, head_idx].detach()
        QK = W_Q @ W_K.T
        QK_sym = (QK + QK.T) / 2

        eigenvalues, eigenvectors = torch.linalg.eigh(QK_sym.float().cpu())
        # Top eigenvectors (largest eigenvalues)
        top_eigvecs = eigenvectors[:, -20:]  # [d_model, 20]

        # Address subspace top-k directions
        top_addr = Vh_addr[:20, :].T  # [d_model, 20]

        # Compute subspace overlap (principal angles)
        overlap = torch.linalg.svdvals(top_eigvecs.T @ top_addr)
        print(f"\n  L1H{head_idx} QK top-20 eigvecs vs address subspace top-20:")
        print(f"    Singular values of overlap: {overlap[:10].detach().numpy()}")
        print(f"    Mean overlap: {overlap.mean().item():.4f}")
        print(f"    Subspace alignment score: {(overlap**2).sum().item() / min(20, 20):.4f}")


def plot_eigenvalue_spectrum(model, filename):
    """Plot eigenvalue spectrum of QK matrices for both L1 heads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eigenvalue Spectrum of $W_Q W_K^T$ (Layer 1)", fontsize=13, fontweight="bold")

    for head_idx, ax in enumerate(axes):
        W_Q = model.W_Q[1, head_idx]
        W_K = model.W_K[1, head_idx]
        QK = W_Q @ W_K.T
        QK_sym = (QK + QK.T) / 2

        eigenvalues = torch.linalg.eigvalsh(QK_sym.float().detach().cpu()).numpy()

        ax.bar(range(len(eigenvalues)), sorted(eigenvalues, reverse=True), width=1.0, color="steelblue", alpha=0.8)
        ax.set_xlabel("Eigenvalue Index (sorted)")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Head L1H{head_idx}")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_ov_singular_values(model, filename):
    """Plot singular value spectrum of OV matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Singular Value Spectrum of $W_V W_O$ (Layer 1)", fontsize=13, fontweight="bold")

    for head_idx, ax in enumerate(axes):
        W_V = model.W_V[1, head_idx]
        W_O = model.W_O[1, head_idx]
        OV = W_V @ W_O
        S = torch.linalg.svdvals(OV.float().detach().cpu()).numpy()

        ax.bar(range(len(S)), S, width=1.0, color="coral", alpha=0.8)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Identity (σ=1)")
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value")
        ax.set_title(f"Head L1H{head_idx}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(device)

    # Weight matrix analysis
    analyze_qk_circuit(model, device)
    analyze_ov_circuit(model, device)

    # Subspace alignment
    probe_subspace_alignment(model, device)

    # Random-subspace null baseline for the alignment metrics above
    compute_random_baseline(d_model=D_MODEL, k=20)

    # Plots
    plot_eigenvalue_spectrum(model, OUTPUT_DIR / "qk_eigenvalue_spectrum.png")
    plot_ov_singular_values(model, OUTPUT_DIR / "ov_singular_values.png")

    print("\n✅ Experiment 2 complete. Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
