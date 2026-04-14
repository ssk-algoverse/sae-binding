"""
Experiment 1: Combinatorial Geometry Analysis
==============================================
Computes pairwise cosine similarities of:
  (a) Isolated E1 vectors from L0H0 output (hook_z, head 0)
  (b) Isolated T vectors from L0H0 output (hook_z, head 0)
  (c) Composed (E1+T) vectors from resid_post
  (d) Isolated E2 vectors from L0H1 output (hook_z, head 1)

Then visualises the distributions and optionally produces PCA/UMAP projections.

Expected result: composed (E1+T) vectors cluster tightly with high cosine similarity,
explaining why SAEs cannot separate them into discrete features.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ─── Constants ──────────────────────────────────────────────────────────────
MODEL_REPO_ID = "sebastianhoenig/2L2H_Final"
MODEL_FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"

E = 100  # entities
T = 10  # relations
SEP = E + T  # 110
Q = E + T + 1  # 111
PAD = E + T + 2  # 112
D_VOCAB = E + T + 3
N_LAYERS = 2
HEADS = 2
D_MODEL = 256
N_CTX = 64

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Model ──────────────────────────────────────────────────────────────────
def build_model():
    d_head = D_MODEL // HEADS
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        n_heads=HEADS,
        d_model=D_MODEL,
        d_head=d_head,
        n_ctx=N_CTX,
        d_vocab=D_VOCAB,
        d_vocab_out=E,
        attn_only=True,
        normalization_type="LN",
        positional_embedding_type="rotary",
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


# ─── Dataset ────────────────────────────────────────────────────────────────
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


# ─── Activation extraction ──────────────────────────────────────────────────
def extract_fact_metadata(tokens):
    """Parse token sequence into list of fact dicts with positions and labels."""
    tokens_np = tokens.cpu().numpy()
    sep_positions = np.where(tokens_np == SEP)[0]
    facts = []
    for sep_pos in sep_positions:
        sep_pos = int(sep_pos)
        if sep_pos < 3:
            continue
        e1 = int(tokens_np[sep_pos - 3])
        t = int(tokens_np[sep_pos - 2])
        e2 = int(tokens_np[sep_pos - 1])
        if e1 < E and (E <= t < E + T) and e2 < E:
            facts.append({
                "sep_pos": sep_pos,
                "e1": e1,
                "t": t - E,  # relation index 0..9
                "e2": e2,
                "e1_t": e1 * T + (t - E),  # joint label
            })
    return facts


@torch.no_grad()
def collect_activations(model, dataset, device, max_examples=2000):
    """Collect per-fact activations at SEP/comma positions."""
    hookz_h0_vecs = []  # L0H0 output at SEP
    hookz_h1_vecs = []  # L0H1 output at SEP
    resid_post_vecs = []  # resid_post at SEP
    e1_labels = []
    t_labels = []
    e2_labels = []
    e1t_labels = []

    n = min(len(dataset), max_examples)
    for idx in tqdm(range(n), desc="Collecting activations"):
        tokens, label = dataset[idx]
        tokens = tokens.to(device)

        facts = extract_fact_metadata(tokens)
        if not facts:
            continue

        _, cache = model.run_with_cache(tokens.unsqueeze(0))

        hookz = cache["blocks.0.attn.hook_z"]  # [1, seq, heads, d_head]
        resid_post = cache["blocks.0.hook_resid_post"]  # [1, seq, d_model]

        for fact in facts:
            sp = fact["sep_pos"]
            hookz_h0_vecs.append(hookz[0, sp, 0, :].cpu())
            hookz_h1_vecs.append(hookz[0, sp, 1, :].cpu())
            resid_post_vecs.append(resid_post[0, sp, :].cpu())
            e1_labels.append(fact["e1"])
            t_labels.append(fact["t"])
            e2_labels.append(fact["e2"])
            e1t_labels.append(fact["e1_t"])

    return {
        "hookz_h0": torch.stack(hookz_h0_vecs),
        "hookz_h1": torch.stack(hookz_h1_vecs),
        "resid_post": torch.stack(resid_post_vecs),
        "e1": np.array(e1_labels),
        "t": np.array(t_labels),
        "e2": np.array(e2_labels),
        "e1t": np.array(e1t_labels),
    }


# ─── Geometry analysis ──────────────────────────────────────────────────────
def compute_class_mean_cosine_matrix(vecs, labels):
    """Compute cosine similarity between class-mean vectors."""
    unique = np.unique(labels)
    means = []
    for c in unique:
        mask = labels == c
        class_vecs = vecs[mask]
        mean = class_vecs.mean(dim=0)
        means.append(mean)
    means = torch.stack(means)
    # Normalise
    means_norm = means / means.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine_matrix = means_norm @ means_norm.T
    return cosine_matrix.numpy(), unique


def compute_pairwise_cosine_stats(vecs, labels, n_sample_pairs=50000):
    """Sample random pairs and compute intra-class and inter-class cosine similarities."""
    vecs_norm = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n = len(labels)

    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=n_sample_pairs)
    idx_b = rng.integers(0, n, size=n_sample_pairs)
    # Remove self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    cos_sim = (vecs_norm[idx_a] * vecs_norm[idx_b]).sum(dim=-1).numpy()
    same_class = labels[idx_a] == labels[idx_b]

    intra = cos_sim[same_class]
    inter = cos_sim[~same_class]
    return intra, inter


def compute_mean_pairwise_cosine(vecs):
    """Compute mean pairwise cosine similarity of all vectors (sampled)."""
    vecs_norm = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n = min(len(vecs), 5000)
    subset = vecs_norm[:n]
    cos_matrix = subset @ subset.T
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool)
    return cos_matrix[mask].mean().item(), cos_matrix[mask].std().item()


# ─── Plotting ───────────────────────────────────────────────────────────────
def plot_cosine_distributions(results, filename):
    """Plot histogram of intra- vs inter-class cosine similarities."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cosine Similarity: Intra-class vs Inter-class", fontsize=14, fontweight="bold")

    for ax, (title, intra, inter) in zip(axes.flat, results):
        ax.hist(inter, bins=80, alpha=0.6, label=f"Inter-class (μ={inter.mean():.3f})", color="gray", density=True)
        ax.hist(intra, bins=80, alpha=0.6, label=f"Intra-class (μ={intra.mean():.3f})", color="royalblue", density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_mean_cosine_bar(results, filename):
    """Bar chart of mean pairwise cosine similarity across representations."""
    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylabel("Mean Pairwise Cosine Similarity")
    ax.set_title("Geometric Density: Mean Pairwise Cosine Similarity\n(Higher = more clustered → harder for SAEs)", fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{m:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# ─── PCA visualisation ──────────────────────────────────────────────────────
def plot_pca_comparison(data, filename):
    """2D PCA of isolated E1 (from hookz_h0) vs composed (E1+T) (from resid_post), colored by E1."""
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PCA Projections Colored by Entity ($E_1$)", fontsize=13, fontweight="bold")

    configs = [
        ("L0H0: Isolated $E_1$", data["hookz_h0"], data["e1"]),
        ("L0H0: Isolated $T$", data["hookz_h0"], data["t"]),
        ("resid_post: Composed $(E_1 + T)$", data["resid_post"], data["e1"]),
    ]

    for ax, (title, vecs, labels) in zip(axes, configs):
        pca = PCA(n_components=2)
        proj = pca.fit_transform(vecs[:3000].numpy())
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels[:3000], cmap="tab20", s=3, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(device)

    print("Loading dataset...")
    dataset = load_data()

    print("Collecting activations...")
    data = collect_activations(model, dataset, device, max_examples=2000)

    n_facts = len(data["e1"])
    print(f"Collected {n_facts} fact activations")

    # ── Analysis 1: Mean pairwise cosine similarity ──
    print("\n=== Mean Pairwise Cosine Similarity ===")
    bar_results = []
    for name, vecs in [
        ("E1 (L0H0)", data["hookz_h0"]),
        ("T (L0H0)", data["hookz_h0"]),
        ("E2 (L0H1)", data["hookz_h1"]),
        ("(E1,T) composed\n(resid_post)", data["resid_post"]),
    ]:
        mean_cos, std_cos = compute_mean_pairwise_cosine(vecs)
        print(f"  {name:30s} → mean={mean_cos:.4f}  std={std_cos:.4f}")
        bar_results.append((name, mean_cos, std_cos))

    plot_mean_cosine_bar(bar_results, OUTPUT_DIR / "mean_pairwise_cosine.png")

    # ── Analysis 2: Intra- vs inter-class cosine similarity ──
    print("\n=== Intra- vs Inter-class Cosine Similarity ===")
    dist_results = []
    configs = [
        ("E1 classes (L0H0 hookz, head 0)", data["hookz_h0"], data["e1"]),
        ("T classes (L0H0 hookz, head 0)", data["hookz_h0"], data["t"]),
        ("E2 classes (L0H1 hookz, head 1)", data["hookz_h1"], data["e2"]),
        ("(E1,T) classes (resid_post)", data["resid_post"], data["e1t"]),
    ]
    for title, vecs, labels in configs:
        intra, inter = compute_pairwise_cosine_stats(vecs, labels)
        print(f"  {title:45s} → intra μ={intra.mean():.4f}  inter μ={inter.mean():.4f}  gap={intra.mean() - inter.mean():.4f}")
        dist_results.append((title, intra, inter))

    plot_cosine_distributions(dist_results, OUTPUT_DIR / "cosine_distributions.png")

    # ── Analysis 3: Class-mean cosine similarity matrices ──
    print("\n=== Class-Mean Cosine Similarity ===")
    for name, vecs, labels in [
        ("E1 class means (hookz h0)", data["hookz_h0"], data["e1"]),
        ("(E1,T) class means (resid_post)", data["resid_post"], data["e1t"]),
    ]:
        cosine_mat, unique = compute_class_mean_cosine_matrix(vecs, labels)
        off_diag = cosine_mat[~np.eye(len(unique), dtype=bool)]
        print(f"  {name:40s} → off-diag mean cosine={off_diag.mean():.4f}  max={off_diag.max():.4f}")

    # ── Analysis 4: PCA projection ──
    print("\nGenerating PCA visualisation...")
    plot_pca_comparison(data, OUTPUT_DIR / "pca_comparison.png")

    print("\n✅ Experiment 1 complete. Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
