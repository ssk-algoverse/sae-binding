import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from dictionary_learning import utils as dl_utils


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_REPO_ID = "sebastianhoenig/2L2H_Final"
MODEL_FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"

ID_REPO_ID = "sojup/entity_binding_test"
ID_FILENAME = "id_to_entity.csv"

E = 100
T = 10
SEP = E + T  # 110
Q = E + T + 1  # 111
PAD = E + T + 2  # 112
D_VOCAB = E + T + 3
N_LAYERS = 2
HEADS = 2
D_MODEL = 256
N_CTX = 64


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


def build_model(n_layers: int = N_LAYERS, n_heads: int = HEADS) -> HookedTransformer:
    d_head = D_MODEL // n_heads
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
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


def load_model_and_data(device: torch.device):
    weights_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    model = build_model().to(device)
    pretrained_weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(pretrained_weights["model"])
    model.cfg.use_attn_result = True
    model.eval()

    dataset = load_dataset("sojup/entity_binding", split="test")
    test_dataset = EntityBindingDataset(dataset.to_pandas())
    return model, test_dataset


def load_autoencoder(ckpt_relpath: str, device: torch.device):
    ckpt_path = REPO_ROOT / ckpt_relpath
    ae, cfg = dl_utils.load_dictionary(str(ckpt_path), device=device)
    ae.eval()
    return ae, cfg


def sae_reconstruct(sae, x: torch.Tensor) -> torch.Tensor:
    features = sae.encode(x)

    if hasattr(sae, "decode"):
        return sae.decode(features)

    if hasattr(sae, "decoder"):
        return sae.decoder(features)

    out = sae(x)
    if torch.is_tensor(out) and out.shape == x.shape:
        return out
    if isinstance(out, tuple):
        for item in out:
            if torch.is_tensor(item) and item.shape == x.shape:
                return item

    raise ValueError("Could not infer SAE reconstruction path.")


def make_replacement_hook(sae, hook_name: str, comma_positions: list[int], mode: str, head_idx: int | None):
    def hook_fn(act, hook):
        act = act.clone()

        if hook_name == "blocks.0.hook_resid_post":
            selected = act[0, comma_positions, :]
        elif hook_name == "blocks.0.attn.hook_z":
            if head_idx is None:
                raise ValueError("head_idx is required for hook_z replacement.")
            selected = act[0, comma_positions, head_idx, :]
        else:
            raise ValueError(f"Unsupported hook {hook_name}")

        if mode == "sae_reconstruct":
            replacement = sae_reconstruct(sae, selected)
        elif mode == "zero_ablate":
            replacement = torch.zeros_like(selected)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        if hook_name == "blocks.0.hook_resid_post":
            act[0, comma_positions, :] = replacement
        else:
            act[0, comma_positions, head_idx, :] = replacement

        return act

    return hook_fn


@torch.no_grad()
def evaluate_conditions(
    model,
    sae,
    test_dataset,
    hook_name: str,
    run_name: str,
    head_idx: int | None = None,
    max_examples: int | None = None,
):
    rows = []
    n = len(test_dataset) if max_examples is None else min(len(test_dataset), max_examples)

    model.eval()
    sae.eval()

    for idx in tqdm(range(n), desc=run_name):
        tokens, label = test_dataset[idx]
        tokens = tokens.to(model.cfg.device)
        label = int(label.item())

        comma_positions = (tokens == SEP).nonzero(as_tuple=False).flatten().tolist()
        if not comma_positions:
            continue

        clean_logits = model(tokens)

        recon_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (
                    hook_name,
                    make_replacement_hook(
                        sae,
                        hook_name=hook_name,
                        comma_positions=comma_positions,
                        mode="sae_reconstruct",
                        head_idx=head_idx,
                    ),
                )
            ],
        )

        zero_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (
                    hook_name,
                    make_replacement_hook(
                        sae,
                        hook_name=hook_name,
                        comma_positions=comma_positions,
                        mode="zero_ablate",
                        head_idx=head_idx,
                    ),
                )
            ],
        )

        for condition, logits in (
            ("clean", clean_logits),
            ("sae_reconstruct", recon_logits),
            ("zero_ablate", zero_logits),
        ):
            final_logits = logits[0, -1, :].detach()
            pred = int(final_logits.argmax().item())

            masked = final_logits.clone()
            masked[label] = float("-inf")
            best_wrong_logit = float(masked.max().item())

            rows.append(
                {
                    "run_name": run_name,
                    "condition": condition,
                    "dataset_idx": idx,
                    "label": label,
                    "pred": pred,
                    "correct": int(pred == label),
                    "correct_logit": float(final_logits[label].item()),
                    "best_wrong_logit": best_wrong_logit,
                    "margin": float(final_logits[label].item() - best_wrong_logit),
                }
            )

    return pd.DataFrame(rows)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby(["run_name", "condition"])
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_correct_logit=("correct_logit", "mean"),
            mean_best_wrong_logit=("best_wrong_logit", "mean"),
        )
        .reset_index()
    )

    clean = (
        summary[summary["condition"] == "clean"][["run_name", "accuracy", "mean_margin"]]
        .rename(columns={"accuracy": "clean_accuracy", "mean_margin": "clean_mean_margin"})
    )

    summary = summary.merge(clean, on="run_name", how="left")
    summary["delta_accuracy_vs_clean"] = summary["accuracy"] - summary["clean_accuracy"]
    summary["delta_margin_vs_clean"] = summary["mean_margin"] - summary["clean_mean_margin"]
    return summary


def run_cli(default_run_name: str, default_checkpoint: str, hook_name: str, head_idx: int | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint)
    parser.add_argument("--run-name", type=str, default=default_run_name)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_REPO_ID}/{MODEL_FILENAME}")
    model, test_dataset = load_model_and_data(device)

    print(f"Loading SAE checkpoint: {args.checkpoint}")
    sae, _ = load_autoencoder(args.checkpoint, device)

    results_df = evaluate_conditions(
        model=model,
        sae=sae,
        test_dataset=test_dataset,
        hook_name=hook_name,
        run_name=args.run_name,
        head_idx=head_idx,
        max_examples=args.max_examples,
    )
    summary_df = summarize_results(results_df)

    pd.set_option("display.max_columns", None)
    print()
    print(summary_df.to_string(index=False))

    if args.csv_out is not None:
        out_path = Path(args.csv_out)
        results_df.to_csv(out_path, index=False)
        print()
        print(f"Saved per-example results to {out_path}")

