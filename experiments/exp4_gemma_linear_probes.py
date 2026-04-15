"""
Experiment 4: Linear Probes on Gemma Fine-Tuned
=================================================
Trains linear probes on residual stream activations (`blocks.L.hook_resid_post`)
across all 26 layers in Gemma-2-2b to decode E1, T, E2, and (E1,T) from fact separator positions.
"""

import os
import json
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

OUTPUT_DIR = "experiments/results/gemma"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_LAYERS = 26

def load_model(device):
    print("Loading pretrained Gemma-2-2b...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", center_unembed=True, center_writing_weights=True, fold_ln=True, device=device)
    model.eval()
    return model

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

def collect_activations(model, max_prompts=300):
    device = model.cfg.device
    print(f"Loading up to {max_prompts} prompts...")
    
    with open("gemma/gemma_toy_dataset_train.jsonl", "r") as f:
        lines = [json.loads(next(f)) for _ in range(max_prompts)]
        
    comma_id = model.to_single_token(",")
    period_id = model.to_single_token(".")
    
    # Store activations: layer -> list of tensors
    # Since d_model = 2304, allocating large arrays.
    layer_acts = {l: [] for l in range(N_LAYERS)}
    metadata = {"e1": [], "t": [], "e2": [], "e1_t": []}
    
    print("Running forward passes and caching resid_post...")
    for idx, line in enumerate(tqdm(lines)):
        inp = line["input"]
        tokens = model.to_tokens(inp).squeeze(0)
        facts = extract_facts_from_tokens(tokens.cpu().numpy(), comma_id, period_id)
        
        if not facts: continue
            
        sep_positions = [f["sep_pos"] for f in facts]
        
        # We only need resid_post for all layers
        filter_names = [f"blocks.{l}.hook_resid_post" for l in range(N_LAYERS)]
        _, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=filter_names)
        
        for fact in facts:
            sp = fact["sep_pos"]
            for l in range(N_LAYERS):
                act = cache[f"blocks.{l}.hook_resid_post"][0, sp, :].cpu().numpy()
                layer_acts[l].append(act)
                
            metadata["e1"].append(fact["e1"])
            metadata["t"].append(fact["t"])
            metadata["e2"].append(fact["e2"])
            metadata["e1_t"].append(fact["e1_t"])
    
    # Convert to numpy arrays
    for l in range(N_LAYERS):
        layer_acts[l] = np.stack(layer_acts[l])
        
    for k in metadata:
        metadata[k] = np.array(metadata[k])
        
    return layer_acts, metadata

def train_linear_probes(layer_acts, labels, target_name):
    """
    Trains logistic regression probes across all layers using 3-fold CV.
    Filters out classes with fewer than 3 instances for stable CV.
    """
    # Filter rare classes
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= 3]
    mask = np.isin(labels, valid_classes)
    
    filtered_labels = labels[mask]
    print(f"  [{target_name}] Training probes on {len(filtered_labels)}/{len(labels)} facts ({len(valid_classes)} classes)")
    
    layer_accs = []
    
    for l in tqdm(range(N_LAYERS), desc=f"Probing {target_name}"):
        X = layer_acts[l][mask]
        y = filtered_labels
        
        # Calculate CV accuracy
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X[train_idx], y[train_idx])
            acc = clf.score(X[test_idx], y[test_idx])
            accs.append(acc)
            
        layer_accs.append(np.mean(accs))
        
    return layer_accs

def plot_probe_accuracies(acc_dict, filename):
    plt.figure(figsize=(10, 6))
    colors = {"e1": "#3498db", "t": "#2ecc71", "e2": "#e74c3c", "e1_t": "#9b59b6"}
    labels_clean = {"e1": "Entity 1 (Payload)", "t": "Relation", "e2": "Entity 2", "e1_t": "Composed Address (E1, T)"}
    
    for target in ["e1", "t", "e2", "e1_t"]:
        plt.plot(range(N_LAYERS), acc_dict[target], marker='o', label=labels_clean[target], color=colors[target], linewidth=2)
        
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.title("Gemma-2-2b: Representation Factorization at Fact Separators", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.xticks(range(0, N_LAYERS, 2))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    layer_acts, metadata = collect_activations(model, max_prompts=150) # 150 prompts ~ 1200 facts
    
    results = {}
    for target in ["e1", "t", "e2", "e1_t"]:
        print(f"\\n=== Probing target: {target} ===")
        accs = train_linear_probes(layer_acts, metadata[target], target)
        results[target] = accs
        
    plot_probe_accuracies(results, os.path.join(OUTPUT_DIR, "gemma_probe_accuracies.png"))
    
    # Save raw array data for later analysis
    np.savez(os.path.join(OUTPUT_DIR, "gemma_probe_results.npz"), **results)
    
if __name__ == "__main__":
    main()
