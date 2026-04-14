"""
Experiment 7: Gemma SAE Feature Recovery analysis
=================================================
Tests whether Gemma-Scope SAEs suffer from the 'dark matter' failure mode.
Compares Ridge Classifier decoding accuracy for generic entities (Payload E2) 
vs composed relational structures (Address E1+T) on raw resid_post 
versus SAE reconstructed resid_post.
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
from sae_lens import SAE
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def load_sae(device, layer=22):
    print(f"Loading Gemma-Scope SAE for Layer {layer}...")
    try:
        # Canonical names for gemma scope
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=device
        )
    except Exception as e:
        print(f"Failed to load canonical, trying standard: {e}")
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res",
            sae_id=f"layer_{layer}/width_16k/average_l0_72",
            device=device
        )
    return sae

def get_data(model, sae, layer=22, max_prompts=300):
    device = model.cfg.device
    print(f"Loading up to {max_prompts} prompts...")
    
    with open("gemma/gemma_toy_dataset_train.jsonl", "r") as f:
        lines = [json.loads(next(f)) for _ in range(max_prompts)]
        
    comma_id = model.to_single_token(",")
    period_id = model.to_single_token(".")
    
    raw_acts = []
    recon_acts = []
    metadata = {"e2": [], "e1_t": []}
    
    filter_name = f"blocks.{layer}.hook_resid_post"
    
    print(f"Running forward passes to collect Layer {layer} activations...")
    for line in tqdm(lines):
        inp = line["input"]
        tokens = model.to_tokens(inp).squeeze(0)
        
        # Extract facts manually from tokens
        tokens_np = tokens.cpu().numpy()
        sep_positions = np.where((tokens_np == comma_id) | (tokens_np == period_id))[0]
        
        valid_facts = []
        for sp in sep_positions:
            if sp < 3: continue
            e1 = int(tokens_np[sp - 3])
            t_rel = int(tokens_np[sp - 2])
            e2 = int(tokens_np[sp - 1])
            valid_facts.append((sp, e1, t_rel, e2))
            
        if not valid_facts: continue
        
        _, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=[filter_name])
        acts = cache[filter_name][0] # [seq, d_model]
        
        for sp, e1, t_rel, e2 in valid_facts:
            raw_x = acts[sp] # [d_model]
            
            with torch.no_grad():
                recon_x = sae(raw_x.unsqueeze(0))
                
            raw_acts.append(raw_x.cpu().numpy())
            recon_acts.append(recon_x.squeeze(0).cpu().numpy())
            
            metadata["e2"].append(e2)
            metadata["e1_t"].append(f"{e1}_{t_rel}")
            
    raw_acts = np.stack(raw_acts)
    recon_acts = np.stack(recon_acts)
    for k in metadata: metadata[k] = np.array(metadata[k])
    
    return raw_acts, recon_acts, metadata

def train_probe(X, labels, target_name, data_type):
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= 3]
    mask = np.isin(labels, valid_classes)
    
    X_f = X[mask]
    y_f = labels[mask]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_f, y_f):
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_f[train_idx], y_f[train_idx])
        acc = clf.score(X_f[test_idx], y_f[test_idx])
        accs.append(acc)
        
    final_acc = np.mean(accs)
    print(f"[{target_name:<5} | {data_type:<15}] Accuracy: {final_acc*100:.1f}% (Classes: {len(valid_classes)})")
    return final_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading pretrained Gemma-2-2b...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", center_unembed=True, center_writing_weights=True, fold_ln=True, device=device)
    model.eval()
    
    # Analyze layer 22 where retrieval happens
    sae = load_sae(device, layer=22)
    
    raw_acts, recon_acts, metadata = get_data(model, sae, layer=22, max_prompts=1500)
    
    targets = ["e2", "e1_t"]
    results = {"raw": {}, "recon": {}}
    
    for t in targets:
        print(f"\\n--- Probing {t} ---")
        acc_raw = train_probe(raw_acts, metadata[t], t, "Raw Activations")
        acc_recon = train_probe(recon_acts, metadata[t], t, "SAE Reconstructed")
        results["raw"][t] = acc_raw
        results["recon"][t] = acc_recon
        
    # Plot results
    labels = ['Payload ($E_2$)', 'Composed Address ($E_1, T$)']
    raw_scores = [results['raw']['e2'], results['raw']['e1_t']]
    recon_scores = [results['recon']['e2'], results['recon']['e1_t']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, raw_scores, width, label='Raw Activations', color='#34495e')
    rects2 = ax.bar(x + width/2, recon_scores, width, label='SAE Reconstructed', color='#e74c3c')

    ax.set_ylabel('Linear Probe Accuracy')
    ax.set_title('Gemma Scope SAE "Dark Matter" Failure (Layer 22)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.1)

    for r in rects1 + rects2:
        h = r.get_height()
        ax.annotate(f'{h*100:.1f}%', xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    out_path = "experiments/results/gemma/gemma_sae_recovery.png"
    plt.savefig(out_path, dpi=150)
    print(f"\\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()
