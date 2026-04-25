import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
import einops

# Add experiments dir to path so we can import _gemma_config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from _gemma_config import get_preset, load_model

def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions):
    ln_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    # apply_ln_to_stack runs LN in fp32 for stability; logit_diff_directions
    # comes from W_U in bf16. Promote both to fp32 for the einsum.
    ln_residual_stack = ln_residual_stack.to(torch.float32)
    logit_diff_directions = logit_diff_directions.to(torch.float32)
    average_logit_diff = einops.einsum(ln_residual_stack, logit_diff_directions, "... batch d_model, batch d_model ->...") / residual_stack.shape[0]
    return average_logit_diff

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset_name = os.environ.get("GEMMA_PRESET", "gemma-2-2b")
    preset = get_preset(preset_name)
    model = load_model(preset, device)
    
    print(f"Loading dataset for {preset_name}...")
    with open("gemma/gemma_pp_dataset.jsonl", "r") as f:
        lines = [json.loads(line) for line in f]

    # Held-out split: we want the FIRST 50% for selection
    SEED = 0
    HOLDOUT_FRAC = 0.50
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(lines))
    n_holdout = int(round(len(lines) * HOLDOUT_FRAC))
    selection_idx = set(perm[n_holdout:].tolist()) # The other half (not holdout)
    selection_lines = [lines[i] for i in sorted(selection_idx)]
    
    print(f"Total prompts: {len(lines)}, Selection set: {len(selection_lines)}")

    clean_inputs = []
    corrupt_inputs = []
    correct_token_ids = []
    incorrect_token_ids = []

    for line in selection_lines:
        clean_inputs.append(line["clean"])
        corrupt_inputs.append(line["corrupt"])
        correct_token_ids.append(model.to_single_token(line["label"]))

    # We need incorrect_token_ids. Let's just run the corrupt inputs and take the argmax 
    # to mimic the original logic, or if we know the corrupt label... 
    # Let's get the max prediction for the corrupt input.
    BATCH_SIZE = 8
    for i in tqdm(range(0, len(corrupt_inputs), BATCH_SIZE), desc="Getting corrupt preds"):
        batch = corrupt_inputs[i:i+BATCH_SIZE]
        tokens = model.to_tokens(batch, prepend_bos=False).to(device)
        with torch.no_grad():
            logits = model(tokens)
            preds = logits[:, -1, :].argmax(dim=-1).tolist()
            incorrect_token_ids.extend(preds)

    print("Computing per-head logit diffs...")
    per_head_logit_diffs_list = []
    
    # Process in batches to save memory
    for i in tqdm(range(0, len(clean_inputs), BATCH_SIZE), desc="Batches"):
        batch_clean = clean_inputs[i:i+BATCH_SIZE]
        batch_correct = correct_token_ids[i:i+BATCH_SIZE]
        batch_incorrect = incorrect_token_ids[i:i+BATCH_SIZE]
        
        tokens = model.to_tokens(batch_clean, prepend_bos=False).to(device)
        
        with torch.no_grad():
            clean_logits, cache = model.run_with_cache(tokens)
            
            correct_token_directions = model.tokens_to_residual_directions(torch.tensor(batch_correct, device=device))
            incorrect_token_directions = model.tokens_to_residual_directions(torch.tensor(batch_incorrect, device=device))
            logit_diff_directions = correct_token_directions - incorrect_token_directions
            
            per_head_residual, _ = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
            per_head_residual = einops.rearrange(
                per_head_residual,
                "(layer head) ... -> layer head ...",
                layer=model.cfg.n_layers
            )
            # shape: [layer, head, batch, d_model]
            
            # calculate diff for this batch
            diff = residual_stack_to_logit_diff(per_head_residual, cache, logit_diff_directions)
            # diff is [layer, head] but we averaged over batch.
            # actually residual_stack_to_logit_diff averages over batch: "... batch d_model -> ..." / batch_size
            per_head_logit_diffs_list.append(diff * len(batch_clean)) # multiply back batch size

    # weighted average
    total_samples = len(clean_inputs)
    final_per_head = sum(per_head_logit_diffs_list) / total_samples

    out_file = "gemma/per_head_logit_diffs.pt"
    if preset_name != "gemma-2-2b":
        out_file = f"gemma/per_head_logit_diffs_{preset_name}.pt"
        
    torch.save(final_per_head.cpu(), out_file)
    print(f"Saved {out_file} with max diff: {final_per_head.max().item():.4f}")
    
    # Print the top head
    max_idx = final_per_head.argmax().item()
    max_layer = max_idx // model.cfg.n_heads
    max_head = max_idx % model.cfg.n_heads
    print(f"Top Head: L{max_layer}H{max_head}")

if __name__ == "__main__":
    main()
