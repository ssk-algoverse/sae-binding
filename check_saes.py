import torch
from sae_lens import SAE

# List available SAEs for gemma-2-2b
try:
    print("Trying to list SAEs for gemma-scope-2b-pt-res...")
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    print(d["gemma-scope-2b-pt-res"].saes.keys())
except Exception as e:
    print("Error:", e)
