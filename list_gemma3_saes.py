"""Enumerate SAE IDs available in the gemma-scope-2-1b-pt-res release.

Usage:
    .venv/bin/python list_gemma3_saes.py
"""
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

RELEASE = "gemma-scope-2-1b-pt-res"

d = get_pretrained_saes_directory()
if RELEASE not in d:
    print(f"Release '{RELEASE}' not found.")
    print("Releases containing 'gemma-scope-2':")
    for k in d:
        if "gemma-scope-2" in k:
            print(f"  {k}")
    raise SystemExit(1)

entry = d[RELEASE]
ids = list(entry.saes_map.keys())
print(f"Release: {RELEASE}")
print(f"Total IDs: {len(ids)}")

# Group by layer
by_layer = {}
for sid in ids:
    # IDs look like "layer_13_width_16k_l0_big"
    parts = sid.split("_")
    try:
        layer = int(parts[parts.index("layer") + 1])
    except (ValueError, IndexError):
        layer = -1
    by_layer.setdefault(layer, []).append(sid)

print("\nLayers with SAEs:")
for layer in sorted(by_layer):
    print(f"  L{layer}: {by_layer[layer]}")
