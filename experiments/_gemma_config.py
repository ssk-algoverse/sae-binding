"""Centralized model preset for Gemma experiments.

Switch models via the ``GEMMA_PRESET`` env var (defaults to the legacy
Gemma-2-2b preset so existing workflows stay unchanged):

    python experiments/exp4_gemma_linear_probes.py                       # gemma-2-2b (default)
    GEMMA_PRESET=gemma-3-1b-pt python experiments/exp4_gemma_linear_probes.py
    GEMMA_PRESET=gemma-3-4b-pt python experiments/exp7_gemma_sae_recovery.py

Architectural constants (n_layers, n_heads, GQA ratio) are NOT hard-coded
here — they are read from ``model.cfg`` after the HookedTransformer load.
That way the config only needs to know the user-meaningful knobs:

  • model_name             — argument to HookedTransformer.from_pretrained
  • target_layer / head    — identified by PathPatchingGemma.ipynb; must be
                             filled in after running path patching on a new
                             model
  • random_head_layer_range — window for exp6's random-head null
  • sae_configs            — list of (label, sae_lens_release, sae_id_template)
                             tuples. ``{layer}`` in the sae_id is substituted
                             with target_layer at runtime.
"""
from __future__ import annotations
import os

PRESETS = {
    # ── Legacy preset (what the paper currently reports) ────────────────────
    "gemma-2-2b": {
        "model_name": "gemma-2-2b",
        # Circuit head identified by PathPatchingGemma.ipynb
        "target_layer": 22,
        "target_head": 4,
        # Deep half of the network, minus the target head itself
        "random_head_layer_range": (18, 26),
        # Gemma-Scope 1 releases
        "sae_configs": [
            ("gs1 / width_16k / canonical (L0~72)",
             "gemma-scope-2b-pt-res-canonical",
             "layer_{layer}/width_16k/canonical"),
            ("gs1 / width_16k / l0~22 (sparser)",
             "gemma-scope-2b-pt-res",
             "layer_{layer}/width_16k/average_l0_22"),
        ],
    },

    # ── Gemma 3 presets ─────────────────────────────────────────────────────
    # NOTE: target_layer/target_head MUST be filled in after re-running
    # PathPatchingGemma.ipynb on the corresponding model. Until then, exp6
    # will raise a clear error.
    #
    # SAE release IDs: verify against https://www.neuronpedia.org/gemma-scope-2
    # or ``python -c "from sae_lens.toolkit.pretrained_saes_directory import
    # get_pretrained_saes_directory as g; print([k for k in g() if 'gemma-scope-2' in k])"``.
    "gemma-3-1b-pt": {
        "model_name": "gemma-3-1b-pt",
        "target_layer": None,
        "target_head": None,
        "random_head_layer_range": None,
        "sae_configs": [
            # Placeholder — confirm exact release/id against Neuronpedia.
            ("gs2 / width_16k / canonical",
             "gemma-scope-2-1b-pt-res",
             "layer_{layer}/width_16k/canonical"),
        ],
    },
    "gemma-3-4b-pt": {
        "model_name": "gemma-3-4b-pt",
        "target_layer": None,
        "target_head": None,
        "random_head_layer_range": None,
        "sae_configs": [
            ("gs2 / width_16k / canonical",
             "gemma-scope-2-4b-pt-res",
             "layer_{layer}/width_16k/canonical"),
        ],
    },
}


def get_preset(name: str | None = None) -> dict:
    if name is None:
        name = os.environ.get("GEMMA_PRESET", "gemma-2-2b")
    if name not in PRESETS:
        raise ValueError(
            f"Unknown GEMMA_PRESET='{name}'. Known: {list(PRESETS)}"
        )
    preset = dict(PRESETS[name])
    preset["_name"] = name
    return preset


def require(preset: dict, *keys: str) -> None:
    """Raise with a helpful message if required preset fields are None."""
    missing = [k for k in keys if preset.get(k) is None]
    if missing:
        raise RuntimeError(
            f"Preset '{preset['_name']}' is missing required fields {missing}. "
            f"Fill them in at experiments/_gemma_config.py — these normally "
            f"come from re-running PathPatchingGemma.ipynb on the new model."
        )


def model_arch(model) -> dict:
    """Read architecture dims from a loaded HookedTransformer.

    Handles GQA: returns `n_q_heads`, `n_kv_heads`, `group_size` where
    `group_size = n_q_heads // n_kv_heads` (=1 for MHA models).
    """
    n_q = model.cfg.n_heads
    n_kv = getattr(model.cfg, "n_key_value_heads", None) or n_q
    return {
        "n_layers": model.cfg.n_layers,
        "n_q_heads": n_q,
        "n_kv_heads": n_kv,
        "group_size": max(1, n_q // n_kv),
        "d_model": model.cfg.d_model,
    }


def kv_head_for(q_head: int, group_size: int) -> int:
    """Map a query-head index to its shared KV-head index under GQA."""
    return q_head // group_size


def resolve_sae_configs(preset: dict, layer: int) -> list[tuple[str, str, str]]:
    """Substitute {layer} into each sae_id template."""
    out = []
    for label, release, template in preset["sae_configs"]:
        out.append((label, release, template.format(layer=layer)))
    return out
