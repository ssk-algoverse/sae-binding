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
                             (also the architecture template when loading FT weights)
  • ft_checkpoint          — optional path to a finetuned HF checkpoint
                             (e.g. ``gemma/gemma2_ft_toy/checkpoint-900``). When set
                             *and the path exists*, weights are swapped into the
                             ``model_name`` template via ``hf_model=...``. This is
                             the load used by ``gemma/pp_toy_dataset.ipynb`` to
                             identify the L22H4 circuit, so probe / Q-K / SAE
                             experiments match the circuit-identification model.
                             If the path is missing we fall back to base with a
                             warning.
  • target_layer / head    — identified by ``gemma/pp_toy_dataset.ipynb``; must be
                             filled in after running path patching on a new
                             model
  • random_head_layer_range — window for exp6's random-head null
  • sae_configs            — list of (label, sae_lens_release, sae_id_template)
                             tuples. ``{layer}`` in the sae_id is substituted
                             with target_layer at runtime.
"""
from __future__ import annotations
import os
import warnings

PRESETS = {
    # ── Legacy preset (what the paper currently reports) ────────────────────
    "gemma-2-2b": {
        "model_name": "gemma-2-2b",
        # FT checkpoint produced by gemma/gemma_toy_ft.ipynb. This is the model
        # gemma/pp_toy_dataset.ipynb uses for path patching, so downstream
        # experiments should match. Falls back to base Gemma-2-2b if the path
        # doesn't exist (prints a warning).
        "ft_checkpoint": "gemma/gemma2_ft_toy/checkpoint-900",
        # Circuit head identified by gemma/pp_toy_dataset.ipynb
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
    # path patching (gemma/pp_toy_dataset.ipynb) on the corresponding model.
    # Until then, exp6 will raise a clear error.
    #
    # SAE release IDs: verify against https://www.neuronpedia.org/gemma-scope-2
    # or ``python -c "from sae_lens.toolkit.pretrained_saes_directory import
    # get_pretrained_saes_directory as g; print([k for k in g() if 'gemma-scope-2' in k])"``.
    "gemma-3-1b-pt": {
        "model_name": "gemma-3-1b-pt",
        "ft_checkpoint": None,   # no FT yet — add once gemma_toy_ft is re-run on Gemma-3
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
        "ft_checkpoint": None,
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


def load_model(preset: dict, device, **hooked_kwargs):
    """Load a HookedTransformer for the given preset.

    If the preset has ``ft_checkpoint`` pointing to an existing local directory,
    we load HF weights from there and swap them into a ``model_name`` HookedTransformer
    via ``hf_model=...`` (same pattern as gemma/pp_toy_dataset.ipynb). Otherwise we
    call ``HookedTransformer.from_pretrained(model_name)`` on the base weights.

    Extra keyword args are passed through to ``from_pretrained``. Sensible
    defaults (``center_unembed=True, center_writing_weights=True, fold_ln=True``)
    are applied if the caller doesn't override them.
    """
    from transformer_lens import HookedTransformer

    defaults = dict(
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )
    defaults.update(hooked_kwargs)

    model_name = preset["model_name"]
    ft_ckpt = preset.get("ft_checkpoint")

    if ft_ckpt and os.path.isdir(ft_ckpt):
        from transformers import AutoModelForCausalLM
        print(f"Loading FT checkpoint from {ft_ckpt} into {model_name} architecture...")
        hf_model = AutoModelForCausalLM.from_pretrained(ft_ckpt)
        model = HookedTransformer.from_pretrained(model_name, hf_model=hf_model, **defaults)
    else:
        if ft_ckpt:
            warnings.warn(
                f"Preset '{preset['_name']}' declares ft_checkpoint='{ft_ckpt}' but "
                f"that directory doesn't exist. Falling back to base {model_name}. "
                f"NOTE: the circuit claims (L{preset.get('target_layer')}H"
                f"{preset.get('target_head')}) were identified on the FT model; "
                f"results on base weights may not reproduce them."
            )
        print(f"Loading pretrained {model_name} (base weights)...")
        model = HookedTransformer.from_pretrained(model_name, **defaults)

    model.eval()
    return model
