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
        "ft_checkpoint": "gemma2_ft_toy/checkpoint-900",
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
        "ft_checkpoint": "gemma3_1b_ft_toy/checkpoint-900",
        "target_layer": None,   # fill in after running gemma/pp_toy_dataset.py
        "target_head": None,    # fill in after running gemma/pp_toy_dataset.py
        "random_head_layer_range": None,  # fill in after target_layer is known
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

    Memory-efficient load path for FT checkpoints: we extract the converted
    TL-format state_dict from the HF model and then **delete the HF model**
    before TL allocates its own parameter tensors. This avoids the usual
    peak where (hf_model + TL_model + fold_ln intermediates) are alive at
    once, which OOMs the 46.5 GiB pod on Gemma-2-2b.

    The processing sequence (fold_ln, centering) runs on CPU; the final model
    is moved to ``device`` only at the very end.
    """
    from transformer_lens import HookedTransformer
    from transformer_lens.loading_from_pretrained import (
        get_pretrained_model_config,
        get_pretrained_state_dict,
    )
    import gc
    import torch as _torch

    defaults = dict(
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=False,
        dtype=_torch.bfloat16,
    )
    # Accept caller overrides (matches the old signature).
    defaults.update(hooked_kwargs)
    # We manage device ourselves — build on CPU, move once at the end.
    defaults.pop("device", None)

    model_name = preset["model_name"]
    ft_ckpt = preset.get("ft_checkpoint")

    if ft_ckpt and os.path.isdir(ft_ckpt):
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        print(f"Loading FT checkpoint from {ft_ckpt} into {model_name} architecture...")

        hf_model = AutoModelForCausalLM.from_pretrained(
            ft_ckpt,
            torch_dtype=_torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # Move HF to GPU before conversion. Keeps CPU RAM free for the
        # later TL allocation. Without fold_ln (which we skip — see below),
        # there's no GPU-VRAM blowup from putting hf_model on the device.
        if str(device) != "cpu" and _torch.cuda.is_available():
            hf_model = hf_model.to(device)

        # Monkey-patch AutoConfig/AutoTokenizer so any TL-internal lookup of
        # ``model_name`` resolves to the FT checkpoint (config / tokenizer).
        orig_config_from_pretrained = AutoConfig.from_pretrained
        orig_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

        def mock_config_from_pretrained(p, *a, **kw):
            if p == model_name or p == f"google/{model_name}":
                return orig_config_from_pretrained(ft_ckpt, *a, **kw)
            return orig_config_from_pretrained(p, *a, **kw)

        def mock_tokenizer_from_pretrained(p, *a, **kw):
            if p == model_name or p == f"google/{model_name}":
                return orig_tokenizer_from_pretrained(ft_ckpt, *a, **kw)
            return orig_tokenizer_from_pretrained(p, *a, **kw)

        AutoConfig.from_pretrained = mock_config_from_pretrained
        AutoTokenizer.from_pretrained = mock_tokenizer_from_pretrained

        try:
            # Step 1: build a TL cfg from the HF config (tiny — no weights).
            # NOTE: fold_ln is disabled to keep weights in bf16 throughout.
            # TL's load_and_process_state_dict upcasts weights to fp32
            # internally for fold_ln numerical stability, which doubles the
            # state_dict footprint on 2B models and OOMs the pod. TL itself
            # prints "With reduced precision, it is advised to use
            # from_pretrained_no_processing" — we honor that here.
            cfg = get_pretrained_model_config(
                model_name,
                hf_cfg=hf_model.config,
                fold_ln=False,
                device="cpu",
                dtype=defaults["dtype"],
            )
            # Step 2: extract + convert HF weights into TL-format state_dict.
            # hf_model is on GPU, so the conversion (rearranges, transposes)
            # runs on GPU and the result tensors live in VRAM, NOT in CPU RAM.
            state_dict = get_pretrained_state_dict(
                model_name,
                cfg,
                hf_model=hf_model,
                dtype=defaults["dtype"],
            )
            # Step 3: pull state_dict to CPU one tensor at a time, freeing
            # each GPU tensor as we go. ``pop`` releases the old dict's
            # reference before the next iteration so VRAM drains as we copy.
            cpu_state_dict = {}
            for k in list(state_dict.keys()):
                cpu_state_dict[k] = state_dict.pop(k).detach().to("cpu")
            state_dict = cpu_state_dict
            del cpu_state_dict

            # Step 4: free the HF model — VRAM and any CPU residue.
            del hf_model
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            # Step 5: build empty TL model on CPU and load the converted
            # weights. Plain load_state_dict (no processing) keeps everything
            # in bf16. strict=False because TL adds a few runtime buffers
            # (IGNORE, mask, etc.) that aren't in the state_dict.
            model = HookedTransformer(cfg, move_to_device=False)
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()

            if defaults["fold_ln"] or defaults["center_writing_weights"] or defaults["center_unembed"]:
                print(
                    "NOTE: load_model skips fold_ln / centering on FT checkpoints "
                    "(bf16-safe, memory-safe). Callers that decompose attention "
                    "per-head should still work: apply_ln_to_stack uses the cached "
                    "LN stats and produces equivalent logit contributions."
                )
        finally:
            AutoConfig.from_pretrained = orig_config_from_pretrained
            AutoTokenizer.from_pretrained = orig_tokenizer_from_pretrained

        # Step 6: move TL model to GPU and drop any lingering cache.
        model = model.to(device)
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
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
        model = HookedTransformer.from_pretrained(model_name, device=device, **defaults)

    # Inference-only: disable autograd to save activation memory during run_with_cache.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
