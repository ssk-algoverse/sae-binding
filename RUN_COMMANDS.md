# Gemma Run Commands

All commands are run from the **project root**. Prefix with `.venv/bin/python -u` to use the venv and get unbuffered output.

## Current State

| Item | Gemma-2-2b | Gemma-3-1b |
|---|---|---|
| FT checkpoint | `gemma2_ft_toy/checkpoint-900` ✓ | `gemma3_1b_ft_toy/checkpoint-900` ✓ |
| Sanity check | not yet run | not yet run |
| `per_head_logit_diffs.pt` | exists (may be stale) | not yet run |
| `target_layer / target_head` | L22 / H4 (already in config) | None — need pp first |
| exp4 | not run | not run |
| exp5 | not run | N/A (gemma-2-2b only) |
| exp6 | not run | not run |
| exp7 | not run | not run |

---

## Memory / OOM Notes

Pod memory limit is ~46.5 GiB. During model load, `load_model()` in `experiments/_gemma_config.py` briefly holds the HF checkpoint **and** the HookedTransformer in memory at the same time. Mitigations already applied:

- HookedTransformer built in `bfloat16` (not fp32) — halves the in-memory copy
- HF model loaded with `low_cpu_mem_usage=True` — avoids a random-init CPU allocation
- HF model moved to GPU immediately after load — frees CPU RAM before HookedTransformer builds

Before each run, clear lingering processes and cache to reclaim RAM:

```bash
pkill -f python ; rm -rf wandb/ ; sync
```

Run the two presets **sequentially**, not in parallel, during Phases 1–2.

---

## Phase 1 — Sanity Checks (run sequentially, ~5 min each)

Confirm each FT model achieves >90% retrieval accuracy before proceeding.

```bash
# Gemma-2-2b
GEMMA_PRESET=gemma-2-2b .venv/bin/python -u gemma/gemma_toy_eval.py

# Gemma-3-1b
GEMMA_PRESET=gemma-3-1b-pt .venv/bin/python -u gemma/gemma_toy_eval.py
```

If either prints `WARNING: accuracy ... < 0.90`, stop — the FT checkpoint is bad and downstream results will be meaningless.

---

## Phase 2 — Path Patching (run sequentially, ~1–2h GPU each)

Identifies the circuit head per model. Run one at a time (both are GPU-heavy).

```bash
# Gemma-2-2b — saves gemma/per_head_logit_diffs.pt
GEMMA_PRESET=gemma-2-2b .venv/bin/python -u gemma/pp_toy_dataset.py

# Gemma-3-1b — saves gemma/per_head_logit_diffs_gemma-3-1b-pt.pt
GEMMA_PRESET=gemma-3-1b-pt .venv/bin/python -u gemma/pp_toy_dataset.py
```

Each script prints `Top Head: L<N>H<M>` at the end. Note those values.

---

## Phase 3 — Fill in Config (manual, after Phase 2)

Edit `experiments/_gemma_config.py`. Gemma-2-2b is already filled in. For gemma-3-1b-pt:

```python
"gemma-3-1b-pt": {
    ...
    "target_layer": <L*>,           # from pp_toy_dataset.py output
    "target_head": <H*>,            # from pp_toy_dataset.py output
    "random_head_layer_range": (<L*-4>, <L*+4>),
    ...
},
```

---

## Phase 4 — Experiments

### Parallel opportunities

Once Phase 2 and 3 are done, experiments across the two models are independent. The A40 (44 GB) can hold both models simultaneously, so terminal-level parallelism is safe.

Within a single model, exp4 / exp6 / exp7 are also independent of each other.

> **Note:** exp5 is gemma-2-2b only — it just renders `gemma/per_head_logit_diffs.pt` and takes <1 min.

### Gemma-2-2b experiments

exp4 can start as soon as Phase 1 passes (no target_layer needed). exp5 needs Phase 2 to complete first.

```bash
# exp4 — linear probes (~30–40 min)
GEMMA_PRESET=gemma-2-2b .venv/bin/python -u experiments/exp4_gemma_linear_probes.py

# exp5 — causal patching plot (<1 min, needs Phase 2 first)
GEMMA_PRESET=gemma-2-2b .venv/bin/python -u experiments/exp5_gemma_causal_patching_plot.py

# exp6 — Q-K matching (~5–15 min)
GEMMA_PRESET=gemma-2-2b .venv/bin/python -u experiments/exp6_gemma_qk_matching.py

# exp7 — SAE recovery (~45–60 min)
GEMMA_PRESET=gemma-2-2b .venv/bin/python -u experiments/exp7_gemma_sae_recovery.py
```

### Gemma-3-1b experiments

All require Phase 2 + 3 to complete first (target_layer/head needed for exp6/exp7; exp4 only needs Phase 1).

```bash
# exp4 — linear probes (~30–40 min)
GEMMA_PRESET=gemma-3-1b-pt .venv/bin/python -u experiments/exp4_gemma_linear_probes.py

# exp6 — Q-K matching (~5–15 min)
GEMMA_PRESET=gemma-3-1b-pt .venv/bin/python -u experiments/exp6_gemma_qk_matching.py

# exp7 — SAE recovery (~30–60 min)
GEMMA_PRESET=gemma-3-1b-pt .venv/bin/python -u experiments/exp7_gemma_sae_recovery.py
```

### Suggested parallel terminal layout (after Phase 3)

```
Terminal A                                    Terminal B
──────────────────────────────────────────    ──────────────────────────────────────────
GEMMA_PRESET=gemma-2-2b   exp4               GEMMA_PRESET=gemma-3-1b-pt   exp4
GEMMA_PRESET=gemma-2-2b   exp5  (fast)
GEMMA_PRESET=gemma-2-2b   exp6               GEMMA_PRESET=gemma-3-1b-pt   exp6
GEMMA_PRESET=gemma-2-2b   exp7               GEMMA_PRESET=gemma-3-1b-pt   exp7
```

Run each row sequentially within a terminal; the two terminals run concurrently.

---

## Output Files

| Script | Output |
|---|---|
| `gemma_toy_eval.py` | stdout accuracy |
| `pp_toy_dataset.py` | `gemma/per_head_logit_diffs.pt` (2-2b) or `gemma/per_head_logit_diffs_gemma-3-1b-pt.pt` (3-1b) |
| exp4 | `experiments/results/gemma/<preset>_probe_accuracies_holdout.png` |
| exp5 | `experiments/results/gemma/gemma_causal_patching.png` |
| exp6 | `experiments/results/gemma/<preset>_qk_matching.png` |
| exp7 | `experiments/results/gemma/<preset>_sae_recovery.png` |
