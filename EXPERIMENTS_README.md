# Reproducing the Paper Experiments

This document is the single source of truth for running and re-running all empirical experiments. It covers the original experiment descriptions, what changed in the hardening pass, what still needs to be run, and how to migrate to Gemma 3.

All scripts are run from the **project root** (paths like `gemma/...` are relative to the root).

---

## Setup

```bash
pip install torch transformer_lens sae_lens scikit-learn datasets seaborn matplotlib tqdm huggingface_hub
```

**Disk / bandwidth budget for the full Gemma run:**

| Asset | Size | Source |
|---|---|---|
| Gemma-2-2b weights | ~5 GB | HuggingFace, auto-downloaded by `transformer_lens` |
| Gemma-2-2b FT checkpoint | ~5 GB | Run `gemma/gemma_toy_ft.ipynb` locally |
| Gemma-Scope SAE width_16k canonical | ~280 MB | `sae_lens` (auto-downloaded) |
| Gemma-Scope SAE width_16k l0_22 | ~280 MB | `sae_lens` (auto-downloaded) |

> **Model loading:** all Gemma scripts (`exp4`–`exp7`) go through `experiments/_gemma_config.py::load_model`, which automatically loads the FT checkpoint (`gemma/gemma2_ft_toy/checkpoint-900/`) if the directory exists, or falls back to base Gemma-2-2b with a warning. The circuit head (L22H4) was identified on the FT model by `gemma/pp_toy_dataset.ipynb`, so the FT checkpoint is required for results to be consistent with the circuit claim.

---

## Experiment Suite

### Priority 0 — Toy Model

The following scripts run against the 2-layer 2-head toy transformer (`sebastianhoenig/2L2H_Final`) and prove the core circuit claims.

---

#### Exp 1 · Combinatorial Geometry Analysis

**Script:** `python experiments/exp1_geometry_analysis.py`

**Claim:** The composed-address representation forms ~1,000 dense, non-orthogonal clusters that L1-sparse methods cannot extract. Measured by intra-class vs inter-class cosine similarity at `blocks.0.hook_resid_post`.

**What changed:** added `set_seed(0)`, bootstrap 95% CIs on the off-diagonal cosine mean and on the intra/inter/gap triple.

**Expected output lines:**
```
intra μ=0.9043 [0.9039, 0.9047]  inter μ=0.2891 [0.2885, 0.2898]  gap=0.6152 [0.6144, 0.6160]
```
Output PNG: `experiments/results/geometry_analysis.png`

---

#### Exp 1b · Toy SAE Recovery *(new)*

**Script:** `python experiments/exp1b_toy_sae_recovery.py`

**Claim:** An L1 SAE trained directly on toy `blocks.0.hook_resid_post` fails to recover (E1,T) through its reconstruction, even when the sparse latents themselves still carry some signal. This is the empirical counterpart to the geometric argument in Exp 1.

Probes: raw X / X_recon (SAE reconstruction) / z (sparse latents) / PCA rank-matched / RandProj rank-matched.

Output PNG: `experiments/results/toy_sae_recovery.png`

---

#### Exp 2 · Weight Matrix Decomposition

**Script:** `python experiments/exp2_weight_analysis.py`

**Claim:** The L1H0 $W_Q W_K^T$ mechanism is hardcoded to the composed-address subspace — measured via principal angles between the QK eigenspace and the empirical class-mean subspace at `blocks.0.hook_resid_post`.

**What changed:** added a label-permutation null (`permutation_null`): the same alignment metric computed 200× over shuffled (E1,T) class assignments. Reports `REAL` vs `NULL CI`. The Gaussian-QR random baseline is retained but demoted — the permutation null is the headline comparison.

Expected:
```
[REAL]   σ1=0.985  σ̄=0.63  align=0.54
[NULL]   σ1=0.24 ± ...  σ̄=...  align=0.08 ± ...
```

---

#### Exp 3 · Causal Necessity Ablation

**Script:** `python experiments/exp3_ablation.py`

**Claim:** L0H0 (Address head) and L0H1 (Payload head) are individually necessary — zeroing either collapses retrieval accuracy. The two heads play distinct roles.

**What changed:**
- Per-condition confusion matrix: `correct` / `wrong_E2_in_prompt` (mis-routed to wrong fact) / `wrong_E2_not_in_prompt` (payload garbled).
- New `swap_L0H0_L0H1` condition swaps the two heads' z-values at SEP positions — a strong role-specialisation test (symmetric heads → no-op swap).

Expected interpretation: ablating H0 inflates `wrong_E2_in_prompt`; ablating H1 inflates `wrong_E2_not_in_prompt`; swap ≠ no-op.

---

### Priority 2 — Gemma-2-2b Replication

> **⚠️ FT-checkpoint alignment:** L22H4 was identified on a *finetuned* Gemma-2-2b (`gemma/gemma2_ft_toy/checkpoint-900/`, output of `gemma/gemma_toy_ft.ipynb`), not on base weights. `load_model` in `_gemma_config.py` auto-loads the FT checkpoint if present. If missing, it warns and falls back to base — results may not reproduce the circuit claim. Regenerate with `gemma/gemma_toy_ft.ipynb` (writes to `./gemma2_ft_toy/`; ensure it lands at `gemma/gemma2_ft_toy/` relative to the project root).

---

#### Exp 4 · Linear Probes Across Layers

**Script:** `python experiments/exp4_gemma_linear_probes.py`

**Claim:** The FT Gemma-2-2b residual stream factorises E2 (payload) and (E1,T) (composed address) across layers, with peak (E1,T) accuracy at mid-to-late layers.

**What changed:** prompt-level held-out split (30% held out, not seen during probe training), 5-seed bootstrap over different splits, `mean ± std` band per layer in the plot. `MAX_PROMPTS=300` (was 150). Model loaded via `load_model(preset, device)` — uses FT checkpoint.

**Run:**
```bash
python experiments/exp4_gemma_linear_probes.py
```

**Outputs** (in `experiments/results/gemma/`):
- `gemma-2-2b_probe_accuracies_holdout.png` — **reporting metric** (use this in the paper)
- `gemma-2-2b_probe_accuracies_cv.png` — train-CV selection metric
- `gemma-2-2b_probe_results.npz` — raw arrays `<target>_{cv,holdout}` shape `[5, n_layers]`

Compute: ~20–40 min on M-series MPS, ~5 min on A100/L4.

---

#### Exp 5 · Causal Patching Heatmap *(plot-only)*

**Script:** `python experiments/exp5_gemma_causal_patching_plot.py`

**Claim:** L22H4 is the primary routing head in Gemma-2-2b (on the FT model + toy dataset).

This script only renders `gemma/per_head_logit_diffs.pt`. That tensor is produced by **`gemma/pp_toy_dataset.ipynb`** — the source of truth for which model, dataset, and metric were used. The old root-level `PathPatchingGemma.ipynb` (base Gemma + prakash boxes dataset) has been removed; it was not the source of the L22H4 claim.

**To regenerate `per_head_logit_diffs.pt`** (required before calling the plotter after a model change):
1. Open `gemma/pp_toy_dataset.ipynb`.
2. Apply the same 50/50 prompt-level split (`seed=0`, first half = selection, second half = held-out) as exp6.
3. Run path-patching on the **selection half only**. Re-export to `gemma/per_head_logit_diffs.pt` (the notebook saves near the bottom — copy to `gemma/` if needed).
4. Re-run `python experiments/exp5_gemma_causal_patching_plot.py`.

**Keep the notebook and `exp5_gemma_causal_patching_plot.py` in lock-step.** Any change to layer range, metric, or prompt subset in the notebook must land alongside an update to this plotter.

---

#### Exp 6 · Q-K Matching Analysis

**Script:** `python experiments/exp6_gemma_qk_matching.py`

**Claim:** L22H4's Q-K dot products are discriminatively higher at the correct comma (following the target fact) than at distractors — ruling out "any head does this" and "head scores all commas equally".

**What changed:**
- Evaluates only on the held-out 50% of `gemma/gemma_pp_dataset.jsonl` (seed=0 split; first 50% is reserved for path-patching head selection in exp5/notebook).
- Three null distributions: distractor commas (same head, other commas), random-position (same head, random comma), random-head (random L,H excluding L22H4, correct comma).
- GQA handled correctly: `kv_head_for(q_head, group_size)` using `group_size` from `model.cfg`.
- Model loaded via `load_model(preset, device)`.

**Run:**
```bash
python experiments/exp6_gemma_qk_matching.py
```

**Output:** `experiments/results/gemma/gemma-2-2b_qk_matching.png` — 4-way KDE. The discriminative claim lives in the gap between green (correct) and gray/blue (nulls).

Compute: ~5–15 min on MPS.

---

#### Exp 7 · Gemma-Scope SAE Recovery

**Script:** `python experiments/exp7_gemma_sae_recovery.py`

**Claim:** Gemma-Scope SAEs fail to recover (E1,T) through reconstruction — the "dark matter" failure mode. The probe-on-z vs probe-on-X_recon split diagnoses whether the latents encode the feature at all vs whether reconstruction destroys it.

**What changed:**
- Sweeps over `sae_configs` list (2 configs by default: canonical L0~72 and sparser average_l0_22). Add configs in `experiments/_gemma_config.py`.
- 5 probed sources per config: raw X / X_recon / z (sparse latents) / PCA rank-matched / RandProj rank-matched.
- 5-seed bootstrap on all probe accuracies.
- Subplot per config; raw dashed reference lines.
- Model loaded via `load_model(preset, device)`.

**Run:**
```bash
python experiments/exp7_gemma_sae_recovery.py
```

**Output:** `experiments/results/gemma/gemma-2-2b_sae_recovery.png`

Compute: ~30–60 min on MPS for 2 configs. Add ~15 min per extra config.

---

## Re-run TODO

Checkboxes represent the state **before** re-running with the hardened scripts. Tick them off as you go.

### Toy (fast — minutes on CPU/MPS)

- [x] **exp1** — re-run to get bootstrap CIs on geometry claims
- [x] **exp1b** — new file, must run; proves SAE failure empirically on toy
- [x] **exp2** — re-run to get permutation-null comparison
- [x] **exp3** — re-run to get confusion-matrix breakdown + head-swap result

### Gemma (slow — GPU recommended)

Prerequisites before any Gemma experiment:
- [ ] Run `gemma/gemma_toy_ft.ipynb` end-to-end → checkpoint at `gemma/gemma2_ft_toy/checkpoint-900/`

Then:
- [ ] **exp4** — re-run for held-out probe numbers; ~30–40 min
- [ ] **exp5 / notebook** — re-run `gemma/pp_toy_dataset.ipynb` with 50/50 held-out split; re-export `gemma/per_head_logit_diffs.pt`; re-run plotter
- [ ] **exp6** — re-run for held-out + 4-way null; ~5–15 min
- [ ] **exp7** — re-run for z-probe + rank-matched controls; ~45–60 min for 2 configs

### Paper updates (after re-runs)

Update `paper.tex` with:
- [ ] **exp1** — replace point estimates `0.904 / 0.289` with bootstrapped CIs
- [ ] **exp2** — report `σ1 / σ̄ / align` as `REAL vs PERMUTATION-NULL CI`; demote Gaussian baseline to footnote
- [ ] **exp3** — add confusion-matrix story and head-swap result
- [ ] **exp1b** — add paragraph on toy SAE empirical probe (raw / recon / z / rank-matched)
- [ ] **exp4** — report **held-out** numbers (with std bands), not CV numbers
- [ ] **exp5** — note that path patching ran on FT model (not base)
- [ ] **exp6** — report `correct vs random-head` and `correct vs random-position` gaps (not just distractor)
- [ ] **exp7** — report full 5-source × 2-config table; discuss `z >> X_recon` interpretation

---

## Suggested Run Order (budget-constrained)

1. `exp1`, `exp2`, `exp3`, `exp1b` — toy, ~10 min total. Biggest evidence delta per hour.
2. Regenerate FT checkpoint (`gemma/gemma_toy_ft.ipynb`).
3. `exp6` — ~15 min. Held-out + null controls is the highest-leverage single Gemma change.
4. `exp4` — ~30 min. Proper held-out probe numbers.
5. `exp7` — ~45 min. Most likely to change the "dark matter" headline framing.
6. **Optional:** re-run `gemma/pp_toy_dataset.ipynb` with the 50/50 split → `exp5` redraw. Closes the "head selected on same data it's evaluated on" critique.

---

## Gemma 3 Migration

The Gemma scripts are parameterised by a single **`GEMMA_PRESET`** env var. Switching models is a one-variable flip. Presets live in `experiments/_gemma_config.py`.

### Current presets

| `GEMMA_PRESET` | model | FT checkpoint | SAE family | status |
|---|---|---|---|---|
| `gemma-2-2b` (default) | gemma-2-2b | `gemma/gemma2_ft_toy/checkpoint-900` | Gemma-Scope 1 (L0~72 + L0~22) | **ready** |
| `gemma-3-1b-pt` | gemma-3-1b-pt | none | Gemma-Scope 2 canonical *(verify id)* | needs FT + path-patching |
| `gemma-3-4b-pt` | gemma-3-4b-pt | none | Gemma-Scope 2 canonical *(verify id)* | needs FT + path-patching |

### Run commands

```bash
# Default (gemma-2-2b, FT checkpoint auto-loaded if present)
python experiments/exp4_gemma_linear_probes.py
python experiments/exp6_gemma_qk_matching.py
python experiments/exp7_gemma_sae_recovery.py

# Gemma-3-1b (exp4 works today; exp6/exp7 error until target_layer/head filled in)
GEMMA_PRESET=gemma-3-1b-pt python experiments/exp4_gemma_linear_probes.py
GEMMA_PRESET=gemma-3-1b-pt python experiments/exp6_gemma_qk_matching.py   # fails: needs target_layer/head
GEMMA_PRESET=gemma-3-1b-pt python experiments/exp7_gemma_sae_recovery.py  # fails: needs target_layer
```

Output files are preset-namespaced so multiple runs coexist in `experiments/results/gemma/`.

### What's automated

- **FT vs base loading** — `load_model` checks `ft_checkpoint` path existence; warns + falls back to base if missing.
- **`N_LAYERS`** — read from `model.cfg.n_layers` after load.
- **GQA mapping** — `kv_head_for(q_head, group_size)` derived from `model.cfg.n_heads / n_key_value_heads`.
- **SAE ids** — `{layer}` substituted from `target_layer` at runtime.

### Checklist for a new Gemma 3 preset

- [ ] Run `gemma/gemma_toy_ft.ipynb` on the new model — update `output_dir` and the preset's `ft_checkpoint` field
- [ ] Run `gemma/pp_toy_dataset.ipynb` on the FT checkpoint — identify `(L*, H*)`, copy/save `per_head_logit_diffs.pt` to `gemma/`
- [ ] Fill in `target_layer`, `target_head`, `random_head_layer_range` in `experiments/_gemma_config.py`
- [ ] Verify Gemma-Scope 2 `sae_lens` release id against Neuronpedia or `sae_lens.toolkit.pretrained_saes_directory`
- [ ] Verify `comma_id` / `period_id` still tokenise to single tokens in the Gemma 3 tokenizer (silent failure if not)

### Cost estimate

| Task | Time |
|---|---|
| FT training on Gemma-3-1b | ~1–3 h GPU |
| Path-patching notebook (Gemma-3-1b) | ~1–2 h GPU |
| exp4 + exp6 + exp7 on Gemma-3-1b | ~1 h GPU total |

### Why do this

- Defuses "why not the most recent model?" reviewer question in one sentence.
- Matryoshka SAEs (Gemma Scope 2) may partially recover (E1,T); if so, the framing shifts from "SAEs fail" to "L1 SAEs at this width fail; Matryoshka partially recovers" — a stronger and more nuanced claim.
- Gemma Scope 2 transcoders give a per-layer compute graph that fits naturally into the Address-vs-Payload story.
