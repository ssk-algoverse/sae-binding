# Reproducing the Paper Experiments

How to set up and run every experiment in the paper. For the actual numerical results and figures from the runs that produced the paper, see [`results.md`](results.md).

All scripts are run from the **project root** (paths like `gemma/...` are relative to the root).

---

## Setup

```bash
pip install torch transformer_lens sae_lens scikit-learn datasets seaborn matplotlib tqdm huggingface_hub
```

### Disk / bandwidth budget

| Asset | Size | Source |
|---|---|---|
| Gemma-2-2b weights | ~5 GB | HuggingFace, auto-downloaded by `transformer_lens` |
| Gemma-2-2b FT checkpoint | ~5 GB | Run `gemma/train_gemma.py` locally |
| Gemma-3-1b weights + FT checkpoint | ~5 GB | HF + `gemma/train_gemma.py` |
| Gemma-Scope SAE width_16k | ~280 MB each | `sae_lens` (auto-downloaded) |

### FT checkpoint loading

All Gemma scripts go through `experiments/_gemma_config.py::load_model`, which auto-loads the FT checkpoint declared in the preset (`ft_checkpoint`) if present, else falls back to base with a warning. Circuit heads (L22H4 for gemma-2-2b, L22H3 for gemma-3-1b-pt) were identified on the FT model, so the FT checkpoint is required for the Gemma probe / Q-K / SAE results to match the paper.

---

## Operational notes

### Memory / OOM (46.5 GiB pod)

Mitigations baked into `load_model()`:

- **HF on GPU during conversion** — `hf_model.to(device)` immediately after load; HF→TL state_dict conversion runs on GPU and CPU RAM is free while TL is built later. The intermediate state_dict is drained to CPU one tensor at a time.
- **Staged load** — extract TL state_dict, `del hf_model`, load into TL, `del state_dict`. Only one full copy of the weights is ever live on CPU at a time.
- **No fold_ln / centering on the FT path** — TL's processing upcasts weights to fp32 internally (TL itself recommends `from_pretrained_no_processing` for reduced precision); the fp32 peak was tipping the pod over. Per-head logit-diff analysis still works — `apply_ln_to_stack` uses cached LN stats at runtime.
- HookedTransformer built in `bfloat16`; HF model loaded with `low_cpu_mem_usage=True`; `requires_grad=False` on all params.

`gemma_toy_eval.py` bypasses HookedTransformer entirely — a plain HF forward pass needs none of the above.

Before each gemma run, clear lingering processes and cache:

```bash
pkill -f python ; rm -rf wandb/ ; sync
```

### Parallel runs

The two presets are independent at the experiment level. The A40 (44 GB) can hold both Gemma-2-2b and Gemma-3-1b simultaneously, so two terminals can run different presets in parallel — but **not the same preset twice** (same model loaded twice would OOM CPU).

Suggested layout once Phase 1 (FT eval) and Phase 2 (path-patching) are done:

```
Terminal A (gemma-2-2b)        Terminal B (gemma-3-1b-pt)
─────────────────────────────  ─────────────────────────────
exp4                            exp4
exp5  (fast)
exp6                            exp6
exp7                            exp7
```

Run each row sequentially within a terminal; the two terminals run concurrently.

### Output files

| Script | Output |
|---|---|
| `gemma_toy_eval.py` | stdout accuracy |
| `pp_toy_dataset.py` | `gemma/per_head_logit_diffs.pt` (2-2b) or `gemma/per_head_logit_diffs_<preset>.pt` |
| exp1 | `experiments/results/{cosine_distributions,mean_pairwise_cosine,pca_comparison}.png` |
| exp1b | `experiments/results/toy_sae_recovery.png` |
| exp2 | `experiments/results/{qk_eigenvalue_spectrum,ov_singular_values}.png` |
| exp3 | `experiments/results/{ablation_accuracy,ablation_logit_diffs}.png` |
| exp4 | `experiments/results/gemma/<preset>_probe_accuracies_{holdout,cv}.png`, `*_probe_results.npz` |
| exp5 | `experiments/results/gemma/<preset>_causal_patching.png` |
| exp6 | `experiments/results/gemma/<preset>_qk_matching.png` |
| exp7 | `experiments/results/gemma/<preset>_sae_recovery.png` |

---

## Experiment Suite

### Priority 0 — Toy Model

The following scripts run against the 2-layer 2-head toy transformer (`sebastianhoenig/2L2H_Final`) and prove the core circuit claims.

#### Exp 1 · Combinatorial Geometry Analysis

**Script:** `python experiments/exp1_geometry_analysis.py`

**Claim:** The composed-address representation forms ~1,000 dense, non-orthogonal clusters that L1-sparse methods cannot extract. Measured by intra-class vs inter-class cosine similarity at `blocks.0.hook_resid_post`.

Includes `set_seed(0)` and bootstrap 95% CIs on the off-diagonal cosine mean and on the intra/inter/gap triple.

#### Exp 1b · Toy SAE Recovery

**Script:** `python experiments/exp1b_toy_sae_recovery.py`

**Claim:** An L1 SAE trained directly on toy `blocks.0.hook_resid_post` fails to recover (E1,T) through its reconstruction, even when the sparse latents themselves still carry some signal. The empirical counterpart to the geometric argument in Exp 1.

Probes: raw X / X_recon (SAE reconstruction) / z (sparse latents) / PCA rank-matched / RandProj rank-matched.

#### Exp 2 · Weight Matrix Decomposition

**Script:** `python experiments/exp2_weight_analysis.py`

**Claim:** The L1H0 $W_Q W_K^T$ mechanism is hardcoded to the composed-address subspace — measured via principal angles between the QK eigenspace and the empirical class-mean subspace at `blocks.0.hook_resid_post`.

Reports `REAL` vs `NULL CI` against a 200×-shuffle label-permutation null. The Gaussian-QR random baseline is retained as a footnote.

#### Exp 3 · Causal Necessity Ablation

**Script:** `python experiments/exp3_ablation.py`

**Claim:** L0H0 (Address head) and L0H1 (Payload head) are individually necessary — zeroing either collapses retrieval accuracy. The two heads play distinct roles.

- Per-condition confusion matrix: `correct` / `wrong_E2_in_prompt` / `wrong_E2_not_in_prompt`.
- `swap_L0H0_L0H1` condition swaps the two heads' z-values at SEP positions — symmetric heads → no-op swap.

---

### Priority 2 — Gemma-2-2b / Gemma-3-1b-pt Replication

> **⚠️ FT-checkpoint alignment:** L22H4 (gemma-2-2b) and L22H3 (gemma-3-1b-pt) were identified on the *finetuned* models. `load_model` auto-loads the FT checkpoint if present; if missing, it warns and falls back to base — results may not reproduce the circuit claim.

#### Phase 1 — Sanity-check the FT model

```bash
GEMMA_PRESET=gemma-2-2b     .venv/bin/python -u gemma/gemma_toy_eval.py
GEMMA_PRESET=gemma-3-1b-pt  .venv/bin/python -u gemma/gemma_toy_eval.py
```

Each prints retrieval accuracy on the held-out eval split (indices 8000+). Stop if either prints `WARNING: accuracy < 0.90` — the FT checkpoint is bad and downstream results will be meaningless.

#### Phase 2 — Identify the circuit head

```bash
GEMMA_PRESET=gemma-2-2b     .venv/bin/python -u gemma/pp_toy_dataset.py
GEMMA_PRESET=gemma-3-1b-pt  .venv/bin/python -u gemma/pp_toy_dataset.py
```

Each script runs path-patching on the first 50% of `gemma_pp_dataset.jsonl` (seed=0) and saves per-head logit diffs to `gemma/per_head_logit_diffs.pt` (gemma-2-2b) or `gemma/per_head_logit_diffs_<preset>.pt`. Each prints `Top Head: L<N>H<M>` at the end. Fill those into `experiments/_gemma_config.py` as `target_layer` / `target_head`.

#### Phase 3 — Fill in the preset

Edit `experiments/_gemma_config.py`. Required fields per preset: `model_name`, `ft_checkpoint`, `target_layer`, `target_head`, `random_head_layer_range`, `sae_configs`. For exp7, also confirm `sae_layer` (defaults to `target_layer`; override when the SAE release lacks weights at `target_layer` or when `(E1,T)` decodability peaks at a different layer).

#### Phase 4 — Run the experiments

```bash
GEMMA_PRESET=<preset> .venv/bin/python -u experiments/exp4_gemma_linear_probes.py
GEMMA_PRESET=<preset> .venv/bin/python -u experiments/exp5_gemma_causal_patching_plot.py
GEMMA_PRESET=<preset> .venv/bin/python -u experiments/exp6_gemma_qk_matching.py
GEMMA_PRESET=<preset> .venv/bin/python -u experiments/exp7_gemma_sae_recovery.py
```

| Exp | Description | Time |
|---|---|---|
| exp4 | Linear probes across all layers (held-out 30%, 5-seed bootstrap, mean ± std) | ~30–40 min |
| exp5 | Plot the per-head causal patching tensor (preset-aware loader) | <1 min |
| exp6 | Q-K matching at `target_head` with 4-way null (correct vs distractor / random-position / random-head). Held-out 50%; GQA via `kv_head_for(q_head, group_size)` | ~5–15 min |
| exp7 | SAE recovery sweep: 5 sources (raw X, X_recon, z, PCA rank-matched, RandProj rank-matched) × N configs. 5-seed bootstrap | ~30–60 min |

> **exp7 solver note:** The probe function detects sparsity: if >50% of entries are zero (SAE z latents), it drops always-zero columns and uses `solver='sparse_cg'`; otherwise `solver='auto'` (Cholesky). For gemma-3-1b-pt this keeps the full sweep within ~30–60 min. For gemma-2-2b, the 1000-class E1,T z-probe with `sparse_cg` still stalls (unknown condition-number issue). A faster alternative: `LogisticRegression(solver='saga', max_iter=200)` for z probes.

---

## Open items

*(none — all experiments complete; see `results.md` for final numbers)*

---

## Gemma 3 Migration

The Gemma scripts are parameterised by a single **`GEMMA_PRESET`** env var. Switching models is a one-variable flip.

### Current presets

| `GEMMA_PRESET` | model | FT checkpoint | SAE family | status |
|---|---|---|---|---|
| `gemma-2-2b` (default) | gemma-2-2b | `gemma2_ft_toy/checkpoint-900` | Gemma-Scope 1 (L0~72 + L0~21) | ready |
| `gemma-3-1b-pt` | gemma-3-1b-pt | `gemma3_1b_ft_toy/checkpoint-900` | Gemma-Scope 2 width_16k (L7/13/17/22) | ready |
| `gemma-3-4b-pt` | gemma-3-4b-pt | none | Gemma-Scope 2 canonical *(verify id)* | needs FT + path-patching |

### What's automated

- **FT vs base loading** — `load_model` checks `ft_checkpoint` path existence; warns + falls back to base if missing.
- **`N_LAYERS`** — read from `model.cfg.n_layers` after load.
- **GQA mapping** — `kv_head_for(q_head, group_size)` derived from `model.cfg.n_heads / n_key_value_heads`.
- **SAE ids** — `{layer}` substituted from `target_layer` (or `sae_layer`) at runtime.

### Adding a new preset

1. **Fine-tune.** `train_gemma.py` is parameterised by env vars:
   ```bash
   HF_TOKEN="..." WANDB_API_KEY="..." \
   TL_MODEL_NAME="gemma-3-1b-pt" \
   HF_MODEL_ID="google/gemma-3-1b-pt" \
   OUTPUT_DIR="./gemma3_1b_ft_toy" \
   BATCH_SIZE="8" GRAD_ACCUM="1" \
   .venv/bin/python -u gemma/train_gemma.py
   ```
   For larger models (e.g. gemma-2-2b), use `BATCH_SIZE=2 GRAD_ACCUM=4` to avoid OOM.

2. **Sanity-check the FT model** — `GEMMA_PRESET=<preset> .venv/bin/python -u gemma/gemma_toy_eval.py`. Confirm >90% retrieval accuracy.

3. **Identify the circuit head** — `GEMMA_PRESET=<preset> .venv/bin/python -u gemma/pp_toy_dataset.py`. Note the `Top Head` printed at end.

4. **Fill in the preset** in `experiments/_gemma_config.py` with `target_layer`, `target_head`, `random_head_layer_range`, `sae_configs`. Verify SAE release IDs against:
   ```bash
   .venv/bin/python -c "import sae_lens, pathlib, yaml; \
     d = yaml.safe_load(open(pathlib.Path(sae_lens.__file__).parent/'pretrained_saes.yaml')); \
     print([s['id'] for s in d['<release-name>']])"
   ```
   Verify `comma_id` / `period_id` tokenise to single tokens: `model.to_tokens(',')` returns a single token.

5. **(Optional) Upload FT to HF** — `.venv/bin/hf upload <user>/<repo> ./<OUTPUT_DIR> .` after `.venv/bin/hf auth login`.

### Cost estimate per new preset

| Task | Time |
|---|---|
| FT training (gemma-3-1b) | ~1–3 h GPU |
| Path-patching | ~1–2 h GPU |
| exp4 + exp6 + exp7 | ~1 h GPU total |

### Why bother with Gemma 3

- Defuses "why not the most recent model?" reviewer question.
- Matryoshka SAEs (Gemma Scope 2) may partially recover (E1,T); if so, the framing shifts from "SAEs fail" to "L1 SAEs at this width fail; Matryoshka partially recovers" — a stronger and more nuanced claim.
- Gemma Scope 2 transcoders give a per-layer compute graph that fits naturally into the Address-vs-Payload story.
