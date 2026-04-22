# Re-running the Hardened Experiments

This document describes what was changed in each experiment script, what to re-run, and how. Scripts marked **(re-run required)** have non-trivial code changes that change the headline numbers; scripts marked **(no re-run)** were either unchanged or only got a documentation/comment update.

---

## 0 · Environment

The repo doesn't ship a `requirements.txt`. The hardening did not introduce new dependencies. You need:

```bash
pip install torch transformer_lens sae_lens scikit-learn datasets seaborn matplotlib tqdm huggingface_hub
```

**Disk / bandwidth budget for the full Gemma re-run:**

| Asset | Size | Source |
|---|---|---|
| Gemma-2-2b weights | ~5 GB | HuggingFace, auto-downloaded by `transformer_lens` |
| Gemma-Scope SAE width_16k canonical | ~280 MB | `sae_lens` (one per (width, L0) config) |
| Gemma-Scope SAE width_16k l0_22 | ~280 MB | `sae_lens` |

If you add wider SAEs to `CONFIGS` in exp7 (e.g. width_65k), each adds ~1 GB.

Run scripts from the **project root** (paths like `gemma/...` are relative).

---

## 1 · Toy experiments (cheap — minutes on CPU/MPS)

### exp1 — Geometry (re-run required)
**Changed:** added `set_seed`, bootstrap 95 % CIs on the off-diagonal cosine mean and on intra/inter cosines + their gap.

```bash
python experiments/exp1_geometry_analysis.py
```

Look for new lines like:
```
intra μ=0.9043 [0.9039, 0.9047]  inter μ=0.2891 [0.2885, 0.2898]  gap=0.6152 [0.6144, 0.6160]
```

### exp1b — Toy SAE recovery (NEW — must run)
**New file.** Trains a small L1 SAE on `blocks.0.hook_resid_post`, then probes (E1,T) and E2 from raw / X_recon / z (sparse latents) / PCA-rank-matched / random-projection-rank-matched controls.

```bash
python experiments/exp1b_toy_sae_recovery.py
```

This is the *empirical* version of the "SAEs fail on the toy" claim that exp1 only inferred from geometry. Look at the `z (SAE latent)` column vs `X_recon (SAE)`. Output: `experiments/results/toy_sae_recovery.png`.

### exp2 — QK weight analysis (re-run required)
**Changed:** added a label-permutation null (`permutation_null` over the (E1,T) class assignments). The Gaussian-QR random subspace baseline is kept but explicitly labeled as a weak null (controls for dimensionality only, not for training-induced co-shaping of weights and activations).

```bash
python experiments/exp2_weight_analysis.py
```

The new output gives `REAL` vs `NULL (label permutation)` per head. The interesting comparison is **REAL alignment vs the permutation NULL CI**, not vs the Gaussian baseline.

### exp3 — Ablation (re-run required)
**Changed:** added a confusion-matrix breakdown per condition (`correct` / `wrong_E2_in_prompt` / `wrong_E2_not_in_prompt`) and a new `swap_L0H0_L0H1` condition that swaps the two heads' z values at SEP. Tests *role specialization*, not just necessity.

```bash
python experiments/exp3_ablation.py
```

The summary table at the end now includes the three category percentages per condition. If H0 is truly the address head and H1 is the payload head, ablating H0 should produce more `wrong_E2_in_prompt` (mis-routed to a wrong fact) and ablating H1 should produce more `wrong_E2_not_in_prompt` (payload garbled). The `swap` condition is a strong test — if the heads play *symmetric* roles, swapping them is a no-op.

---

## 2 · Gemma experiments (slower — laptop will sweat, cloud GPU recommended)

> **⚠️ FT-checkpoint alignment (now automatic):** the L22H4 circuit was identified by `gemma/pp_toy_dataset.ipynb` on a *finetuned* checkpoint (`gemma/gemma2_ft_toy/checkpoint-900/`, output of `gemma/gemma_toy_ft.ipynb`), not on base `gemma-2-2b`. Earlier versions of exp4/exp6/exp7 silently loaded base weights, so the probes / Q-K / SAE measurements were on a different model than the circuit claim. The shared `experiments/_gemma_config.py::load_model` helper now loads the FT checkpoint when present, so all four scripts (the notebook + exp4/6/7) use the same weights.
>
> If the FT checkpoint directory is missing, `load_model` prints a warning and falls back to base. To regenerate it, run `gemma/gemma_toy_ft.ipynb` end-to-end (writes to `./gemma2_ft_toy/checkpoint-900/`; move into `gemma/` so it's discoverable from project-root invocations).

### exp4 — Linear probes (re-run required)
**Changed:** held-out split at the **prompt level** (30 % held out), 5-seed bootstrap over different splits, plotting now shows `mean ± std` band per layer. Also bumped `MAX_PROMPTS=300` (from 150) so the holdout has enough class coverage.

```bash
python experiments/exp4_gemma_linear_probes.py
```

Outputs:
- `experiments/results/gemma/gemma_probe_accuracies_holdout.png` — the **reporting** metric (use this in the paper).
- `experiments/results/gemma/gemma_probe_accuracies_cv.png` — the train-CV selection metric.
- `experiments/results/gemma/gemma_probe_results.npz` — raw arrays `<target>_cv` and `<target>_holdout` of shape `[5, 26]`.

Compute estimate: ~20–40 min on M-series MPS, ~5 min on a single A100/L4. The bottleneck is the per-prompt forward pass with 26-layer caching; the probe training itself is fast.

### exp5 — Causal patching plot (NO re-run from script alone)
**Changed:** updated NOTICE comment to point at the correct source notebook (`gemma/pp_toy_dataset.ipynb`) and to flag that path patching ran on the FT checkpoint. The old root-level `PathPatchingGemma.ipynb` (base Gemma + prakash boxes dataset) has been **deleted** — it was a stale sibling, not the source of `gemma/per_head_logit_diffs.pt`.

> **⚠️ The actual head-selection logic lives in `gemma/pp_toy_dataset.ipynb`.**
> That notebook loads the FT checkpoint via `AutoModelForCausalLM.from_pretrained("./gemma2_ft_toy/checkpoint-900/")` and wraps it in HookedTransformer with the gemma-2-2b architecture template.
>
> If you want to address the "exp5 selected L22H4 on the same prompts everything else uses" critique:
> 1. Open `gemma/pp_toy_dataset.ipynb`.
> 2. Apply the same 50/50 prompt-level split that exp6 now uses on `gemma/gemma_pp_dataset.jsonl` (use `seed=0`, take the first half as the *selection* set, second half as held-out).
> 3. Run path-patching only on the **selection** half. Re-export `gemma/per_head_logit_diffs.pt` (the notebook saves to `e1/per_head_logit_diffs.pt` near line 1452 — copy/symlink into `gemma/` for the plotter to find it).
> 4. Re-run `python experiments/exp5_gemma_causal_patching_plot.py` to redraw the heatmap.
> 5. The held-out validation that L22H4 is special is what exp6 now does.
>
> **Keep the notebook and `exp5_gemma_causal_patching_plot.py` in lock-step.** Any change to layer range, prompt subset, or metric in the notebook must land in the same PR as the corresponding edits in the plotter.

### exp6 — Q-K matching (re-run required)
**Changed:** evaluates only on the held-out 50 % of `gemma_pp_dataset.jsonl` (seed=0 split, matches the split exp5/notebook should use). Adds two null distributions:
- **random-position null** — same head, but on a randomly chosen comma in the prompt
- **random-head null** — random `(L, H)` from layers 18–25 excluding L22H4, on the correct comma

```bash
python experiments/exp6_gemma_qk_matching.py
```

Output: `experiments/results/gemma/gemma_qk_matching.png` — now a 4-way KDE. The discriminative claim about L22H4 lives in the gap between the green (correct) and the gray/blue (null) distributions, not the original red (distractor) which was a weak null.

Compute estimate: ~5–15 min on MPS (one forward pass per prompt, K/Q caching for 8 layers).

### exp7 — SAE recovery (re-run required)
**Changed substantially:**
- Loops over `CONFIGS` (a list of SAE specs) so you can sweep widths/L0s.
- Adds **probe-on-z** condition (probe the sparse latents directly — the actual interpretability question).
- Adds **PCA rank-matched** and **random-projection rank-matched** controls so the SAE result can be compared against a same-rank lossy reconstruction.
- 5-seed bootstrap on every probe accuracy.
- Plot is now one subplot per config, with raw-X dashed reference lines.

Default `CONFIGS` runs two SAE configs (canonical L0~72 and average_l0_22 for sparser comparison). Edit the list at the top of the file to add `width_65k/canonical` if you want a width sweep too — costs another ~1 GB download.

```bash
python experiments/exp7_gemma_sae_recovery.py
```

Output: `experiments/results/gemma/gemma_sae_recovery.png`.

Compute estimate: ~30–60 min on MPS for two configs (forward pass over 1500 prompts is the bottleneck, the SAE encode/decode is cheap). Add ~15 min per extra config.

---

## 3 · Suggested run order (if you have to budget time)

1. `exp1`, `exp2`, `exp3`, `exp1b` — toy, ~10 min total. Cheapest, biggest delta from quick wins (permutation null + confusion matrix + actual SAE-on-toy).
2. `exp6` — Gemma, ~15 min. The held-out + null controls is the highest-leverage Gemma change.
3. `exp4` — Gemma, ~30 min. Held-out probes give the proper reporting numbers.
4. `exp7` — Gemma, ~45 min for default 2 configs. Most likely to change the headline "dark matter" framing.
5. **Optional**: re-run `PathPatchingGemma.ipynb` with the same 50/50 split, then `exp5` to redraw. This is the only way to fully close the "L22H4 was selected on the same data it's evaluated on" loop.

---

## 4 · What the new numbers should change in the README/paper

After re-running, update `EXPERIMENTS_README.md` and `paper.tex` with:

- **exp1**: replace point estimates `0.904 / 0.289` with `0.904 [CI] / 0.289 [CI]`.
- **exp2**: report `σ1`, `σ̄`, alignment as `REAL vs PERMUTATION-NULL CI` (the Gaussian baseline is still there but should be demoted to a footnote, not the headline comparison).
- **exp3**: add the confusion-matrix story (e.g. "ablating H0 produces X% wrong-but-in-prompt errors vs Y% for H1"), and report the head-swap result.
- **exp1b**: add a paragraph: "to test the geometric inference directly, we trained an L1 SAE on the toy resid_post and report probe accuracies on raw / recon / z / rank-matched controls".
- **exp4**: report the **held-out** numbers (with std), not the CV numbers.
- **exp6**: report `correct vs random-head` and `correct vs random-position` gaps, not just `correct vs distractor`.
- **exp7**: report the full 5-source × 2-config table (not just raw vs recon for the canonical SAE), and discuss what `z >> X_recon` means for the "dark matter" claim.

---

## 5 · Things explicitly NOT done in this pass

- Did not re-train any model (the toy 2L2H model is loaded as-is from HF).
- Did not retrain the Gemma-Scope SAEs (we only sweep the released ones).
- Did not regenerate `gemma/per_head_logit_diffs.pt` — `PathPatchingGemma.ipynb` was left untouched. See the exp5 box above.
- Did not retrain the toy model on a label-shuffled dataset (a stronger null for exp2 than the in-script label permutation, but expensive). The current permutation null is a reasonable cheap proxy.

---

## 6 · Migrating to Gemma 3 + Gemma Scope 2

The Gemma scripts are now parameterised by a single **`GEMMA_PRESET`** env var, so switching between Gemma-2-2b and Gemma-3 is a one-variable flip at invocation time. The presets live in `experiments/_gemma_config.py`.

### Current presets

| `GEMMA_PRESET` | model | FT checkpoint | SAE family | status |
|---|---|---|---|---|
| `gemma-2-2b` (default) | gemma-2-2b | `gemma/gemma2_ft_toy/checkpoint-900` | Gemma-Scope 1 (L0~72 canonical + L0~22 sparser) | **ready** — circuit head filled in |
| `gemma-3-1b-pt` | gemma-3-1b-pt | `None` (FT not yet produced) | Gemma-Scope 2 canonical (placeholder — verify release id) | **needs FT + path-patching** |
| `gemma-3-4b-pt` | gemma-3-4b-pt | `None` | Gemma-Scope 2 canonical (placeholder — verify release id) | **needs FT + path-patching** |

The Gemma-3 presets have `target_layer / target_head / random_head_layer_range` set to `None`. exp6 and exp7 call `require(preset, ...)` and fail with a clear error until those are filled in.

### Run commands

```bash
# Gemma-2-2b (legacy — the headline paper numbers)
python experiments/exp4_gemma_linear_probes.py
python experiments/exp6_gemma_qk_matching.py
python experiments/exp7_gemma_sae_recovery.py

# Gemma-3-1b — works for exp4 today; exp6/exp7 need path-patching first
GEMMA_PRESET=gemma-3-1b-pt python experiments/exp4_gemma_linear_probes.py
GEMMA_PRESET=gemma-3-1b-pt python experiments/exp6_gemma_qk_matching.py   # errors: target_layer/head missing
GEMMA_PRESET=gemma-3-1b-pt python experiments/exp7_gemma_sae_recovery.py  # errors: target_layer missing

# Gemma-3-4b
GEMMA_PRESET=gemma-3-4b-pt python experiments/exp4_gemma_linear_probes.py
```

Output files are now preset-namespaced (e.g. `gemma-3-1b-pt_probe_accuracies_holdout.png`) so both model runs coexist in `experiments/results/gemma/`.

### What the parameterisation handles automatically

- **Model loading** — `load_model(preset, device)` in `_gemma_config.py` either calls `HookedTransformer.from_pretrained(model_name)` (base) or wraps an FT checkpoint via `hf_model=AutoModelForCausalLM.from_pretrained(ft_checkpoint)`, picking based on whether the `ft_checkpoint` path exists. exp4/6/7 all go through this helper.
- **`N_LAYERS`** — read from `model.cfg.n_layers` after load. No hardcoded `26`.
- **GQA group size** — `kv_head_for(q_head, group_size)` uses `group_size = n_q_heads // n_kv_heads` from `model.cfg`. The old Gemma-2-2b-specific `q_head // 2` is gone.
- **Number of query heads** — read from `model.cfg.n_heads`. Random-head null picks from the full head set per layer.
- **SAE ids** — `{layer}` placeholder in the preset's `sae_id` template is substituted with `target_layer` at runtime.

### What still needs manual work per new preset

1. **Re-run `gemma/gemma_toy_ft.ipynb`** on the new model to produce a finetuned checkpoint. Save the output directory and update the preset's `ft_checkpoint` field. (Skip if you intentionally want to operate on base weights — leave `ft_checkpoint=None` and accept that the circuit head you identify won't generalise to base.)
2. **Re-run `gemma/pp_toy_dataset.ipynb`** on the FT checkpoint with the same 50/50 prompt-level split that exp6 uses. Identify the new circuit head `(L*, H*)`.
3. **Fill in `experiments/_gemma_config.py`**: set `target_layer`, `target_head`, `random_head_layer_range` (something like `(L*-4, L*+4)`) for the preset.
4. **Re-export `gemma/per_head_logit_diffs.pt`** from the notebook for the exp5 plot. Preset-namespacing the artefact path is a future nice-to-have; today exp5 still hardcodes `gemma/per_head_logit_diffs.pt`, so swap the file when switching presets.
5. **Verify the Gemma-Scope 2 release id** against Neuronpedia or `sae_lens.toolkit.pretrained_saes_directory`. The placeholder ids in the preset (`gemma-scope-2-1b-pt-res` etc.) are guesses — confirm before running exp7.
6. **Verify `comma_id` and `period_id`** tokenise to single tokens in the new tokenizer. Gemma 3 uses a different tokenizer; the single-token assumption may break. If it does, exp4/exp6/exp7 all silently collect fewer facts.

### Cost estimate

- **Path-patching notebook re-run on Gemma-3-1b**: 1–2 hours on a single GPU. Bottleneck is the per-head activation patching sweep, not the forward pass.
- **Filling in the preset + running exp4/exp6/exp7**: ~15 min active work once path-patching is done. Total wall clock: ~1 hour compute for all three experiments on Gemma-3-1b.
- **Gemma Scope 2 download**: similar per-config size to Gemma Scope 1 (~280 MB for width_16k). Matryoshka SAEs give you multiple ranks from one release.

### Why do this

- **Reviewer question** "why not the most recent model?" — one-sentence defuse: "we replicate on Gemma-3-1b with Gemma-Scope 2 in §X."
- **Matryoshka SAEs** may partially recover (E1,T) where the Gemma-Scope-1 L1 SAE didn't. If they do, the paper's framing changes from "SAEs fail on composed features" to "L1 SAEs at this width fail; Matryoshka partially recovers" — a more nuanced (and arguably more interesting) claim.
- **Gemma Scope 2 transcoders** are a per-layer compute graph, which is a natural fit for the Address-vs-Payload routing story. Adding a `transcoder` source to exp7 alongside `z` is a clean follow-up once the base preset works.
