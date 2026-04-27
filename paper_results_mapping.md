# Mapping of `paper.tex` Results to Notebooks

Based on the sections in `paper.tex` and the contents of the `recovery-allcode` directory, here is how the results reported in the paper map to the specific Jupyter notebooks:

### 4.1 A Staged Retrieval Circuit
* **Paper Result**: Linear probes reveal that Layer 0 factorizes identity and payload vectors (Table 1: Fact-side and Query-side probes).
* **Primary Notebook**: `recovery-allcode/circuit-stuff/Part1_1.4_ARENA-IOI-Replication-On-Our-Model.ipynb`
    * *Evidence*: Contains the bulk of the linear probe code (47 `probe` references, uses `RidgeClassifier`/`LogisticRegression`) targeting $E_2$, $E_1$, $T$, and $(E_1, T)$ at the comma and `?` positions across `blocks.0.hook_resid_post`, `blocks.0.hook_resid_pre`, and `blocks.0.attn.hook_z` (heads 0 and 1) — matching every row in Table 1.
* **Supporting Notebook**: `recovery-allcode/circuit-stuff/Relational Composition Analysis.ipynb`
    * *Evidence*: Contains the compositional-probing hypothesis test ($z(E_1, T) = \text{dir}(E_1) + \text{dir}(T)$, cos $\approx 0.96$–$0.97$) that motivates the factorized structure reported in §4.1 but not tabulated in Table 1 itself.

### 4.2 Retrieval via Same-Head Query-Key Matching
* **Paper Result**: Layer 1 performs retrieval using same-head query-key dot products between the `?` token and the preceding commas (100% top-1 rate, margin $+103.2$ for L1H0, $+102.5$ for L1H1).
* **Notebook**: `recovery-allcode/circuit-stuff/Part1_1.4_ARENA-IOI-Replication-On-Our-Model.ipynb`
    * *Evidence*: Contains the query/key extraction code (`blocks.1.attn.hook_q`, `blocks.1.attn.hook_k`), same-head vs cross-head dot-product sweeps, top-1 rate computation, and the retrieval-margin value $103.2$ cited in the paper. (The `Relational Composition Analysis.ipynb` notebook does NOT contain this analysis — no `hook_q`/`hook_k`/`top1` references.)

### 4.3 Causal Validation of the Address/Payload Split
* **Paper Result**: Causal patching interventions on the address representations at the comma tokens (swapping class-mean representations to flip the model's prediction; clean logit diff $-24.67$, patched $+24.60$). Mean-ablation accuracies: L0H0 $\to 2.15\%$, L0H1 $\to 0.9\%$.
* **Primary Notebook**: `recovery-allcode/circuit-stuff/Part1_1.4_ARENA-IOI-Replication-On-Our-Model.ipynb`
    * *Evidence*: Contains extensive usage of `transformer_lens.patching` methods (`get_act_patch_resid_pre`, `get_act_patch_attn_head_out_all_pos`, etc.), logit-diff calculations (`clean_logit_diff` vs `corrupted_logit_diff`), and the ablation result `2.15` cited in §4.3.
* **Supporting Notebook**: `recovery-allcode/circuit-stuff/Relational Composition Analysis.ipynb`
    * *Evidence*: Cells 61–74 implement `causal_intervention_analysis` with `clean_margin`/`patched_margin_part1`/`patched_margin_part2`, generating the class-mean address swaps visualized in Figure~\ref{fig:causal}.

### 4.4 SAE Feature Recovery Pre- vs. Post-Composition
* **Paper Result (Pre-Composition)**: SAEs applied at the output of the address head (`blocks.0.attn.hook_z`), cleanly recovering relational concepts `E1` and `T`.
* **Notebook**: `recovery-allcode/sae-stuff/hookz-analyse.ipynb`
    * *Evidence*: Computes SAE feature recovery F1-scores specifically for the `hook_z` activation site.

* **Paper Result (Post-Composition)**: SAEs applied after composition in the residual stream (`blocks.0.hook_resid_post`), showing near-perfect payload recovery (`E2`) but catastrophic degradation of address components.
* **Notebook**: `recovery-allcode/sae-stuff/resid-analyse.ipynb`
    * *Evidence*: Computes SAE feature recovery F1-scores specifically for the `resid_post` activation site.
