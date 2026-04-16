# Reproducing the Paper Experiments

This directory contains the necessary scripts to run all empirical experiments presented in the paper. The scripts generate the plots and metrics confirming the "dark matter" failure mode of Sparse Autoencoders (SAEs) and the staged relational retrieval circuit.

## Setup
Ensure you have `transformer_lens`, `sae_lens`, `scikit-learn`, `datasets`, `torch`, and `seaborn` installed.
```bash
pip install transformer_lens sae_lens scikit-learn datasets torch seaborn
```

All results are automatically saved as PNG files into `experiments/results/` and `experiments/results/gemma/`.

---

## The Experiments

### Priority 0: Toy Model Core Proofs
The following scripts execute against the 2L2H toy transformer (`sebastianhoenig/2L2H_Final`). They prove the exact circuit components and mathematically explain why SAEs fail to recover the composed features.

1. **Combinatorial Geometry Analysis (The "Why")**
    *   **Script:** `python experiments/exp1_geometry_analysis.py`
    *   **Purpose:** Proves mathematically why SAEs fail on composed relations. Computes the intra-class (0.904) vs inter-class (0.289) cosine similarities, proving the representation forms 1,000 dense, non-orthogonal topological clusters that cannot be extracted by L1-sparse methods.
2. **Weight Matrix Decomposition (The "Circuit")**
    *   **Script:** `python experiments/exp2_weight_analysis.py`
    *   **Purpose:** Preempts "ephemeral activation" critiques by tracing the retrieval routing directly to the network weights. Generates the SVD eigenspectrum proving the exact Layer 1 $W_Q W_K^T$ mechanism is hardcoded with massive overlap to the composed address subspace empirically measured at `blocks.0.hook_resid_post` (top principal-angle cosine $\sigma_1 = 0.985$, mean overlap $\bar\sigma = 0.63$ vs. random baseline $\approx 0.24$, subspace alignment score $\Sigma\sigma_i^2/k = 0.54$ vs. random baseline $\approx 0.08 = k/d$). The random baselines are computed inline by `compute_random_baseline()`.
3. **Causal Necessity Ablation**
    *   **Script:** `python experiments/exp3_ablation.py`
    *   **Purpose:** Upgrades causal patching (sufficiency) to strict ablation (necessity). Zeroes out the $L_0H_0$ (Address) and $L_0H_1$ (Payload) representations during forward passes, proving that both are individually essential to retrieval (accuracy zeroes out if either is ablated).

### Priority 2: Gemma-2-2B Replication
The following scripts scale the identical tests up to open-weight LLMs, ensuring the circuit holds natively.

4. **Staged Retrieval Factorization Scaling**
    *   **Script:** `python experiments/exp4_gemma_linear_probes.py`
    *   **Purpose:** Trains linear probes spanning all 26 layers of Gemma-2-2B to extract $E_2$ and $(E_1, T)$ target clusters. Verifies the representation accurately factorizes addresses vs target variables natively.
5. **Gemma Causal Patching Heatmap**
    *   **Script:** `python experiments/exp5_gemma_causal_patching_plot.py`
    *   **Purpose:** Renders the precomputed per-head logit-difference tensor (`gemma/per_head_logit_diffs.pt`, produced by `PathPatchingGemma.ipynb`) into a heatmap. Isolates **Layer 22 Head 4** as the primary functional routing head. *Note: this script is plot-only — run the notebook first to generate the input tensor.*
6. **Gemma Q-K Matching Analysis**
    *   **Script:** `python experiments/exp6_gemma_qk_matching.py`
    *   **Purpose:** Performs exact query-key dot product tests across the context on `L22H4` to mathematically prove it pattern-matches in real LLMs exactly as tested in `exp2/exp3`.
7. **Gemma-Scope SAE Recovery Evaluation (Dark Matter)**
    *   **Script:** `python experiments/exp7_gemma_sae_recovery.py`
    *   **Purpose:** Employs the `sae_lens` library to inject Gemma activations through the official `gemma-scope-2b-pt-res` dict layer mapping. Evaluates target discriminability from $X_{recon}$, proving that even Google's native dictionary erases the compositional axes relative to standalone vectors.
