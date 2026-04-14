# Mapping of `paper.tex` Results to Notebooks

Based on the sections in `paper.tex` and the contents of the `recovery-allcode` directory, here is how the results reported in the paper map to the specific Jupyter notebooks:

### 4.1 A Staged Retrieval Circuit
* **Paper Result**: Linear probes reveal that Layer 0 factorizes identity and payload vectors (Table 1: Fact-side and Query-side probes).
* **Notebook**: `recovery-allcode/circuit-stuff/Relational Composition Analysis.ipynb`
    * *Evidence*: This notebook contains the setup and execution for probing on `blocks.0.attn.hook_z` and other locations, matching the data shown in Table 1.

### 4.2 Retrieval via Same-Head Query-Key Matching
* **Paper Result**: Layer 1 performs retrieval using same-head query-key dot products between the question mark token and the preceding commas (100% top-1 rate, huge positive margins).
* **Notebook**: `recovery-allcode/circuit-stuff/Relational Composition Analysis.ipynb`
    * *Evidence*: Circuit analysis code related to attention patterns and compositions are located here.

### 4.3 Causal Validation of the Address/Payload Split
* **Paper Result**: Causal patching interventions on the address representations at the comma tokens (swapping class-mean representations to flip the model's prediction).
* **Notebook**: `recovery-allcode/circuit-stuff/Part1_1.4_ARENA-IOI-Replication-On-Our-Model.ipynb`
    * *Evidence*: Contains extensive usage of `transformer_lens.patching` methods (`get_act_patch_resid_pre`, `get_act_patch_attn_head_out_all_pos`, etc.) and logit diff calculations (`clean_logit_diff` vs `corrupted_logit_diff`) matching the causal interventions cited in the paper.

### 4.4 SAE Feature Recovery Pre- vs. Post-Composition
* **Paper Result (Pre-Composition)**: SAEs applied at the output of the address head (`blocks.0.attn.hook_z`), cleanly recovering relational concepts `E1` and `T`.
* **Notebook**: `recovery-allcode/sae-stuff/hookz-analyse.ipynb`
    * *Evidence*: Computes SAE feature recovery F1-scores specifically for the `hook_z` activation site.

* **Paper Result (Post-Composition)**: SAEs applied after composition in the residual stream (`blocks.0.hook_resid_post`), showing near-perfect payload recovery (`E2`) but catastrophic degradation of address components.
* **Notebook**: `recovery-allcode/sae-stuff/resid-analyse.ipynb`
    * *Evidence*: Computes SAE feature recovery F1-scores specifically for the `resid_post` activation site.
