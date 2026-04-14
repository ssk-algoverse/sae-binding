from eval_sae_common import run_cli


if __name__ == "__main__":
    run_cli(
        default_run_name="hookz_h0_d1024_k8_1M",
        default_checkpoint="saes/b0_hookz_h0_sep_sweep_d1024_k8_1M",
        hook_name="blocks.0.attn.hook_z",
        head_idx=0,
    )
