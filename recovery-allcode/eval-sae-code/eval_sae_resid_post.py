from eval_sae_common import run_cli


if __name__ == "__main__":
    run_cli(
        default_run_name="resid_post_d1024_k8_500k",
        default_checkpoint="saes/b0_residpost_sep_d1024_k8_500000",
        hook_name="blocks.0.hook_resid_post",
        head_idx=None,
    )

