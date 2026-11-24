"""Command line interface for `model_quant.py`."""

import argparse

from .lm_eval_tasks import ALL_LM_EVAL_TASKS

try:
    import wandb
except ImportError:
    wandb = None


def auto_or_int(value: str):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'")

NVFP_GROUPSIZE=16
MXFP_GROUPSIZE=32

def parse_args(argv: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to quantized model.",
    )

    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to the calibration dataset.",
    )
    parser.add_argument(
        "--sequence_length",
        default=2048,
        type=int,
        help="Length of calibration sequences."
    )
    parser.add_argument(
        "--num_sequences",
        default=1024,
        type=int,
        help="Number of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--format",
        type=str,
        default="int",
        choices=["int", "fp", "nvfp", "mxfp"],
        help="Quantization format.",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Weight bit-width; >= 16 disables weight quantization.",
    )
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="Activation bit-width; >= 16 disables activation quantization.",
    )
    parser.add_argument(
        "--scale_precision",
        type=str,
        default="fp16",
        choices=["fp32","fp16", "e8m0", "e4m3"],
        help="Scale precision.",
    )
    parser.add_argument(
        "--w_group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument("--gen_kwargs",
        type=str,
        default=None,
        help="Kwargs for the generative tasks for the lm_eval",
    )
    parser.add_argument(
        "--weight_codebook_names",
        type=str,
        nargs="+",
        default=None,
        help="One or more weight grid names from grids.json. With >1, per-group best grid is picked by the kernel.",
    )
    parser.add_argument(
        "--act_codebook_names",
        type=str,
        nargs="+",
        default=None,
        help="One or two activation grid names from grids.json (activation quantizer is 2-grid).",
    )
    parser.add_argument(
        "--a_group_size",
        type=int,
        default=None,
        help="How many activation columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--dont_quant_scales",
        action="store_true",
        help="Do not quantize scales",
    )
    # GPTQ params
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="Run GPTQ quantization.",
    )
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument(
        "--adaptive_grids",
        type=str,
        default=None,
        choices=["weighted", "datafree", "None"],
        help="Search weight grids per layer: Hessian-weighted (weighted) or plain MSE (datafree)."
    )
    parser.add_argument("--n_books", type=int, default=2, help="Number of weight grids found by the adaptive grid search.")
    parser.add_argument("--grid_size", type=int, default=16, help="Number of values per grid found by the adaptive grid search.")
    # Transform params
    parser.add_argument(
        "--transform_class",
        type=str,
        default="identity",
        choices=["identity", "hadamard", "wush"],
        help="The transform class."
    )
    parser.add_argument(
        "--hadamard_group_size",
        type=int,
        default=128,
        help="Hadamard group size"
    )
    # Logging params
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log to wandb."
    )
    # Misc params
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--amp", action="store_true", help="whether to enable fp16 autocasting.")
    parser.add_argument("--compile", action="store_true", help="whether to use torch.compile.")
    parser.add_argument("--fuse_global_scale", action="store_true", help="whether to fuse global scale in qkv and gate_up.")
    # Eval params
    parser.add_argument("--eval_perplexity", action="store_true", help="whether to eval perplexity after quantization.")
    parser.add_argument("--eval_downstream_tasks", action="store_true", help="whether to eval downstream (OpenLLM v1) tasks after quantization.")
    parser.add_argument("--eval_kl_divergence", action="store_true", help="whether to eval KL divergence to the original model after quantization.")

    # LM eval params
    parser.add_argument(
        "--lm_eval_batch_size",
        type=auto_or_int,
        default="auto",
        help="LM eval batch size to evaluate after quantization.",
    )
    # LM eval params
    parser.add_argument(
        "--lm_eval_batch_size_mmlu_gsm8k",
        type=auto_or_int,
        default=16,
        help="LM eval batch size to evaluate after quantization.",
    )

    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        type=str,
        choices=ALL_LM_EVAL_TASKS,
        default=["mmlu", "mmlu_cot_llama", "mmlu_pro", "arc_easy", "arc_challenge", "winogrande", "hellaswag", "lambada", "piqa", "gsm8k", "gsm8k_llama", "ifeval"],
        help="OpenLLMv1 tasks to evaluate after quantization."
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Whether to disable thinking mode for Qwen3.",
    )
    # Parse arguments
    args = parser.parse_args(argv)
    # Normalize "None" coming from the shell script
    if args.adaptive_grids == "None":
        args.adaptive_grids = None
    # Check and fix group_size (if needed)
    if args.format == "nvfp" :
        if args.w_group_size != NVFP_GROUPSIZE:
            args.w_group_size = NVFP_GROUPSIZE
            print(f"Changed weight group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.a_group_size != NVFP_GROUPSIZE:
            args.a_group_size = NVFP_GROUPSIZE
            print(f"Changed activation group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.scale_precision != "e4m3":
            args.scale_precision = "e4m3"
            print(f"Changed scale_precision to e4m3 for nvfp format.")
    elif args.format == "mxfp":
        if args.w_group_size != MXFP_GROUPSIZE:
            args.w_group_size = MXFP_GROUPSIZE
            print(f"Changed weight group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.a_group_size != MXFP_GROUPSIZE:
            args.a_group_size = MXFP_GROUPSIZE
            print(f"Changed activation group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.scale_precision != "e8m0":
            args.scale_precision = "e8m0"
            print(f"Changed scale precision to e8m0 for mxfp format.")
    if args.dont_quant_scales:
        args.scale_precision = "fp32"
    # Check logging
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
    return args