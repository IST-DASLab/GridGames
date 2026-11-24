import json
import warnings
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from src.metrics.perplexity import compute_perplexity
from src.quantization import   grid_quantization_gptq
from src.utils.common_utils import fix_seed, clear_device_cache
from src.utils.arg_utils import parse_args
from src.utils.data_utils import get_data, get_wikitext2
from src.metrics.kl_divergence import compute_kl_divergence
from src.metrics.evaluation import compute_baseline_topk, compute_kl_and_ear
from src.utils.lm_eval_tasks import LM_EVAL_GROUPS


try:
    import wandb
except ImportError:
    wandb = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False



# Evaluation settings shared by the KL and OpenLLM evaluations
EVAL_MAX_LENGTH = 4096  # from OpenLLM v1
KL_DATASET = "c4"
KL_NUM_SEQUENCES = 512
KL_TOPK = 10
KL_TEACHER_BATCH_SIZE = 4
KL_STUDENT_BATCH_SIZE = 2


def print_results(results: dict):
    print(make_table({"results": results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))

def _resolve_grids(names, codebooks, device, dtype=None):
    if not names or not codebooks:
        return None
    return [torch.as_tensor(codebooks[n], dtype=dtype, device=device) for n in names]


def evaluate_kl_divergence(model, tokenizer, args, device):
    """Compare the model against the original (unquantized) checkpoint on C4 and Wikitext-2."""
    # `compute_kl_divergence` runs both models on the same batch, so there is no
    # block-wise offloading to be had here: the teacher is loaded straight onto
    # `device` and stays resident alongside the student.
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    teacher_model.config.use_cache = False
    teacher_model.requires_grad_(False)

    c4_data = get_data(KL_DATASET, tokenizer, EVAL_MAX_LENGTH, KL_NUM_SEQUENCES, 0)
    c4_batch = torch.cat(c4_data, dim=0)

    # Top-k KL and exact agreement rate on C4
    baseline_topk_vals, baseline_topk_idx = compute_baseline_topk(
        teacher_model, c4_batch, KL_TEACHER_BATCH_SIZE, KL_TOPK
    )
    topk_kl, ear, _ = compute_kl_and_ear(
        model, baseline_topk_vals, baseline_topk_idx, c4_batch, KL_STUDENT_BATCH_SIZE
    )
    print(f"C4 top-k KL: {topk_kl}, EAR: {ear}")
    if args.log_wandb:
        wandb.log({"eval/top_k_kl_c4": topk_kl, "eval/top_k_ear_c4": ear})
    del baseline_topk_vals, baseline_topk_idx, c4_batch

    # Full KL on Wikitext-2 and C4
    for name, wandb_key, data in [
        ("Wikitext-2", "eval/kl_wiki2", get_wikitext2(tokenizer, EVAL_MAX_LENGTH)),
        ("C4", "eval/kl_c4", c4_data),
    ]:
        kl = compute_kl_divergence(
            quantized_model=model,
            teacher_model=teacher_model,
            data=data,
            batch_size=1,
            pad_token_id=tokenizer.pad_token_id,
            temperature=1.0,
        )
        print(f"{name} KL divergence: {round(kl, 5):.5f}")
        if args.log_wandb:
            wandb.log({wandb_key: kl})

    del teacher_model
    clear_device_cache(garbage_collection=True)


def evaluate_openllm(model, tokenizer, args):
    """Run the requested lm-eval tasks and return their merged results."""
    results = {}
    for group in LM_EVAL_GROUPS:
        selected = [task for task in group.tasks if task.name in args.lm_eval_tasks]
        if not selected:
            continue

        batch_size = getattr(args, group.batch_size_arg)
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=EVAL_MAX_LENGTH,
        )
        task_manager = TaskManager()

        for task in selected:
            extra_kwargs = {}
            if task.generative:
                extra_kwargs = dict(
                    apply_chat_template=True,
                    fewshot_as_multiturn=True,
                    gen_kwargs=args.gen_kwargs,
                )
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks=task.name,
                num_fewshot=task.num_fewshot,
                batch_size=batch_size,
                task_manager=task_manager,
                **extra_kwargs,
            )["results"]
            results.update(task_results)
            print_results(task_results)
            if args.log_wandb:
                wandb.log({task.wandb_key: task_results})
            clear_device_cache(garbage_collection=True)

        del lm, task_manager
        clear_device_cache(garbage_collection=True)

    return results


def main():
    args = parse_args()
    # Fix seed
    fix_seed(args.seed)
    # Set device
    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    # Get dtype
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Init logger
    if args.log_wandb:
        wandb.init(config=args)

    with open("grids/grids.json", "r") as f:
            codebooks = json.load(f)
    
    if args.w_bits >= 16:
        print("Not quantizing weights")
        args.grids_w = None
    else:
        args.grids_w = _resolve_grids(args.weight_codebook_names, codebooks=codebooks, device=device, dtype=torch.float32)
        if args.adaptive_grids not in ["weighted", "datafree"]:
            assert args.grids_w is not None, "Must provide --weight_codebook_names when w_bits < 16."
        if args.grids_w is not None and len(args.grids_w) > 1:
            print(f"Will run multi-grid ({len(args.grids_w)}) quantization for weights")

    if args.a_bits >= 16:
        print("Not quantizing activations")
        args.grids_a = None
    else:
        args.grids_a = _resolve_grids(args.act_codebook_names, codebooks=codebooks, device=device, dtype=torch.float32)
        assert args.grids_a is not None, "Must provide --act_codebook_names when a_bits < 16."
        assert 1 <= len(args.grids_a) <= 2, "Activation quantizer supports 1 or 2 grids (--act_codebook_names)."
        if len(args.grids_a) > 1:
            print(f"Will run multi-grid ({len(args.grids_a)}) quantization for activations")

    print("Weight grids:", args.weight_codebook_names if args.grids_w is not None else None, args.grids_w)
    print("Activation grids:", args.act_codebook_names if args.grids_a is not None else None, args.grids_a)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=args.dtype, 
        device_map=None if args.cpu_offload_modules else device,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Sanity check
    if args.eval_downstream_tasks:
        assert hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None, "OpenLLM v1 works only with chat template."
        if args.disable_thinking:
            if "qwen3" in model.config.model_type:
                tokenizer.apply_chat_template = partial(
                    tokenizer.apply_chat_template, 
                    enable_thinking=False
                )
            else:
                warnings.warn("`disable_thinking` has no effect on non-Qwen3 models.")


    # Prepare calibration data
    calibration_data = get_data(
        args.dataset_name_or_path,
        tokenizer,
        args.sequence_length,
        args.num_sequences,
        args.seed
    )
    print("Calibration data size:", len(calibration_data))

    if not args.cpu_offload_modules:
        model.to(device)

    if args.w_bits<16 or args.a_bits<16:
        grid_quantization_gptq(model, calibration_data, args, device)

    if args.cpu_offload_modules:
        model.to(device)

    if args.compile:
        model = torch.compile(model)

    if args.eval_perplexity:
        eval_data = get_wikitext2(tokenizer, args.sequence_length)
        ppl = compute_perplexity(model, eval_data)
        print(f"Wikitext-2 perplexity: {round(ppl, 2):.2f}")
        if args.log_wandb:
            wandb.log({"eval/wikitext2_ppl": ppl})

    if args.eval_kl_divergence:
        evaluate_kl_divergence(model, tokenizer, args, device)

    if args.eval_downstream_tasks:
        results = evaluate_openllm(model, tokenizer, args)
        print("### Final results ###")
        print_results(results)


if __name__ == "__main__":
    main()
