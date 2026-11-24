"""Registry of the lm-eval tasks run after quantization.

Shared by the CLI (it drives the `--lm_eval_tasks` choices) and the evaluation
loop (it drives which HFLM instance / batch size each task runs under).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LMEvalTask:
    """A single lm-eval task and the way it should be run/logged."""

    name: str
    wandb_key: str = "eval/openllm"
    num_fewshot: Optional[int] = None
    # Generative tasks are run through the chat template with `--gen_kwargs`.
    generative: bool = False


@dataclass
class LMEvalGroup:
    """A set of tasks sharing one HFLM instance (and therefore one batch size)."""

    batch_size_arg: str
    tasks: list = field(default_factory=list)


# Task groups are split by batch size: the heavier (mostly generative) tasks
# use `--lm_eval_batch_size_mmlu_gsm8k`, the rest use `--lm_eval_batch_size`.
LM_EVAL_GROUPS = [
    LMEvalGroup(
        batch_size_arg="lm_eval_batch_size",
        tasks=[
            LMEvalTask("winogrande", num_fewshot=5),
            LMEvalTask("arc_easy", wandb_key="eval/oneshot"),
            LMEvalTask("arc_challenge", wandb_key="eval/oneshot"),
            LMEvalTask("lambada", wandb_key="eval/oneshot"),
            LMEvalTask("piqa", wandb_key="eval/oneshot"),
            LMEvalTask("hellaswag", wandb_key="eval/oneshot"),
        ],
    ),
    LMEvalGroup(
        batch_size_arg="lm_eval_batch_size_mmlu_gsm8k",
        tasks=[
            LMEvalTask("hellaswag", num_fewshot=10),
            LMEvalTask("mmlu"),
            LMEvalTask("mmlu_cot_llama", generative=True),
            LMEvalTask("ifeval", generative=True),
            LMEvalTask("mmlu_pro", generative=True),
            LMEvalTask("gsm8k_llama", generative=True),
            LMEvalTask("gsm8k", generative=True),
        ],
    ),
]

ALL_LM_EVAL_TASKS = sorted({task.name for group in LM_EVAL_GROUPS for task in group.tasks})