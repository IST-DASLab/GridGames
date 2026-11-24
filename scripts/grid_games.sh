export OMP_NUM_THREADS=16
export HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Allow overrides passed as positional KEY=VALUE arguments.
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        export "$arg"
    else
        echo "Ignoring unrecognized argument: $arg" >&2
    fi
done

MODEL=${MODEL:-"Qwen/Qwen3.5-4B"}
MODEL_ID=$(echo "$MODEL" | awk -F/ '{print $NF}')

NUM_SEQUENCES=${NUM_SEQUENCES:-1024}
SEED=${SEED:-0}

FORMAT=${FORMAT:-"nvfp"}
W_BITS=${W_BITS:-4}
A_BITS=${A_BITS:-4}
W_GROUP_SIZE=${W_GROUP_SIZE:-16}
A_GROUP_SIZE=${A_GROUP_SIZE:-16}
GPTQ=${GPTQ:-0}

# Per-layer adaptive weight grid search: "weighted" (Hessian), "datafree" (MSE) or "None" (use WEIGHT_CODEBOOK_NAMES).
ADAPTIVE_GRIDS=${ADAPTIVE_GRIDS:-"None"}
N_BOOKS=${N_BOOKS:-2}
GRID_SIZE=${GRID_SIZE:-16}


ADDITIONAL_NAME=${ADDITIONAL_NAME:-""}

WEIGHT_CODEBOOK_NAMES=${WEIGHT_CODEBOOK_NAMES-"fp4_e2m1"}
ACT_CODEBOOK_NAMES=${ACT_CODEBOOK_NAMES-"fp4_e2m1"}

TRANSFORM_CLASS=${TRANSFORM_CLASS:-"identity"}

EVAL_PERPLEXITY=${EVAL_PERPLEXITY:-1}
EVAL_OPENLLM=${EVAL_OPENLLM:-1}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-"auto"}
LM_EVAL_BATCH_SIZE_MMLU_GSM8K=${LM_EVAL_BATCH_SIZE_MMLU_GSM8K:-"auto"}
LM_EVAL_TASKS=${LM_EVAL_TASKS:-"mmlu arc_easy arc_challenge winogrande hellaswag lambada piqa gsm8k_llama ifeval"}

DONT_QUANT_SCALES=${DONT_QUANT_SCALES:-0}
LOG_WANDB=${LOG_WANDB:-1}
DTYPE=${DTYPE:-"auto"}
CPU_OFFLOAD_ACTIVATIONS=${CPU_OFFLOAD_ACTIVATIONS:-1}
CPU_OFFLOAD_MODULES=${CPU_OFFLOAD_MODULES:-1}

SCRIPT_ARGS=""

if [[ $DONT_QUANT_SCALES == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --dont_quant_scales"
fi

if [[ $GPTQ == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --gptq"
fi

ADAPTIVE_SLUG=""
if [[ -n $ADAPTIVE_GRIDS && $ADAPTIVE_GRIDS != "None" ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --adaptive_grids ${ADAPTIVE_GRIDS} --n_books ${N_BOOKS} --grid_size ${GRID_SIZE}"
    ADAPTIVE_SLUG="-adaptive_${ADAPTIVE_GRIDS}${N_BOOKS}x${GRID_SIZE}"
fi

if [[ $EVAL_PERPLEXITY == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_perplexity --eval_kl_divergence"
fi

if [[ $EVAL_OPENLLM == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_downstream_tasks"
fi

if [[ $LOG_WANDB == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --log_wandb"
fi

if [[ $CPU_OFFLOAD_MODULES == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_modules"
fi

if [[ $CPU_OFFLOAD_ACTIVATIONS == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_activations"
fi

METHOD_NAME=""
if [[ $GPTQ == 1 ]]; then
    METHOD_NAME="GPTQ"
else
    METHOD_NAME="RTN"
fi

# Drop "" / "None" entries from the lists and pass what's left to
# model_quant.py's --weight_codebook_names / --act_codebook_names (nargs="+").
W_CODEBOOKS=()
A_CODEBOOKS=()
for n in $WEIGHT_CODEBOOK_NAMES; do [[ -n "$n" && "$n" != "None" ]] && W_CODEBOOKS+=("$n"); done
for n in $ACT_CODEBOOK_NAMES;    do [[ -n "$n" && "$n" != "None" ]] && A_CODEBOOKS+=("$n"); done

CODEBOOK_ARGS=""
(( ${#W_CODEBOOKS[@]} > 0 )) && CODEBOOK_ARGS="${CODEBOOK_ARGS} --weight_codebook_names ${W_CODEBOOKS[*]}"
(( ${#A_CODEBOOKS[@]} > 0 )) && CODEBOOK_ARGS="${CODEBOOK_ARGS} --act_codebook_names ${A_CODEBOOKS[*]}"

# Hyphen-joined slugs for the WANDB run name and save path.
W_CODEBOOK_SLUG=$(IFS=- ; echo "${W_CODEBOOKS[*]:-none}")
A_CODEBOOK_SLUG=$(IFS=- ; echo "${A_CODEBOOKS[*]:-none}")

export WANDB_ENTITY=grid_games
export WANDB_PROJECT="GRID_SEARCH"
export WANDB_NAME="${MODEL}/${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}-w_${W_CODEBOOK_SLUG}-a_${A_CODEBOOK_SLUG}${ADAPTIVE_SLUG}-${ADDITIONAL_NAME}"

echo "Starting quantization with the following settings:"
echo "Model: $MODEL"
echo "Format: $FORMAT"
echo "Weight Bits: $W_BITS"
echo "Activation Bits: $A_BITS"
echo "Weight Group Size: $W_GROUP_SIZE"
echo "Activation Group Size: $A_GROUP_SIZE"
echo "Transform Class: $TRANSFORM_CLASS"
echo "Adaptive Grids: $ADAPTIVE_GRIDS (n_books=$N_BOOKS, grid_size=$GRID_SIZE)"
echo "SEED: $SEED"
echo "LM Eval Batch Size: $LM_EVAL_BATCH_SIZE"
echo "LM Eval Batch Size MMLU/GSM8K: $LM_EVAL_BATCH_SIZE_MMLU_GSM8K"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "Script Args: $SCRIPT_ARGS"
echo "Codebook Args: $CODEBOOK_ARGS"

python model_quant.py \
    --model_name_or_path="${MODEL}" \
    --format="${FORMAT}" \
    --w_bits="${W_BITS}" \
    --a_bits="${A_BITS}" \
    --w_group_size="${W_GROUP_SIZE}" \
    --a_group_size="${A_GROUP_SIZE}" \
    --transform_class="${TRANSFORM_CLASS}" \
    ${SCRIPT_ARGS} \
    ${CODEBOOK_ARGS} \
    --dataset_name_or_path="fineweb-edu" \
    --num_sequences="${NUM_SEQUENCES}" \
    --sequence_length=2048 \
    --dtype="${DTYPE}" \
    --lm_eval_batch_size="${LM_EVAL_BATCH_SIZE}" \
    --lm_eval_batch_size_mmlu_gsm8k="${LM_EVAL_BATCH_SIZE_MMLU_GSM8K}" \
    --disable_thinking \
    --seed="${SEED}" \
    --lm_eval_tasks ${LM_EVAL_TASKS} \
    --gen_kwargs="temperature=1.0,do_sample=True,top_k=20,top_p=0.95"