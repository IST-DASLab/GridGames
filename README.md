# Official Repository for "Grid Games: The Power of Multiple Grids for Quantizing Large Language Models" (NeurIPS 2026)

[arXiv.org](https://arxiv.org/abs/2605.12327) |
[Citation](#citation)

## Repository Structure

The repository consists of:

- [`model_quant.py`](./model_quant.py) - entry point: calibration, quantization and evaluation.
- [`scripts/grid_games.sh`](./scripts/grid_games.sh) - launcher script with all settings exposed as environment variables.
- [`src/`](./src) - the post-training quantization pipeline: CLI, RTN/GPTQ, block-transform math, adaptive grid search, evaluation metrics.
- [`grids/grids.json`](./grids/grids.json) - named quantization fixed grids (codebooks), e.g. `fp4_e2m1`.
- [`inference_lib/`](./inference_lib) - git submodule with the portable **Triton** kernels: block transform, multi-grid quantization, FP4 matmul.

## Environment Setup

<p>
<a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.14-blue.svg"></a>
<a href="https://pytorch.org/get-started/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.14-orange.svg"></a>
<a href="https://developer.nvidia.com/cuda-downloads"><img alt="CUDA 13.2" src="https://img.shields.io/badge/CUDA-13.2-green.svg"></a>
</p>

**Note:** The repository is built on top of the codebase from the paper [WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](https://github.com/IST-DASLab/WUSH).

Below are installation instructions with `uv`.

```bash
# Clone our repository (the Triton kernels live in a submodule)
git clone --recurse-submodules https://github.com/IST-DASLab/GridGames.git
cd GridGames

# Install Python environment
uv venv .venv --python 3.14
source .venv/bin/activate

# Install PyTorch
uv pip install torch torchvision

# Install other dependencies 
uv pip install scipy transformers datasets accelerate matplotlib pytest "lm_eval[hf,ifeval]"

# [Optional] Logging to Weights & Biases
uv pip install wandb
```

## Quantization Pipeline

The entry point is [`model_quant.py`](model_quant.py). The easiest way to run it is [`scripts/grid_games.sh`](scripts/grid_games.sh), **from the repository root** (grids are loaded from `grids/grids.json` relative to the working directory).

Every setting in the script can be overridden with `KEY=VALUE` arguments:

```bash
# Default: Qwen/Qwen3.5-4B, MXFP4 W4A4, RTN, fp4_e2m1 grid for weights and activations
bash scripts/grid_games.sh LOG_WANDB=0

# GPTQ with two weight grids (the kernel picks the best grid per group)
bash scripts/grid_games.sh GPTQ=1 WEIGHT_CODEBOOK_NAMES="fp4_e2m1 int4"

# GPTQ with per-layer adaptive grid search (Hessian-weighted), 2 grids of 16 values each
bash scripts/grid_games.sh GPTQ=1 ADAPTIVE_GRIDS=weighted N_BOOKS=2 GRID_SIZE=16

# NVFP4 with a Hadamard transform, weight-only
bash scripts/grid_games.sh FORMAT=nvfp TRANSFORM_CLASS=hadamard A_BITS=16
```

Grid names must match keys in `grids/grids.json`.

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-4B` | Hugging Face model name or local path |
| `FORMAT` | `mxfp` | `mxfp` (group size 32, E8M0 scales) or `nvfp` (group size 16, E4M3 scales) |
| `W_BITS` / `A_BITS` | `4` / `4` | Set to `16` to leave weights / activations unquantized |
| `GPTQ` | `0` | `1` for GPTQ, `0` for RTN |
| `WEIGHT_CODEBOOK_NAMES` | `fp4_e2m1` | One or more weight grids from `grids/grids.json` |
| `ACT_CODEBOOK_NAMES` | `fp4_e2m1` | One or two activation grids from `grids/grids.json` |
| `ADAPTIVE_GRIDS` | `None` | Per-layer grid search: `weighted` (Hessian) or `datafree` (MSE); replaces `WEIGHT_CODEBOOK_NAMES` |
| `N_BOOKS` / `GRID_SIZE` | `2` / `16` | Number of grids and values per grid for the adaptive search |
| `TRANSFORM_CLASS` | `identity` | `identity`, `hadamard` or `wush` |
| `NUM_SEQUENCES` | `1024` | Calibration sequences (FineWeb-Edu, 2048 tokens each) |
| `EVAL_PERPLEXITY` | `1` | Wikitext-2 perplexity and KL divergence to the original model |
| `EVAL_OPENLLM` | `1` | OpenLLM v1 tasks via `lm_eval` (`LM_EVAL_TASKS`) |
| `LOG_WANDB` | `1` | Log to Weights & Biases; set `WANDB_ENTITY` / `WANDB_PROJECT` in the script to your own |

## Citation

```
@article{egiazarian2026grid,
  title={Grid Games: The Power of Multiple Grids for Quantizing Large Language Models},
  author={Egiazarian, Vage and Schultheis, Erik and Panferov, Andrei and Killian, Earl and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2605.12327},
  year={2026}
}
```