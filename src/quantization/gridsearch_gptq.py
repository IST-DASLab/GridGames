import argparse
import functools
import itertools
import torch
from torch import nn
from transformers import PreTrainedModel
from inference_lib.triton.PO2_quant_dequant import quantize_N_triton, dequantize_N_triton
from .chosing_grid import search_M, search_M_hessian, generate_fp_codebook

import scipy.linalg

try:
    import wandb
except ImportError:
    wandb = None


get_normalized_hadamard_transform = lambda size, dtype=torch.float64, device=torch.device('cpu'): torch.as_tensor(scipy.linalg.hadamard(size), dtype=dtype, device=device) * size ** -.5

from .accumulate_hessian import accumulate_hessian
from ..utils.common_utils import clear_device_cache, to, maybe_first_element
from ..utils.model_utils import InputCollector, ForwardInterrupt
from ..utils.gridgames_utils import QuantLinear, BlockTransformQuantizerGrid


torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")

def triton_kernel_wrapper(x: torch.Tensor, existing_scale: torch.Tensor | None = None, grids: list[torch.Tensor] | None = None, gs: int = 16,
                           scale_quant=False, global_scale: torch.Tensor | None = None, grid_ids: torch.Tensor | None = None):
    assert grids is not None and len(grids) > 0, "triton_kernel_wrapper requires a non-empty list of grids"
    q, scales_out, grid_ids_out, tail = quantize_N_triton(
        x, grids=grids, gs=gs, scale_quant=scale_quant, global_scale=global_scale,
        scales=existing_scale, grid_ids=grid_ids,
    )
    x_dequantized = dequantize_N_triton(
        q, scales=scales_out, grid_ids=grid_ids_out, grids=grids, gs=gs, tail=tail, dtype=x.dtype,
    )
    result: dict = {
        'quantized': q,
        'fake_quant': x_dequantized.to(dtype=x.dtype),
        'scale_quant': scales_out,
        'global_scale': global_scale,
        'grid_ids': grid_ids_out,
    }
    return result

def gptq_babai_outer(
        parameter: torch.Tensor,
        hessian: torch.Tensor | None = None,
        inner_fn=lambda basis, y, **_: {'coefficient': torch.linalg.solve_triangular(basis, y, upper=True, left=True, unitriangular=False)},
        collate_fn=lambda *results, **_: torch.cat([r['coefficient'] for r in results], dim=-2).transpose(-2, -1),
        block_size: int = -1,
        force_rtn_outer: bool = False,
) -> dict | torch.Tensor:
    """
    GPTQ/LDLQ/Babai outer quantization
    parameter: (..., R, C)
    hessian: (..., C, C), None: RTN fallback
    inner_fn: function, inner quantization function
    collate_fn: function
    block_size: int, block size for processing channels, B
    force_rtn_outer: bool, RTN fallback
    returns: list[dict]
    """
    dtype, high_dtype = parameter.dtype, torch.float64
    device: torch.device = parameter.device

    n_vectors: int = parameter.size(-1)  # C
    if block_size <= 0:
        block_size = n_vectors

    x: torch.Tensor = parameter.transpose(-2, -1).to(dtype=high_dtype, device=device, copy=True, memory_format=torch.contiguous_format)  # (..., C, R), coefficient, updated in-place

    # decompose hessian matrix
    if hessian is not None:
        hessian_copy: torch.Tensor = hessian.to(dtype=high_dtype, device=device, copy=True, memory_format=torch.contiguous_format)  # (..., C, C)
    else:
        hessian_copy: torch.Tensor = torch.eye(n_vectors, dtype=high_dtype, device=device)  # (C, C)
    damp_ratio_cholesky: float = 5e-2
    diag_indices: torch.Tensor = torch.arange(n_vectors, dtype=torch.int64, device=device)  # (C)
    hessian_copy[..., diag_indices, diag_indices] += damp_ratio_cholesky * hessian_copy.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # (..., C) <= (..., 1)
    max_try: int = 100
    basis: torch.Tensor = torch.empty_like(hessian_copy)  # (..., N=C, C)
    info: torch.Tensor = torch.empty(hessian_copy.shape[:-2], dtype=torch.int32, device=device)  # (...)
    while (max_try := max_try - 1) >= 0:
        torch.linalg.cholesky_ex(hessian_copy, upper=True, check_errors=False, out=(basis, info))  # basis: (..., N=C, C), upper triangular, column vectors, basis.t() @ basis = hessian_copy
        if not info.to(dtype=torch.bool).any():
            break
        hessian_copy[..., diag_indices, diag_indices] += damp_ratio_cholesky * hessian_copy.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # (..., C) <= (..., 1)
    assert max_try >= 0, 'Hessian decomposition failed. Please try using more samples or increasing damp_ratio.'

    y: torch.Tensor = basis @ x  # (..., N=C, R), target (residual) column vectors, updated in-place

    # blockwise babai's nearest plane algorithm
    results: list[dict] = []
    # print(y.shape)
    for i1 in range((n_vectors - 1) // block_size * block_size, -1, -block_size):
        i2: int = i1 + block_size
        results.append(inner_fn(
            basis=basis[..., i1:i2, i1:i2],  # (..., N=B, B)
            y=y[..., i1:i2, :],  # (..., N=B, R)
        ))
        if not force_rtn_outer:
            x[..., i1:i2, :] = results[-1]['coefficient']  # (..., B, R)
        y[..., :i1, :] -= basis[..., :i1, i1:i2] @ x[..., i1:i2, :]  # (..., ?, R)
    results_collate = collate_fn(*results[::-1])
    # here we need to extract fake
    return results_collate


def block_transform_gptq_babai_inner(
        basis: torch.Tensor,
        y: torch.Tensor,
        quant_fn=lambda x, existing_scale=None, **_: {'fake_quant': x},
        basis_transform_fn=lambda basis, y, **_: torch.eye(basis.size(-1), dtype=basis.dtype, device=basis.device),
        force_rtn_inner: bool = False,
) -> dict:
    """
    inner function for gptq quantization with block transform
    y <=> basis @ x
      <=> basis @ transform.inv() @ (transform @ x)
      <=> basis @ transform_basis.t() @ x_fake_quant
      <=> basis @ x_coefficient
    basis: (..., N=B, B), upper triangular, column vectors
    y: (..., N=B, R), column vectors
    quant_fn: function to perform fake quantization
    basis_transform_fn: function to compute the transform for the basis
    force_rtn_inner: bool, RTN fallback
    returns: dict
    """
    # x: torch.Tensor = torch.linalg.solve_triangular(basis, y, upper=True, left=True, unitriangular=False)  # (..., B, R), basis @ x = y
    transform_basis: torch.Tensor = basis_transform_fn(basis=basis, y=y)  # (..., B, B)

    rotation, basis_new = torch.linalg.qr(basis @ transform_basis.transpose(-2, -1), mode='reduced')  # (..., N=B, N=B) orthogonal, (..., N=B, B) upper triangular column vectors
    y_new: torch.Tensor = rotation.transpose(-2, -1) @ y  # (..., N=B, R), updated in-place

    x_transformed: torch.Tensor = torch.linalg.solve_triangular(
        basis_new,  # (..., N=B, B), upper triangular, column vectors
        y_new,  # (..., N=B, R)
        upper=True,
        left=True,
        unitriangular=False,
    )  # (..., B, R), basis @ transform_basis.t() @ x_transformed = y <=> basis_new @ x_transformed = y_new
    # print(x_transformed.shape)
    result: dict = quant_fn(x_transformed.transpose(-2, -1), existing_scale=None)

    if force_rtn_inner:
        x_transformed: torch.Tensor = result['fake_quant'].transpose(-2, -1)  # (..., B, R)
    else:  # babai's nearest plane algorithm
        for i1 in range(basis_new.size(-1) - 1, -1, -1):
            i2: int = i1 + 1
            x_transformed[..., i1:i2, :] = y_new[..., i1:i2, :] / basis_new[..., i1:i2, i1:i2]
            result: dict = quant_fn(x_transformed.transpose(-2, -1), existing_scale=result['scale_quant'], grid_ids=result['grid_ids'])
            x_transformed: torch.Tensor = result['fake_quant'].transpose(-2, -1)  # (..., B, R)
            y_new[..., :i1, :] -= basis_new[..., :i1, i1:i2] * x_transformed[..., i1:i2, :]

    x_coefficient: torch.Tensor = transform_basis.transpose(-2, -1) @ x_transformed  # (..., B, R)
    result['coefficient'] = x_coefficient  # (..., B, R)
    result['transform_basis'] = transform_basis  # (..., B, B)
    return result


def collate_xxfp4(
        *results,
        dtype: torch.dtype,
) -> dict:
    """
    collate packed xxfp4 results
    results: list[dict]
    dtype: torch.dtype
    returns: (..., R, C)
    """

    x: torch.Tensor = torch.cat([result['fake_quant'] for result in results], dim=-1).to(dtype=dtype, copy=False)  # (..., R, C)
    if results[0]['quantized'] is not None:
        quant_weight: torch.Tensor = torch.cat([result['quantized'] for result in results], dim=-1)  # (..., C//2)
        quant_weight = quant_weight.cpu()
    else:
        quant_weight=None
    if results[0]['scale_quant'] is not None:
        scale_quant: torch.Tensor = torch.cat([result['scale_quant'] for result in results], dim=-1)  # (..., C//B)
        scale_quant = scale_quant.cpu()
    else:
        scale_quant=None
    transform_basis: torch.Tensor = torch.stack([result['transform_basis'].to(dtype=dtype) for result in results], dim=-3)  # (..., C//B, B, B)
    collated_results: dict = {'fake_quant': x, 'transform_basis': transform_basis,'global_scale': results[0]['global_scale'], 'qweight': quant_weight , 'scale_quant': scale_quant}
    return collated_results


def get_transform(
        basis: torch.Tensor,
        y: torch.Tensor,
        transform_type: str = 'identity',
        transform_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """

    basis: (..., B, B)
    y: (..., B, R)
    """
    dtype, high_dtype = basis.dtype, torch.float64
    device: torch.device = basis.device
    transform_block_size: int = basis.size(-1)

    match transform_type:
        case 'identity':
            transform: torch.Tensor = torch.eye(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
        case 'random_rotation':
            transform: torch.Tensor = torch.linalg.qr(torch.randn(*basis.shape[:-2], transform_block_size, transform_block_size, dtype=dtype, device=device), mode='reduced').Q  # (..., B, B)
        case 'hadamard':
            transform: torch.Tensor = get_normalized_hadamard_transform(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
        case 'wush' | 'wus':
            if transform_type == 'wus':
                hadamard: torch.Tensor = torch.eye(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
            else:
                hadamard: torch.Tensor = get_normalized_hadamard_transform(transform_block_size, dtype=high_dtype, device=device)  # (B, B)
            v, s, uh = torch.linalg.svd(y, full_matrices=False)  # (..., B, B), (..., B), (..., B, R)
            s *= y.size(-1) ** -.5  # (..., B)
            damp_ratio_eigen: float = 1e-2
            s += s.mean(dim=-1, keepdim=True) * damp_ratio_eigen  # (..., B) <= (..., 1)
            transform: torch.Tensor = torch.linalg.solve_triangular(
                basis.transpose(-2, -1),  # (..., B, B)
                hadamard * s[..., None, :] ** .5 @ v.transpose(-2, -1),  # (..., B, B)
                upper=False,
                left=False,
                unitriangular=False,
            )  # (..., B, B), T_{hsvx}^{-\top}, transform = hadamard @ diag(s ** .5) @ v.t() @ basis.t().inv()
        case _:
            raise NotImplementedError

    transform: torch.Tensor = transform.to(dtype=transform_dtype).to(dtype=dtype)
    assert transform.isfinite().all()
    return transform


def block_transform(
        x: torch.Tensor,
        transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    compute x @ transform.t()
    x: (..., C)
    transform: (..., C//B, B, B)
    returns: (..., C)
    """
    dtype, high_dtype = x.dtype, torch.float64

    if transform is None:
        return x  # (..., C)

    x_transformed: torch.Tensor = (
        x.unflatten(dim=-1, sizes=(-1, 1, transform.size(-1))).to(dtype=high_dtype)  # (..., C//B, 1, B)
        @
        transform.transpose(-2, -1).to(dtype=high_dtype)  # (..., C//B, B, B)
    ).flatten(start_dim=-3).to(dtype=dtype)  # (..., C)
    return x_transformed  # (..., C)


def _grids_max_repr(grids: list[torch.Tensor]) -> torch.Tensor:
    """Return min over grids of (max(|g|), -min(g) if min(g) <= 0 else inf), preserving original nvfp4 logic."""
    vals = []
    for g in grids:
        vals.append(g.abs().max())
        vals.append(-g.min() if g.min() <= 0 else torch.as_tensor(float("inf"), device=g.device))
    return torch.stack([v.to(grids[0].device) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=grids[0].device) for v in vals]).min()


def adaptive_grid_search(
        weight: torch.Tensor,
        hessian: torch.Tensor | None = None,
        hessian_hold: torch.Tensor | None = None,
        transform_class: str = 'identity',
        block_size: int = 32,
        candidates: torch.Tensor = generate_fp_codebook("fp6_e2m3fn"),
        n_books: int = 2,
        k: int = 16,
) -> list[torch.Tensor]:
    """
    search n_books grids in the transformed domain, i.e. where block_transform_gptq_babai_inner quantizes
    x_transformed = transform @ x, hessian_transformed = transform @ hessian @ transform.t() (orthogonal transform)
    weight: (R, C)
    hessian: (C, C), None: unweighted MSE search
    hessian_hold: (C, C), held-out hessian for evaluation, None: use hessian
    transform_class: str, same as get_transform's transform_type
    block_size: int, quantization group size == transform block size, B
    returns: list of n_books grids
    """
    assert transform_class not in ('wush', 'wus'), 'dont want to deal with (wush/wus) for now'
    dtype, high_dtype = weight.dtype, torch.float64
    device: torch.device = weight.device
    n_vectors: int = weight.size(-1)  # C
    n_blocks: int = n_vectors // block_size  # C//B
    assert n_blocks * block_size == n_vectors

    # data-independent transforms only use basis for its shape
    transform_basis: torch.Tensor = get_transform(
        basis=torch.eye(block_size, dtype=high_dtype, device=device).expand(n_blocks, block_size, block_size),  # (C//B, B, B)
        y=None,
        transform_type=transform_class,
        transform_dtype=dtype,
    ).expand(n_blocks, block_size, block_size)  # (C//B, B, B)

    weight_transformed: torch.Tensor = block_transform(weight.to(dtype=high_dtype), transform=transform_basis).float()  # (R, C)

    if hessian is None:
        best_M, res = search_M(weight_transformed, candidates, k=k, gs=block_size, n_books=n_books)
        books = res[best_M][1]
    else:
        transform_full: torch.Tensor = torch.block_diag(*transform_basis)  # (C, C)
        rotate = lambda h: (transform_full @ h.to(dtype=high_dtype, device=device) @ transform_full.transpose(-2, -1)).float()  # (C, C)
        hessian_transformed: torch.Tensor = rotate(hessian)
        hessian_hold_transformed: torch.Tensor = rotate(hessian_hold) if hessian_hold is not None else hessian_transformed
        best_M, res = search_M_hessian(weight_transformed, hessian_transformed, hessian_hold_transformed, candidates, k=k, gs=block_size, n_books=n_books)
        books = res[best_M]['books']

    return [books[j].float().to(device) for j in range(n_books)]

@torch.no_grad()
def quantize_layers(
        *weights,
        hessian: torch.Tensor,
        quant_type: str,
        transform_class: str,
        force_rtn: bool = False,
        grids_w: list[torch.Tensor] | None = None,
        grids_a: list[torch.Tensor] | None = None,
        args

) -> list[nn.Module]:
    """
    Quantize the weight matrix using GPTQ (or skip weight quantization when grids_w is None).
    """
    weight: torch.Tensor = torch.cat(weights, dim=0)
    dtype, device = weight.dtype, weight.device

    match quant_type:
        case 'mxfp4':
            block_size: int = 32
            scale_dtype = torch.float8_e8m0fnu if args.scale_precision not in ["fp32", "fp16"] else torch.float32
        case 'nvfp4':
            block_size: int = 16
            scale_dtype = torch.float8_e4m3fn if args.scale_precision not in ["fp32", "fp16"] else torch.float32
        case _:
            raise NotImplementedError(f"Unsupported quant_type '{quant_type}' (use --format mxfp or nvfp).")

    if scale_dtype is torch.float8_e8m0fnu:
        scale_quant_type = "e8m0"
    elif scale_dtype is torch.float8_e4m3fn:
        scale_quant_type = "e4m3"
    else:
        scale_quant_type = False

    if args.adaptive_grids in ["weighted", "datafree"]:
        grids_w = adaptive_grid_search(
            weight.detach().clone().to(device).float(),
            hessian=hessian if args.adaptive_grids == "weighted" else None,
            transform_class=transform_class,
            block_size=block_size,
            n_books=args.n_books,
            k=args.grid_size,
        )
        print("using", grids_w)
        assert len(grids_w) == args.n_books

    if grids_w is None:
        # Weights stay in high precision. QuantLinear expects weights in the transformed
        # domain (W @ T^-1 == W @ T^T for orthogonal T), so apply the same data-free transform
        # the activation quantizer will use.
        assert transform_class not in ("wush", "wus"), \
            "wush/wus transforms are computed from the GPTQ basis and need weight quantization (w_bits < 16)."
        n_blocks: int = weight.size(-1) // block_size
        transform_basis: torch.Tensor = get_transform(
            basis=torch.eye(block_size, dtype=torch.float64, device=device).expand(n_blocks, block_size, block_size),
            y=None,
            transform_type=transform_class,
            transform_dtype=dtype,
        ).expand(n_blocks, block_size, block_size).to(dtype=dtype)  # (C//B, B, B)
        weight_transformed_dequantized: torch.Tensor = block_transform(weight, transform=transform_basis)  # (R, C)
        weight_global_scale = None
        print(f"Not quantizing weights, transform {transform_class}, block size {block_size}")
    else:
        match quant_type:
            case 'mxfp4':
                global_scale: torch.Tensor = torch.tensor(1.)
            case 'nvfp4':
                number = _grids_max_repr(grids_w)
                global_scale = ((number * 448) / (weight.abs().max())).to(torch.float32)

        print("Quantizing scales with ", scale_quant_type, "global scale", global_scale, f"block size {block_size}")

        quant_fn = functools.partial(
            triton_kernel_wrapper,
            grids=grids_w,
            gs=block_size,
            scale_quant=scale_quant_type,
            global_scale=global_scale,
        )

        weight_quant_result: dict = gptq_babai_outer(
            parameter=weight,
            hessian=hessian,
            inner_fn=functools.partial(
                block_transform_gptq_babai_inner,
                quant_fn=quant_fn,
                basis_transform_fn=functools.partial(
                    get_transform,
                    transform_type=transform_class,
                    transform_dtype=dtype,
                ),
                force_rtn_inner=force_rtn,  # use gptq
            ),
            collate_fn=functools.partial(collate_xxfp4, dtype=dtype),
            block_size=block_size,
            force_rtn_outer=force_rtn,  # use gptq
        )
        weight_transformed_dequantized: torch.Tensor = weight_quant_result['fake_quant']  # (R, C), transformed domain
        transform_basis: torch.Tensor = weight_quant_result['transform_basis']  # (C//B, B, B)
        weight_global_scale = weight_quant_result['global_scale']

        # Measure the error in the original domain: W_hat = W_q @ T (inverse of W @ T^-1).
        weight_reconstructed: torch.Tensor = block_transform(weight_transformed_dequantized, transform=transform_basis.transpose(-2, -1))
        mse = ((weight_reconstructed.float() - weight.float()) ** 2).mean()
        relative_mse = mse / (weight.float() ** 2).mean()
        print("MSE: ", mse.item(), " Relative MSE:", relative_mse.item())
        if args.log_wandb:
            wandb.log({"calibration/weight_mse": mse.item()})
            wandb.log({"calibration/weight_rmse": relative_mse.item()})

    new_weights: list[torch.Tensor] = weight_transformed_dequantized.split([w.size(0) for w in weights], dim=0)
    quant_linear_layers: list[nn.Module] = [QuantLinear(
        weight=new_w,
        activation_quantizer=BlockTransformQuantizerGrid(
            transform=transform_basis,
            grids=grids_a,
            group_size=block_size,
            global_scale=None if quant_type == 'nvfp4' else torch.tensor(1.),
            scale_precision=scale_dtype,
        ),
    ).to(device=device) for new_w in new_weights]

    quant_results = [{'transform_basis': transform_basis, 'qweight': None, 'scale_quant': None, 'global_scale': weight_global_scale} for _ in range(len(weights))]

    return quant_linear_layers, quant_results


class Hessian_layer:
    def __init__(
        self,
        layer: nn.Module,
    ):
        assert isinstance(layer, nn.Linear)
        self.layer = layer
        self.hessian = None
        self.num_samples: int = 0
        self.hessian_sum = None

    # preparatory methods
    @torch.no_grad()
    def update(self, inp: torch.Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            inp: batch of layer inputs
        """
        inp = inp.flatten(end_dim=-2)
        if self.hessian_sum is None:
            self.hessian_sum = torch.zeros(inp.size(-1), inp.size(-1), dtype=torch.float32, device=inp.device)
        batch_size: int = inp.size(0)
        # self.hessian *= self.num_samples / (self.num_samples + batch_size)
        accumulate_hessian(self.hessian_sum, inp.to(dtype=torch.float32))  # X^T X
        self.num_samples += batch_size

    def get_hessian(self) -> torch.Tensor:
        assert self.num_samples > 0, "No samples have been accumulated for Hessian estimation."
        if self.hessian is None:
            self.hessian = self.hessian_sum/self.num_samples
        return self.hessian

def grid_quantization_gptq(
    model: PreTrainedModel,
    calibration_data: list[torch.Tensor],
    args: argparse.Namespace, 
    device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("grid quantization started...")
    n_w = 0 if args.grids_w is None else len(args.grids_w)
    n_a = 0 if args.grids_a is None else len(args.grids_a)
    print(f"Using {n_w} weight grid(s) and {n_a} activation grid(s)")
    quantized_state_dict = {}
    non_quantized_state_dict = {}
    act_offload_device = "cpu" if args.cpu_offload_activations else device

    blocks = model.model.layers
    blocks[0] = InputCollector(blocks[0], cpu_offload=args.cpu_offload_activations)
    if args.cpu_offload_modules:
        model.get_input_embeddings().to(device)
        blocks[0] = blocks[0].to(device)

    for sample in calibration_data:
        try:
            with torch.no_grad():
                model(sample.to(device=device))
        except ForwardInterrupt:
            pass

    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    if args.cpu_offload_modules:
        model.get_input_embeddings().cpu()

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")
        if args.cpu_offload_modules:
            block.to(device)
        # 4. Create WUSH handles and hooks

        use_fused_layers: bool = False
        
        print("All nn.Linear layers:\n")
        layer_sequences=[]
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_sequences.append(name)
        layer_sequences = [layer_sequences]

        print(layer_sequences)
        if not use_fused_layers:
            layer_sequences: list[list[str]] = [[v] for v in itertools.chain.from_iterable(layer_sequences)]  # for non-vllm
            print("Layer sequences:", layer_sequences)
        else:
            raise NotImplementedError
        
        _handles = {}
        hooks = {}
        for layer_name, *_ in layer_sequences:
            layer: nn.Module = block.get_submodule(layer_name)
            _handles[layer_name] = Hessian_layer(layer)
            def update_handle_hook(name):
                def _hook(_, inp, __):
                    _handles[name].update(inp[0])
                return _hook
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

        # 5. Process calibration data
        output_ref = []
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad():
                out_ref = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out_ref = maybe_first_element(out_ref).to(act_offload_device)
            output_ref.append(out_ref)
        # Remove hooks
        for hook in hooks.values():
            hook.remove()

        configs: list[tuple] = list(itertools.product(
            [('force_rtn', v) for v in [not args.gptq]],
            [('transform_class', v) for v in [args.transform_class]],
        ))
        print(configs)
        quant_layers: dict[tuple, dict[str, nn.Module]] = {config: {} for config in configs}
        real_quant_results: dict[tuple, dict[str, nn.Module]] = {config: {} for config in configs}

        for layer_sequence in layer_sequences:
            hessian = _handles[layer_sequence[0]].get_hessian()
            # print(f"Hessian for layer {layer_sequence[0]}:", hessian)
            weights = [block.get_submodule(layer_name).weight for layer_name in layer_sequence]
            for config in configs:
                qlinear_layers, quant_results = quantize_layers(*weights, hessian=hessian, quant_type=args.format + "4", grids_w=args.grids_w, grids_a=args.grids_a, args=args, **dict(config))
                quant_layers[config].update({layer_name: qlinear_layer for layer_name, qlinear_layer in zip(layer_sequence, qlinear_layers)})
                real_quant_results[config].update({layer_name: quant_result for layer_name, quant_result in zip(layer_sequence, quant_results)})

        for config in configs:
            for layer_name, qlinear_layer in quant_layers[config].items():
                block.set_submodule(layer_name, qlinear_layer)
            for inp_args, inp_kwargs, out_ref in zip(input_args, input_kwargs, output_ref):
                with torch.no_grad():
                    out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = maybe_first_element(out)
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QuantLinear):
                    with torch.no_grad():
                        layer.activation_quantizer._track_global_scale = False
                        print(layer_name, f" global scale { layer.activation_quantizer.global_scale}")

        l2_losses: dict[tuple, float] = {}
        relative_losses: dict[tuple, float] = {}
        for config in configs:
            for layer_name, qlinear_layer in quant_layers[config].items():
                block.set_submodule(layer_name, qlinear_layer)
            _losses = []
            _relative_losses = []
            for inp_args, inp_kwargs, out_ref in zip(input_args, input_kwargs, output_ref):
                with torch.no_grad():
                    out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = maybe_first_element(out)
                _losses.append(torch.linalg.vector_norm(out - out_ref.to(out.device), dtype=torch.float32).pow(2.) / out_ref.numel())
                _relative_losses.append(_losses[-1]/(out_ref**2).mean())
            l2_loss: float = torch.stack(_losses).mean().item()
            relative_l2_loss: float = torch.stack(_relative_losses).mean().item()
            l2_losses[config] = l2_loss
            relative_losses[config] = relative_l2_loss
            print('l2_loss', config, l2_loss)
            
        # find the min loss config
        best_config = min(l2_losses, key=l2_losses.get)
        best_quant_layers = quant_layers[best_config]
        print('best_config', best_config, 'l2_loss', l2_losses[best_config])
        if args.log_wandb:
                wandb.log({"calibration/activation_mse": l2_losses[best_config]})
                wandb.log({"calibration/relative_mse": relative_losses[best_config]})


        for layer_name, qlinear_layer in best_quant_layers.items():
            block.set_submodule(layer_name, qlinear_layer)
        # 8. Update activations
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad():
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out = maybe_first_element(out).to(act_offload_device)
            # change only first input argument
            if len(inp_args) > 0:
                inp_args[0].data = out
            elif "hidden_states" in inp_kwargs:
                inp_kwargs["hidden_states"] = out
            else:
                raise ValueError("Unsupported block input format.")
        
        if args.cpu_offload_modules:
            block.cpu()

        # 8. Clean-up
        del _handles
        del hooks
        clear_device_cache(garbage_collection=True)
        

    clear_device_cache(garbage_collection=True)
    print(model)
    return quantized_state_dict, non_quantized_state_dict