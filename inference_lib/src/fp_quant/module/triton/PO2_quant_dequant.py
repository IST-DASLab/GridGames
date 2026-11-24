import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

# All block sizes must be divisible by GROUP_SIZE.
_BLOCK_SIZES = [32 * 16, 64 * 16, 128 * 16, 256 * 16, 512 * 16]

# Multipliers searched around the absmax-derived scale. Powers of two only, so
# E8M0 scales stay exactly representable. 1.0 comes first: ties keep the
# earliest candidate, which makes the plain absmax scale the default winner.
_DEFAULT_SCALE_CANDIDATES = (1.0, 0.5, 0.25, 0.125, 2.0, 4.0, 8.0)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _quantize_scale_e8m0_from_absmax(absmax, grid_max):
    """
    Power-of-two shared scale covering absmax for grids with symmetric endpoints.
    """
    absmax = tl.where(absmax < 1e-30, 1e-30, absmax)
    scale = tl.exp2(tl.ceil(tl.log2(absmax / grid_max)))
    # Guard against log2 rounding down just above a power-of-two boundary.
    return tl.where(scale * grid_max < absmax, scale * 2.0, scale)


def _check_group_size(gs):
    if not isinstance(gs, int):
        raise TypeError(f"gs must be an int, got {type(gs).__name__}")
    if gs <= 0:
        raise ValueError(f"gs must be positive, got {gs}")

    bad = [bs for bs in _BLOCK_SIZES if bs % gs != 0]
    if bad:
        raise ValueError(
            f"gs={gs} must divide all Triton BLOCK_SIZE configs, but failed for {bad}"
        )


def _validate_x_for_triton(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor for Triton kernels")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating-point, got {x.dtype}")


def _prepare_grid(grid, device):
    grid = torch.as_tensor(grid, dtype=torch.float32, device=device).flatten()

    if grid.numel() == 0:
        raise ValueError("Grid must contain at least one value")
    if not torch.isfinite(grid).all():
        raise ValueError("Grid must contain only finite values")

    grid = grid.sort()[0].contiguous()
    thresholds = ((grid[:-1] + grid[1:]) / 2).contiguous()

    grid_max = grid.abs().max().item()
    if grid_max <= 0:
        raise ValueError("Grid must contain at least one non-zero value")

    return grid, thresholds, grid_max

def _prepare_grids_N(grids, device):
    """
    Prepare a list/tuple of N candidate grids.

    Returns:
        grids_stacked:      (N_TOTAL_GRIDS, N_GRID)     sorted grid values, float32
        thresholds_stacked: (N_TOTAL_GRIDS, N_GRID - 1) midpoint thresholds,  float32
        grid_maxes:         (N_TOTAL_GRIDS,)            max(abs(grid)) per grid, float32
        n_grid:             int, number of levels (same across all grids)
    """
    if not isinstance(grids, (list, tuple)):
        raise TypeError(f"grids must be a list or tuple, got {type(grids).__name__}")
    if len(grids) == 0:
        raise ValueError("grids must contain at least one grid")

    prepared = [_prepare_grid(g, device) for g in grids]
    n_grid = prepared[0][0].numel()
    for i, (g, _, _) in enumerate(prepared):
        if g.numel() != n_grid:
            raise ValueError(
                f"All grids must have the same number of levels, "
                f"got {n_grid} for grid 0 and {g.numel()} for grid {i}"
            )

    grids_stacked = torch.stack([p[0] for p in prepared], dim=0).contiguous()
    thresholds_stacked = torch.stack([p[1] for p in prepared], dim=0).contiguous()
    grid_maxes = torch.tensor(
        [p[2] for p in prepared], dtype=torch.float32, device=device
    )

    return grids_stacked, thresholds_stacked, grid_maxes, n_grid



def _parse_scale_quant(scale_quant, global_scale):
    """
    False / None  -> disabled
    True          -> E4M3 scale quantization, backward-compatible alias
    "e4m3"        -> E4M3 scale quantization
    "e8m0"        -> power-of-two scale rounded up to cover the group maximum
    """
    if scale_quant in (False, None):
        return False, False

    if scale_quant == "e8m0":
        return True, True

    if scale_quant is True or scale_quant == "e4m3":
        if global_scale is None:
            raise ValueError(
                "global_scale must be provided when scale_quant is True or 'e4m3'. "
                "Use a scalar like torch.tensor(448.0), or use scale_quant='e8m0'."
            )
        return True, False

    raise ValueError(
        f"Unsupported scale_quant={scale_quant!r}. "
        "Use False, True, 'e4m3', or 'e8m0'."
    )


def _prepare_global_scale(global_scale, device):
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=device)
    if gs.numel() != 1:
        raise ValueError(f"global_scale must be a scalar, got shape {tuple(gs.shape)}")
    if not torch.isfinite(gs).all() or gs.item() <= 0:
        raise ValueError(f"global_scale must be finite and positive, got {gs.item()}")
    return gs.reshape(1).contiguous()


def _pick_code_dtype(n_grid):
    if n_grid <= 256:
        return torch.uint8
    if n_grid <= 32768:
        return torch.int16
    raise ValueError(f"Grid too large for supported code dtypes: {n_grid} levels")


# ═════════════════════════════════════════════════════════════════════════════
#  N-GRID QUANTIZE / DEQUANTIZE KERNELS
# ═════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in _BLOCK_SIZES],
    key=[],
)
@triton.jit
def _N_grid_quantize_to_codes_kernel(
    x_ptr,
    grids_ptr,            # (N_TOTAL_GRIDS, N_GRID)   stacked sorted grid values
    thresholds_ptr,       # (N_TOTAL_GRIDS, N_GRID-1) stacked midpoint thresholds
    grid_maxes_ptr,       # (N_TOTAL_GRIDS,)          max(abs(grid)) per grid row
    scale_mults_ptr,      # (N_SCALE_CAND,)           multipliers applied to the base scale
    global_scale_ptr,
    q_ptr,
    scales_ptr,
    grid_ids_ptr,
    n_elements,
    n_groups_total,
    GROUP_SIZE: tl.constexpr,
    N_GRID: tl.constexpr,
    N_TOTAL_GRIDS: tl.constexpr,
    N_SCALE_CAND: tl.constexpr,
    SCALE_QUANT: tl.constexpr,
    E8M0_SCALE: tl.constexpr,
    EXTERNAL_SCALES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE

    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    n_groups: tl.constexpr = BLOCK_SIZE // GROUP_SIZE
    x_g = tl.reshape(x, (n_groups, GROUP_SIZE))

    group_offsets = (start // GROUP_SIZE) + tl.arange(0, n_groups)
    group_mask = group_offsets < n_groups_total

    if EXTERNAL_SCALES:
        # Caller has already chosen scale and grid per group; just assign codes
        # using the thresholds of the chosen grid for each group.
        ext_scale = tl.load(scales_ptr + group_offsets, mask=group_mask, other=1.0)
        ext_gid = tl.load(grid_ids_ptr + group_offsets, mask=group_mask, other=0).to(tl.int32)

        scale_sel = tl.reshape(ext_scale, (n_groups, 1))
        ext_gid_g = tl.reshape(ext_gid, (n_groups, 1))

        xn = tl.reshape(x_g / scale_sel, (BLOCK_SIZE,))

        q_sel = tl.zeros((n_groups, GROUP_SIZE), dtype=tl.int32)
        for g_idx in range(N_TOTAL_GRIDS):
            idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            thresh_base = g_idx * (N_GRID - 1)
            for i in range(N_GRID - 1):
                t = tl.load(thresholds_ptr + thresh_base + i)
                idx += (xn > t).to(tl.int32)
            idx_g = tl.reshape(idx, (n_groups, GROUP_SIZE))
            q_sel = tl.where(ext_gid_g == g_idx, idx_g, q_sel)

        q_sel_flat = tl.reshape(q_sel, (BLOCK_SIZE,))
        tl.store(q_ptr + offsets, q_sel_flat, mask=mask)
        return

    vec_max = tl.max(tl.abs(x_g), axis=1, keep_dims=True)

    if SCALE_QUANT and not E8M0_SCALE:
        gs = tl.load(global_scale_ptr)

    # Joint search per group over (candidate grid, scale multiplier): scale and
    # grid are both stored per group here, so every (grid, mult) pair is a valid
    # combination and the argmin is taken over all of them.
    best_mse = tl.full((n_groups, 1), float("inf"), dtype=tl.float32)
    best_q = tl.zeros((n_groups, GROUP_SIZE), dtype=tl.int32)
    best_scale = tl.zeros((n_groups, 1), dtype=tl.float32)
    best_gid = tl.zeros((n_groups, 1), dtype=tl.int32)

    for g_idx in tl.static_range(N_TOTAL_GRIDS):
        g_max = tl.load(grid_maxes_ptr + g_idx)

        for s_idx in tl.static_range(N_SCALE_CAND):
            # scaling vec_max (rather than the final scale) keeps the quantized
            # scale paths exact: every multiplier is a power of two.
            vec_max_s = vec_max * tl.load(scale_mults_ptr + s_idx)

            s = vec_max_s / g_max
            s = tl.where(s < 1e-10, 1e-10, s)

            if SCALE_QUANT:
                if E8M0_SCALE:
                    s = _quantize_scale_e8m0_from_absmax(vec_max_s, g_max)
                else:
                    s = (s * gs).to(tl.float32).to(tl.float8e4nv).to(tl.float32) / gs
                    s = tl.where(s < 2e-3, 2e-3, s)

            xn = tl.reshape(x_g / s, (BLOCK_SIZE,))

            idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            thresh_base = g_idx * (N_GRID - 1)
            for i in range(N_GRID - 1):
                t = tl.load(thresholds_ptr + thresh_base + i)
                idx += (xn > t).to(tl.int32)

            grid_base = g_idx * N_GRID
            q_vals = tl.reshape(tl.load(grids_ptr + grid_base + idx), (n_groups, GROUP_SIZE))
            dq = q_vals * s
            err = x_g - dq
            mse = tl.sum(err * err, axis=1, keep_dims=True)

            is_better = mse < best_mse
            idx_g = tl.reshape(idx, (n_groups, GROUP_SIZE))
            best_q = tl.where(is_better, idx_g, best_q)
            best_scale = tl.where(is_better, s, best_scale)
            best_mse = tl.where(is_better, mse, best_mse)
            best_gid = tl.where(is_better, g_idx, best_gid)

    q_sel_flat = tl.reshape(best_q, (BLOCK_SIZE,))
    tl.store(q_ptr + offsets, q_sel_flat, mask=mask)

    scale_1d = tl.reshape(best_scale, (n_groups,))
    grid_id_1d = tl.reshape(best_gid, (n_groups,))

    tl.store(scales_ptr + group_offsets, scale_1d, mask=group_mask)
    tl.store(grid_ids_ptr + group_offsets, grid_id_1d, mask=group_mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in _BLOCK_SIZES],
    key=[],
)
@triton.jit
def _N_grid_dequantize_from_codes_kernel(
    q_ptr,
    scales_ptr,
    grid_ids_ptr,
    grids_ptr,
    out_ptr,
    n_elements,
    GROUP_SIZE: tl.constexpr,
    N_GRID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE

    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    q = tl.load(q_ptr + offsets, mask=mask, other=0).to(tl.int32)

    group_offsets = offsets // GROUP_SIZE
    scales = tl.load(scales_ptr + group_offsets, mask=mask, other=1.0)
    grid_ids = tl.load(grid_ids_ptr + group_offsets, mask=mask, other=0).to(tl.int32)

    grid_offset = grid_ids * N_GRID + q
    vals = tl.load(grids_ptr + grid_offset, mask=mask, other=0.0)
    out = vals * scales

    tl.store(out_ptr + offsets, out, mask=mask)



@torch.compiler.disable()
def quantize_N_triton(
    x,
    grids,
    gs=16,
    scale_quant=False,
    global_scale=None,
    scales=None,
    grid_ids=None,
    scale_candidates=_DEFAULT_SCALE_CANDIDATES,
):
    """
    Quantize x using N candidate grids and choose the best grid per group by MSE.

    `grids` is a list or tuple of grids; all grids must have the same number of
    levels. The kernel evaluates every grid for each group and selects the one
    with the lowest per-group MSE.

    The scale is searched jointly with the grid: each (grid, multiplier) pair is
    evaluated, where the multipliers come from `scale_candidates` and scale the
    absmax-derived base scale (default 1, 1/2, 1/4, 1/8, 2, 4, 8). Ties keep the
    earlier candidate, so leading the list with 1.0 makes the plain absmax scale
    win on ties.

    E8M0 rounds each candidate's scale up to cover the scaled absmax. Multipliers
    below 1 can still clip the original input; use (1.0,) to disable that search.

    If both `scales` and `grid_ids` are provided, they are used as fixed
    per-group scale + grid selector and the kernel skips MSE comparison.
    `scale_quant` / `global_scale` / `scale_candidates` are ignored in that case.
    `grid_ids` must hold integer ids in [0, len(grids)).

    Returns:
        q:         integer codes, shape [..., n_full * gs]
        scales:    per-group scales, shape [..., n_full]
        grid_ids:  per-group grid selector (uint8), shape [..., n_full]
        tail:      unquantized tail if x.shape[-1] is not divisible by gs, else None
    """
    _validate_x_for_triton(x)
    _check_group_size(gs)

    *leading, dim = x.shape
    n_full = dim // gs
    rem = dim - n_full * gs

    tail = x[..., n_full * gs:].contiguous() if rem > 0 else None

    if (scales is None) != (grid_ids is None):
        raise ValueError(
            "scales and grid_ids must both be provided together, or both be None"
        )
    use_external_scales = scales is not None

    if use_external_scales:
        expected_group_shape = tuple(leading) + (n_full,)
        for name, tensor, expected_dtype in (
            ("scales", scales, torch.float32),
            ("grid_ids", grid_ids, torch.uint8),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if tuple(tensor.shape) != expected_group_shape:
                raise ValueError(
                    f"{name} must have shape {expected_group_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != expected_dtype:
                raise TypeError(f"{name} must be {expected_dtype}, got {tensor.dtype}")
            if tensor.device != x.device:
                raise ValueError(f"{name} must be on {x.device}, got {tensor.device}")

    grids_stacked, thresholds_stacked, grid_maxes, n_grid = _prepare_grids_N(grids, x.device)
    n_total_grids = grids_stacked.shape[0]
    q_dtype = _pick_code_dtype(n_grid)

    scale_mults = torch.as_tensor(
        tuple(scale_candidates), dtype=torch.float32, device=x.device
    ).contiguous()
    if scale_mults.ndim != 1 or scale_mults.numel() == 0:
        raise ValueError(
            "scale_candidates must be a non-empty 1-D sequence, got shape "
            f"{tuple(scale_mults.shape)}"
        )
    n_scale_cand = scale_mults.numel()

    if use_external_scales:
        # grid_ids must be in range for the kernel's gather to be valid.
        if int(grid_ids.max().item()) >= n_total_grids:
            raise ValueError(
                f"grid_ids contains an id >= n_total_grids ({n_total_grids})"
            )

    if n_full == 0:
        q = torch.empty((*leading, 0), device=x.device, dtype=q_dtype)
        out_scales = scales if use_external_scales else torch.empty(
            (*leading, 0), device=x.device, dtype=torch.float32
        )
        out_grid_ids = grid_ids if use_external_scales else torch.empty(
            (*leading, 0), device=x.device, dtype=torch.uint8
        )
        return q, out_scales, out_grid_ids, tail

    if use_external_scales:
        use_scale_quant = False
        use_e8m0 = False
        gs_t = torch.zeros(1, dtype=torch.float32, device=x.device)
    else:
        use_scale_quant, use_e8m0 = _parse_scale_quant(scale_quant, global_scale)

        if use_scale_quant and not use_e8m0:
            gs_t = _prepare_global_scale(global_scale, x.device)
        else:
            gs_t = torch.zeros(1, dtype=torch.float32, device=x.device)

    x_flat = x[..., : n_full * gs].reshape(-1).contiguous()
    n = x_flat.numel()
    n_groups_total = n // gs

    q_flat = torch.empty((n,), device=x.device, dtype=q_dtype)

    if use_external_scales:
        scales_flat = scales.reshape(-1).contiguous()
        grid_ids_flat = grid_ids.reshape(-1).contiguous()
    else:
        scales_flat = torch.empty((n_groups_total,), device=x.device, dtype=torch.float32)
        grid_ids_flat = torch.empty((n_groups_total,), device=x.device, dtype=torch.uint8)

    if n == 0:
        q = q_flat.reshape(*leading, n_full * gs)
        out_scales = scales if use_external_scales else scales_flat.reshape(*leading, n_full)
        out_grid_ids = grid_ids if use_external_scales else grid_ids_flat.reshape(*leading, n_full)
        return q, out_scales, out_grid_ids, tail

    launch_grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    if not use_e8m0 and len(scale_mults)>1 and not use_external_scales:
        scale_mults = torch.tensor(
        [
            1.0,
            0.125, 0.140625, 0.15625, 0.171875,
            0.1875, 0.203125, 0.21875, 0.234375,

            0.25, 0.28125, 0.3125, 0.34375,
            0.375, 0.40625, 0.4375, 0.46875,

            0.5, 0.5625, 0.625, 0.6875,
            0.75, 0.8125, 0.875, 0.9375,

            1.125, 1.25, 1.375, 1.5,
            1.625, 1.75, 1.875,

            2.0, 2.25, 2.5, 2.75,
            3.0, 3.25, 3.5, 3.75,

            4.0, 4.5, 5.0, 5.5,
            6.0, 6.5, 7.0, 7.5,

            8.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    n_scale_cand = scale_mults.numel()
    
        
    _N_grid_quantize_to_codes_kernel[launch_grid](
        x_flat,
        grids_stacked,
        thresholds_stacked,
        grid_maxes,
        scale_mults,
        gs_t,
        q_flat,
        scales_flat,
        grid_ids_flat,
        n,
        n_groups_total,
        GROUP_SIZE=gs,
        N_GRID=n_grid,
        N_TOTAL_GRIDS=n_total_grids,
        N_SCALE_CAND=n_scale_cand,
        SCALE_QUANT=use_scale_quant,
        E8M0_SCALE=use_e8m0,
        EXTERNAL_SCALES=use_external_scales,
    )

    q = q_flat.reshape(*leading, n_full * gs)
    out_scales = scales if use_external_scales else scales_flat.reshape(*leading, n_full)
    out_grid_ids = grid_ids if use_external_scales else grid_ids_flat.reshape(*leading, n_full)

    return q, out_scales, out_grid_ids, tail


@torch.compiler.disable()
def dequantize_N_triton(
    q,
    scales,
    grid_ids,
    grids,
    gs=16,
    tail=None,
    dtype=torch.float32,
):
    """
    Dequantize codes produced by quantize_N_triton.
    """
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"q must be a torch.Tensor, got {type(q).__name__}")
    if q.ndim == 0:
        raise ValueError("q must have at least one dimension")
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")

    _check_group_size(gs)

    *leading, dim = q.shape

    if dim % gs != 0:
        raise ValueError(f"q.shape[-1] must be divisible by gs={gs}, got {dim}")

    n_full = dim // gs
    expected_group_shape = tuple(leading) + (n_full,)

    if tuple(scales.shape) != expected_group_shape:
        raise ValueError(
            f"scales must have shape {expected_group_shape}, got {tuple(scales.shape)}"
        )
    if tuple(grid_ids.shape) != expected_group_shape:
        raise ValueError(
            f"grid_ids must have shape {expected_group_shape}, got {tuple(grid_ids.shape)}"
        )

    if not scales.is_cuda:
        raise ValueError("scales must be a CUDA tensor")
    if not grid_ids.is_cuda:
        raise ValueError("grid_ids must be a CUDA tensor")

    grids_stacked, _, _, n_grid = _prepare_grids_N(grids, q.device)

    q_flat = q.reshape(-1).contiguous()
    scales_flat = scales.reshape(-1).contiguous()
    grid_ids_flat = grid_ids.reshape(-1).contiguous()

    n = q_flat.numel()
    out_flat = torch.empty((n,), device=q.device, dtype=dtype)

    if n > 0:
        launch_grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        _N_grid_dequantize_from_codes_kernel[launch_grid](
            q_flat,
            scales_flat,
            grid_ids_flat,
            grids_stacked,
            out_flat,
            n,
            GROUP_SIZE=gs,
            N_GRID=n_grid,
        )

    out = out_flat.reshape(*leading, dim)

    if tail is not None:
        if tuple(tail.shape[:-1]) != tuple(leading):
            raise ValueError(
                f"tail leading shape must be {tuple(leading)}, got {tuple(tail.shape[:-1])}"
            )
        out = torch.cat([out, tail.to(device=out.device, dtype=out.dtype)], dim=-1)

    return out


@torch.compiler.disable()
def fake_quant_dequant_N_triton(
    x,
    grids,
    gs=16,
    scale_quant=False,
    global_scale=None,
    scales=None,
    grid_ids=None,
    scale_candidates=_DEFAULT_SCALE_CANDIDATES,
):
    """
    Backward-compatible N-grid fake-quant wrapper.

    Quantizes to explicit codes/scales/grid_ids, then immediately dequantizes.

    If both `scales` and `grid_ids` are provided, they are used as fixed
    per-group scale + grid selector instead of being chosen by the kernel.
    """
    q, scales_out, grid_ids_out, tail = quantize_N_triton(
        x,
        grids,
        gs=gs,
        scale_quant=scale_quant,
        global_scale=global_scale,
        scales=scales,
        grid_ids=grid_ids,
        scale_candidates=scale_candidates,
    )

    return dequantize_N_triton(
        q,
        scales_out,
        grid_ids_out,
        grids,
        gs=gs,
        tail=tail,
        dtype=x.dtype,
    )


__all__ = [
    "quantize_N_triton",
    "dequantize_N_triton",
    "fake_quant_dequant_N_triton",
]
