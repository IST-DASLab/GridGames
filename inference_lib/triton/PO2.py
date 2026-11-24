import torch
import inference_lib.triton as triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_BLOCK_SIZES = [ 128 * 16, 256 * 16, 512 * 16]


@triton.jit
def _quantize_scale_e8m0_from_absmax(absmax, grid_max):
    """
    Quantize scale with ceil
    """
    ratio = tl.where(absmax > 0, absmax / grid_max, 1.0)
    scale = tl.exp2(tl.ceil(tl.log2(ratio)))
    # Guard against log2 rounding down just above a power-of-two boundary.
    scale = tl.where(scale * grid_max < absmax, scale * 2.0, scale)
    return scale


def _check_group_size(gs):
    if gs <= 0:
        raise ValueError(f"gs must be positive, got {gs}")
    bad = [bs for bs in _BLOCK_SIZES if bs % gs != 0]
    if bad:
        raise ValueError(
            f"gs={gs} must divide all Triton BLOCK_SIZE configs, but failed for {bad}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SINGLE GRID QUANTIZATION KERNEL
# ═════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in _BLOCK_SIZES],
    key=[],
)
@triton.jit
def _grid_quantize_kernel(
    x_ptr,
    grid_ptr,
    thresholds_ptr,
    global_scale_ptr,
    output_ptr,
    grid_max,
    n_elements,
    GROUP_SIZE: tl.constexpr,
    N_GRID: tl.constexpr,
    SCALE_QUANT: tl.constexpr,
    E8M0_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    n_groups: tl.constexpr = BLOCK_SIZE // GROUP_SIZE
    x_g = tl.reshape(x, (n_groups, GROUP_SIZE))

    vec_max = tl.max(tl.abs(x_g), axis=1, keep_dims=True)
    scale = vec_max / grid_max
    scale = tl.where(scale < 1e-10, 1e-10, scale)

    if SCALE_QUANT:
        if E8M0_SCALE:
            scale = _quantize_scale_e8m0_from_absmax(vec_max, grid_max)
        else:
            gs = tl.load(global_scale_ptr)
            scale = (scale * gs).to(tl.float8e4nv).to(tl.float32) / gs
            scale = tl.where(scale < 2e-3, 2e-3, scale)

    x_n = x_g / scale

    x_flat = tl.reshape(x_n, (BLOCK_SIZE,))
    idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in range(N_GRID - 1):
        t = tl.load(thresholds_ptr + i)
        idx += (x_flat > t).to(tl.int32)

    x_q = tl.reshape(tl.load(grid_ptr + idx), (n_groups, GROUP_SIZE))
    x_dq = tl.reshape(x_q * scale, (BLOCK_SIZE,))

    tl.store(output_ptr + offsets, x_dq, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
#  DUAL GRID QUANTIZATION KERNEL
# ═════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in _BLOCK_SIZES],
    key=[],
)
@triton.jit
def _dual_grid_quantize_kernel(
    x_ptr,
    grid_a_ptr, thresholds_a_ptr,
    grid_b_ptr, thresholds_b_ptr,
    global_scale_ptr,
    output_ptr,
    grid_a_max, grid_b_max,
    n_elements,
    GROUP_SIZE: tl.constexpr,
    N_GRID: tl.constexpr,
    SCALE_QUANT: tl.constexpr,
    E8M0_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    n_groups: tl.constexpr = BLOCK_SIZE // GROUP_SIZE
    x_g = tl.reshape(x, (n_groups, GROUP_SIZE))
    vec_max = tl.max(tl.abs(x_g), axis=1, keep_dims=True)

    # ── Grid A ──────────────────────────────────────────────────────────
    s_a = vec_max / grid_a_max
    s_a = tl.where(s_a < 1e-10, 1e-10, s_a)
    if SCALE_QUANT:
        if E8M0_SCALE:
            s_a = _quantize_scale_e8m0_from_absmax(vec_max, grid_a_max)
        else:
            gs = tl.load(global_scale_ptr)
            s_a = (s_a * gs).to(tl.float8e4nv).to(tl.float32) / gs
            s_a = tl.where(s_a < 2e-3, 2e-3, s_a)

    xn_a = tl.reshape(x_g / s_a, (BLOCK_SIZE,))
    i_a = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in range(N_GRID - 1):
        i_a += (xn_a > tl.load(thresholds_a_ptr + i)).to(tl.int32)

    q_a = tl.reshape(tl.load(grid_a_ptr + i_a), (n_groups, GROUP_SIZE))
    dq_a = q_a * s_a
    err_a = x_g - dq_a
    mse_a = tl.sum(err_a * err_a, axis=1, keep_dims=True)

    # ── Grid B ──────────────────────────────────────────────────────────
    s_b = vec_max / grid_b_max
    s_b = tl.where(s_b < 1e-10, 1e-10, s_b)
    if SCALE_QUANT:
        if E8M0_SCALE:
            s_b = _quantize_scale_e8m0_from_absmax(vec_max, grid_b_max)
        else:
            gs = tl.load(global_scale_ptr)
            s_b = (s_b * gs).to(tl.float8e4nv).to(tl.float32) / gs
            s_b = tl.where(s_b < 2e-3, 2e-3, s_b)

    xn_b = tl.reshape(x_g / s_b, (BLOCK_SIZE,))
    i_b = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in range(N_GRID - 1):
        i_b += (xn_b > tl.load(thresholds_b_ptr + i)).to(tl.int32)

    q_b = tl.reshape(tl.load(grid_b_ptr + i_b), (n_groups, GROUP_SIZE))
    dq_b = q_b * s_b
    err_b = x_g - dq_b
    mse_b = tl.sum(err_b * err_b, axis=1, keep_dims=True)

    x_dq = tl.where(mse_a <= mse_b, dq_a, dq_b)
    tl.store(output_ptr + offsets, tl.reshape(x_dq, (BLOCK_SIZE,)), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
#  PYTHON WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════

def _prepare_grid(grid, device):
    if not isinstance(grid, torch.Tensor):
        grid = torch.tensor(grid, dtype=torch.float32)
    grid = grid.float().sort()[0].to(device).contiguous()
    thresholds = ((grid[:-1] + grid[1:]) / 2).contiguous()
    grid_max = grid.abs().max().item()
    if grid_max <= 0:
        raise ValueError("Grid must contain at least one non-zero value")
    return grid, thresholds, grid_max


def _parse_scale_quant(scale_quant, global_scale):
    """
    False / None  -> disabled
    True          -> E4M3 (backward compatible)
    "e4m3"        -> E4M3
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
    gs = torch.as_tensor(global_scale, dtype=torch.float32)
    if gs.numel() != 1:
        raise ValueError(f"global_scale must be a scalar, got shape {tuple(gs.shape)}")
    return gs.reshape(1).to(device).contiguous()


@torch.compiler.disable()
def fake_quant_grid_triton(x, grid, gs=16, scale_quant=False, global_scale=None):
    _check_group_size(gs)

    *leading, dim = x.shape
    n_full = dim // gs
    rem = dim - n_full * gs
    if n_full == 0:
        return x

    g, t, gmax = _prepare_grid(grid, x.device)
    use_scale_quant, use_e8m0 = _parse_scale_quant(scale_quant, global_scale)

    x_flat = x[..., : n_full * gs].reshape(-1).contiguous()
    n = x_flat.numel()
    out = torch.empty_like(x_flat)

    # Always pass a valid pointer.
    if use_scale_quant and not use_e8m0:
        gs_t = _prepare_global_scale(global_scale, x.device)
    else:
        gs_t = torch.zeros(1, dtype=torch.float32, device=x.device)
    launch_grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    with torch.device(x.device):
        _grid_quantize_kernel[launch_grid](
            x_flat,
            g,
            t,
            gs_t,
            out,
            gmax,
            n,
            GROUP_SIZE=gs,
            N_GRID=len(g),
            SCALE_QUANT=use_scale_quant,
            E8M0_SCALE=use_e8m0,
        )

    result = out.reshape(*leading, n_full * gs)

    # Tail elements that do not fill a whole group are returned unquantized.
    if rem > 0:
        result = torch.cat([result, x[..., -rem:]], dim=-1)
    return result


@torch.compiler.disable()
def fake_quant_dual_triton(x, grid_a, grid_b, gs=16,
                           scale_quant=False, global_scale=None):
    _check_group_size(gs)

    *leading, dim = x.shape
    n_full = dim // gs
    rem = dim - n_full * gs
    if n_full == 0:
        return x

    ga, ta, ga_max = _prepare_grid(grid_a, x.device)
    gb, tb, gb_max = _prepare_grid(grid_b, x.device)
    if len(ga) != len(gb):
        raise ValueError("Both grids must have the same number of levels")

    use_scale_quant, use_e8m0 = _parse_scale_quant(scale_quant, global_scale)

    x_flat = x[..., : n_full * gs].reshape(-1).contiguous()
    n = x_flat.numel()
    out = torch.empty_like(x_flat)

    # Always pass a valid pointer.
    if use_scale_quant and not use_e8m0:
        gs_t = _prepare_global_scale(global_scale, x.device)
    else:
        gs_t = torch.zeros(1, dtype=torch.float32, device=x.device)

    launch_grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    with torch.device(x.device):
        _dual_grid_quantize_kernel[launch_grid](
            x_flat,
            ga, ta,
            gb, tb,
            gs_t,
            out,
            ga_max, gb_max,
            n,
            GROUP_SIZE=gs,
            N_GRID=len(ga),
            SCALE_QUANT=use_scale_quant,
            E8M0_SCALE=use_e8m0,
        )

    result = out.reshape(*leading, n_full * gs)
    # Tail elements that do not fill a whole group are returned unquantized.
    if rem > 0:
        result = torch.cat([result, x[..., -rem:]], dim=-1)
    return result
