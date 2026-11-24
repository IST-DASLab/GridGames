"""
Tests for the N-grid Triton quantize/dequantize API in
fp_quant.module.triton.PO2_quant_dequant.

Run with: python test_grids.py
"""
import sys
import os


import torch
from transformers import AutoModelForCausalLM
from inference_lib.src.fp_quant.module.triton.PO2_quant_dequant import (
    fake_quant_dequant_N_triton,   
    quantize_N_triton,
    dequantize_N_triton,
)
from inference_lib.src.fp_quant.module.triton.PO2 import (
    fake_quant_grid_triton,
    fake_quant_dual_triton,
)
from inference_lib.tests.reference_implementation import (
    ScaleType,
    compute_scales,
    quantize_simple_po2,
    dequantize_simple_po2,
)

EPS=10**-5

def _make_inputs(device, seed=0):
    torch.manual_seed(seed)

    x = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype=torch.float32).model.layers[10].mlp.gate_proj.weight.detach().to(device).contiguous()
    
    grid_a = torch.tensor([-7.0000, -5.0000, -3.7500, -2.7500, -2.0000, -1.2500, -0.6250,  0.0000,
         0.5000,  1.1250,  1.7500,  2.2500,  3.0000,  4.0000,  5.0000,  7.0000]
        , device=device
    )
    grid_b = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 
                           0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device
    )
    return x, grid_a, grid_b


def test_n2_matches_dual_no_scale_quant(x, grid_a, grid_b, device):
    gs = 16

    y_dual = fake_quant_dual_triton(x, grid_a, grid_b, gs=gs)
    y_N = fake_quant_dequant_N_triton(x, [grid_a, grid_b], gs=gs, scale_candidates=(1.0,))

    diff = (y_dual - y_N).abs().max().item()
    assert diff < EPS, f"dual vs N=2 output diff: {diff}"
    print(f"  [ok] dual vs N=2 (no scale_quant)        max-abs = {diff}")


def test_n2_matches_dual_e8m0(x, grid_a, grid_b, device):
    gs = 16

    y_dual = fake_quant_dual_triton(x, grid_a, grid_b, gs=gs, scale_quant="e8m0")
    y_N = fake_quant_dequant_N_triton(x, [grid_a, grid_b], gs=gs, scale_quant="e8m0", scale_candidates=(1.0,))

    diff = (y_dual - y_N).abs().max().item()
    assert diff < EPS, f"dual vs N=2 (E8M0) output diff: {diff}"
    print(f"  [ok] dual vs N=2 (E8M0)                  max-abs = {diff}")




def test_n3_is_at_least_as_good_as_n2(x, grid_a, grid_b, device):
    """Adding a third candidate grid must not increase MSE."""
    gs = 16

    y_N2 = fake_quant_dequant_N_triton(x, [grid_a, grid_b], gs=gs)
    y_N3 = fake_quant_dequant_N_triton(x, [grid_a, grid_b, grid_a + 0.5], gs=gs)

    mse2 = (y_N2 - x).pow(2).mean().item()
    mse3 = (y_N3 - x).pow(2).mean().item()
    assert mse3 <= mse2, f"N=3 MSE ({mse3}) > N=2 MSE ({mse2})"
    print(f"  [ok] N=3 MSE <= N=2 MSE                   N=2={mse2:.6e}  N=3={mse3:.6e}")


def test_round_trip_matches_fake_quant_dequant(x, grid_a, grid_b, device):
    """Explicit quantize→dequantize must match the fake_quant_dequant wrapper."""
    gs = 16
    y_fqd = fake_quant_dequant_N_triton(x, [grid_a, grid_b], gs=gs)
    q, sc, gid, _ = quantize_N_triton(x, [grid_a, grid_b], gs=gs)
    y_rt = dequantize_N_triton(q, sc, gid, [grid_a, grid_b], gs=gs)
    print(_)
    diff = (y_rt - y_fqd).abs().mean().item()
    print((y_rt - y_fqd).abs())
    assert diff < EPS, f"round-trip vs fake_qd diff: {diff}"

    used = torch.unique(gid).tolist()
    assert all(0 <= g < 2 for g in used), f"unexpected grid_ids: {used}"
    print(f"  [ok] quantize->dequantize vs fake_qd     max-abs = {diff}  gids_used = {used}")

def test_quant_doing_quant(x, grid_a, grid_b, device):
    """Explicit quantize→dequantize must match the fake_quant_dequant wrapper."""
    gs = 16
    q, sc, gid, _ = quantize_N_triton(x, [grid_a, grid_b], gs=gs)
    print(q.unique())
    print(gid.unique())
    assert len(q.unique()) <= 16
    assert len(gid.unique()) <= 2
    q, sc, gid, _ = quantize_N_triton(x, [grid_a, grid_b, grid_a + 0.5], gs=gs)
    assert len(q.unique()) <= 16
    print("unqique values", gid.unique())
    assert len(gid.unique()) <= 3
    assert len(q.unique()) <= 16
    y_rt = dequantize_N_triton(q, sc, gid, [grid_a, grid_b, grid_a + 0.5], gs=gs)
    assert ((x-y_rt)**2).mean()<= ((x)**2).mean()
    assert torch.numel(sc)<=torch.numel(x)/gs
    print(f"number of weights/number of scales {torch.numel(x)/torch.numel(sc)}")
    print(sc.dtype)
    print(q.dtype)
    # print(sc)
    # assert len(sc.unique())<=2**8, sc.unique() # Hey it is not quantized function


def test_quant_doing_quant_nvfp(x, grid_a, grid_b, device):
    """Quantize→dequantize with NVFP-style E4M3 scale quantization."""
    gs = 16
    global_scale = torch.tensor(448.0, device=device)

    q, sc, gid, _ = quantize_N_triton(
        x, [grid_a, grid_b], gs=gs, scale_quant="e4m3", global_scale=global_scale
    )
    print(q.unique())
    print(gid.unique())
    assert len(q.unique()) <= 16
    assert len(gid.unique()) <= 2

    q, sc, gid, _ = quantize_N_triton(
        x, [grid_a, grid_b, grid_a + 0.5], gs=gs,
        scale_quant="e4m3", global_scale=global_scale,
    )
    assert len(q.unique()) <= 16
    print("unqique values", gid.unique())
    assert len(gid.unique()) <= 3
    assert len(q.unique()) <= 16

    y_rt = dequantize_N_triton(q, sc, gid, [grid_a, grid_b, grid_a + 0.5], gs=gs)
    assert ((x - y_rt) ** 2).mean() <= ((x) ** 2).mean()
    assert torch.numel(sc) <= torch.numel(x) / gs
    print(f"number of weights/number of scales {torch.numel(x)/torch.numel(sc)}")
    print(sc.dtype)
    print(q.dtype)
    assert len(sc.unique()) <= 2 ** 8


def test_quant_doing_quant_mxfp(x, grid_a, grid_b, device):
    """Quantize→dequantize with MXFP-style E8M0 scale quantization."""
    gs = 32

    q, sc, gid, _ = quantize_N_triton(x, [grid_a, grid_b], gs=gs, scale_quant="e8m0")
    print(q.unique())
    print(gid.unique())
    assert len(q.unique()) <= 16
    assert len(gid.unique()) <= 2

    q, sc, gid, _ = quantize_N_triton(
        x, [grid_a, grid_b, grid_a + 0.5], gs=gs, scale_quant="e8m0"
    )
    assert len(q.unique()) <= 16
    print("unqique values", gid.unique())
    assert len(gid.unique()) <= 3
    assert len(q.unique()) <= 16

    y_rt = dequantize_N_triton(q, sc, gid, [grid_a, grid_b, grid_a + 0.5], gs=gs)
    assert ((x - y_rt) ** 2).mean() <= ((x) ** 2).mean()
    assert torch.numel(sc) <= torch.numel(x) / gs
    print(f"number of weights/number of scales {torch.numel(x)/torch.numel(sc)}")
    print(sc.dtype)
    print(q.dtype)
    assert len(sc.unique()) <= 2 ** 8


def _reference_fake_quant_dequant(x, grids, scaler, gs):
    """Run reference quantize→dequantize pipeline, return reconstruction matching x.shape/dtype."""
    luts = torch.stack(list(grids), dim=0)
    indices, scales, lut_idx, global_scale = quantize_simple_po2(x, luts, scaler, gs)
    scales_dequant = scaler.dequantize(scales).to(luts.dtype)
    recon = dequantize_simple_po2(indices, scales_dequant, lut_idx, luts)
    recon = recon.to(torch.float32) * global_scale.to(torch.float32)
    return recon.reshape(x.shape).to(x.dtype)


def _triton_global_scale(x, grids, scaler, gs):
    """Inverse of the reference's pre-normalization global_scale (lines 219-224 of
    reference_implementation.py), which is what the triton E4M3 path expects."""
    x_reshaped = x.reshape(-1, gs)
    luts = torch.stack(list(grids), dim=0)
    r = luts.shape[0]
    max_representable = scaler.max()
    raw_scales = torch.stack(
        [compute_scales(x_reshaped, luts[i]) for i in range(r)], dim=0
    )
    global_scale_ref = (raw_scales.amax() / max_representable).clamp(min=1e-10)
    return (1.0 / global_scale_ref).to(torch.float32)


def test_nvfp_matches_reference(x, grid_a, grid_b, device):
    """Triton NVFP (E4M3) fake-quant matches the reference E4M3 pipeline."""
    gs = 16
    grids = [grid_a, grid_b]
    global_scale = _triton_global_scale(x, grids, ScaleType.E4M3, gs)

    y_triton = fake_quant_dequant_N_triton(
        x, grids, gs=gs, scale_quant="e4m3", global_scale=global_scale,
    )
    y_ref = _reference_fake_quant_dequant(x, grids, ScaleType.E4M3, gs)

    mse_diff = ((y_triton.float() - y_ref.float()) ** 2).mean().item()
    print(f"  [ok] NVFP triton vs reference (E4M3)   mse = {mse_diff:.3e}")
    assert mse_diff < EPS, f"NVFP triton vs reference diff: {mse_diff}"

    mse_triton = ((y_triton.float() - x) ** 2).mean().item()
    mse_ref    = ((y_ref.float()    - x) ** 2).mean().item()
    mse_zero   = (x ** 2).mean().item()
    assert mse_triton < mse_zero, "triton MXFP no better than zero predictor"
    assert mse_ref    < mse_zero, "reference E7M0 no better than zero predictor"

    ratio = max(mse_triton / mse_ref, mse_ref / mse_triton)
    print(f"{ratio=}")
    assert ratio < 1.2, f"triton/reference MXFP MSE ratio too large: {ratio}"

def test_mxfp_matches_reference(x, grid_a, grid_b, device):
    """Triton MXFP (E8M0) fake-quant reconstruction is consistent with reference E7M0 pipeline.

    Triton's E8M0 uses floor(log2(vec_max)) - floor(log2(grid_max)) while the
    reference uses ceil(log2(raw_scale)). They agree on most groups but can
    differ by one power-of-two on groups where log2(vec_max) is fractional, so
    we compare reconstruction MSE rather than expecting bit-exact equality.
    """
    gs = 32
    grids = [grid_a, grid_b]

    y_triton = fake_quant_dequant_N_triton(x, grids, gs=gs, scale_quant="e8m0")
    y_ref = _reference_fake_quant_dequant(x, grids, ScaleType.E7M0, gs)

    mse_triton = ((y_triton.float() - x) ** 2).mean().item()
    mse_ref    = ((y_ref.float()    - x) ** 2).mean().item()
    mse_zero   = (x ** 2).mean().item()
    print(
        f"  [ok] MXFP triton vs reference (E7M0)      "
        f"mse_tri={mse_triton:.3e} mse_ref={mse_ref:.3e} mse_zero={mse_zero:.3e}"
    )
    assert mse_triton < mse_zero, "triton MXFP no better than zero predictor"
    assert mse_ref    < mse_zero, "reference E7M0 no better than zero predictor"

    ratio = max(mse_triton / mse_ref, mse_ref / mse_triton)
    print(f"{ratio=}")
    print((y_triton-y_ref).abs())
    assert ratio < 1.2, f"triton/reference MXFP MSE ratio too large: {ratio}"



def test_mxfp_matches_curated_reference(x, grid_a, grid_b, device):
    """Triton MXFP (E8M0) fake-quant reconstruction is consistent with reference E7M0 pipeline.

    Triton's E8M0 uses floor(log2(vec_max)) - floor(log2(grid_max)) while the
    reference uses ceil(log2(raw_scale)). They agree on most groups but can
    differ by one power-of-two on groups where log2(vec_max) is fractional, so
    we compare reconstruction MSE rather than expecting bit-exact equality.
    """
    gs = 32
    grids = [grid_a, grid_b]

    y_triton = fake_quant_dequant_N_triton(x, grids, gs=gs, scale_quant="e8m0")
    y_ref = _reference_fake_quant_dequant(x, grids, ScaleType.E7M0, gs)

    mse_triton = ((y_triton.float() - x) ** 2).mean().item()
    mse_ref    = ((y_ref.float()    - x) ** 2).mean().item()
    mse_zero   = (x ** 2).mean().item()
    print(
        f"  [ok] MXFP triton vs reference (E7M0)      "
        f"mse_tri={mse_triton:.3e} mse_ref={mse_ref:.3e} mse_zero={mse_zero:.3e}"
    )
    assert mse_triton < mse_zero, "triton MXFP no better than zero predictor"
    assert mse_ref    < mse_zero, "reference E7M0 no better than zero predictor"
    
    assert mse_triton==mse_ref
    assert mse_triton==0

def test_mxfp_matches_reference_n4(x, grid_a, grid_b, device):
    """Same as test_mxfp_matches_reference but with 4 candidate grids."""
    gs = 32
    grids = [grid_a, grid_b, grid_a + 0.5, grid_b - 0.25]

    y_triton = fake_quant_dequant_N_triton(x, grids, gs=gs, scale_quant="e8m0")
    y_ref = _reference_fake_quant_dequant(x, grids, ScaleType.E7M0, gs)

    mse_triton = ((y_triton.float() - x) ** 2).mean().item()
    mse_ref    = ((y_ref.float()    - x) ** 2).mean().item()
    mse_zero   = (x ** 2).mean().item()
    print(
        f"  [ok] MXFP N=4 triton vs reference (E7M0)  "
        f"mse_tri={mse_triton:.3e} mse_ref={mse_ref:.3e} mse_zero={mse_zero:.3e}"
    )
    assert mse_triton < mse_zero, "triton MXFP N=4 no better than zero predictor"
    assert mse_ref    < mse_zero, "reference E7M0 N=4 no better than zero predictor"

    ratio = max(mse_triton / mse_ref, mse_ref / mse_triton)
    print(f"{ratio=}")
    assert ratio < 1.2, f"triton/reference MXFP N=4 MSE ratio too large: {ratio}"


def test_n4_quant_doing_quant_mxfp(x, grid_a, grid_b, device):
    """Quantize→dequantize with 4 candidate grids and MXFP-style E8M0 scales."""
    gs = 32
    grids = [grid_a, grid_b, grid_a + 0.5, grid_b - 0.25]

    q, sc, gid, _ = quantize_N_triton(x, grids, gs=gs, scale_quant="e8m0")
    print(q.unique())
    print("unique grid ids", gid.unique())
    assert len(q.unique()) <= 16
    assert len(gid.unique()) <= 4

    y_rt = dequantize_N_triton(q, sc, gid, grids, gs=gs)
    y_fqd = fake_quant_dequant_N_triton(x, grids, gs=gs, scale_quant="e8m0")
    diff = (y_rt - y_fqd).abs().max().item()
    assert diff < EPS, f"round-trip vs fake_qd diff: {diff}"

    mse4 = ((x - y_rt) ** 2).mean().item()
    mse2 = ((x - fake_quant_dequant_N_triton(x, [grid_a, grid_b], gs=gs, scale_quant="e8m0")) ** 2).mean().item()
    assert mse4 <= mse2, f"N=4 MSE ({mse4}) > N=2 MSE ({mse2})"
    assert torch.numel(sc) <= torch.numel(x) / gs
    assert len(sc.unique()) <= 2 ** 6
    print(f"  [ok] N=4 (E8M0) quant/dequant    N=2={mse2:.6e}  N=4={mse4:.6e}")




def test_n1_matches_single_grid_mxfp(x, grid_a, grid_b, device):
    """N=1 (single grid) must match the single-grid kernels for MXFP (E8M0).

    Compares fake_quant_dequant_N_triton([grid]) against:
      - fake_quant_dequant_grid_triton(grid)   (PO2_quant_dequant.py)
      - fake_quant_grid_triton(grid)           (PO2.py)
    """
    gs = 32

    y_N = fake_quant_dequant_N_triton(x, [grid_a], gs=gs, scale_quant="e8m0", scale_candidates=(1.0,))
    y_fq = fake_quant_grid_triton(x, grid_a, gs=gs, scale_quant="e8m0")

    diff_fq = (y_N - y_fq).abs().max().item()
    assert diff_fq < EPS, f"N=1 vs fake_quant_grid_triton (E8M0) diff: {diff_fq}"
    print(
        f"  [ok] N=1 vs single-grid (MXFP/E8M0)      "
        f"  max-abs(fq)={diff_fq}"
    )



def test_n2_matches_dual_mxfp(x, grid_a, grid_b, device):
    """N=2 must match the dual-grid kernels for MXFP (E8M0).

    Compares fake_quant_dequant_N_triton([grid_a, grid_b]) against:
      - fake_quant_dequant_dual_triton(grid_a, grid_b)  (PO2_quant_dequant.py)
      - fake_quant_dual_triton(grid_a, grid_b)          (PO2.py)

    The dequant path matches bit-for-bit. PO2.py's fused fake_quant_dual_triton
    may differ on a handful of groups where both grids give (near-)equal MSE and
    the tie is broken differently; both are correct, so we only require the
    reconstruction MSE to agree.
    """
    gs = 32

    y_N = fake_quant_dequant_N_triton(x, [grid_a, grid_b], gs=gs, scale_quant="e8m0", scale_candidates=(1.0,))
    y_qd = fake_quant_dual_triton(x, grid_a, grid_b, gs=gs, scale_quant="e8m0")

    diff_qd = (y_N - y_qd).abs().max().item()
    assert diff_qd < EPS, f"N=2 vs fake_quant_dequant_dual_triton (E8M0) diff: {diff_qd}"

    mse_N = ((y_N.float() - x) ** 2).mean().item()
    print(
        f"  [ok] N=2 vs dual-grid (MXFP/E8M0)        "
        f"max-abs(qd)={diff_qd}"
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping (Triton kernels require GPU).")
        return

    device = "cuda"
    print(f"Running N-grid tests on device: {device}")
    print()
    x, grid_a, grid_b = _make_inputs(device)

    test_n2_matches_dual_no_scale_quant(x, grid_a, grid_b,device)
    test_n2_matches_dual_e8m0(x, grid_a, grid_b,device)

    test_n3_is_at_least_as_good_as_n2(x, grid_a, grid_b,device)
    test_round_trip_matches_fake_quant_dequant(x, grid_a, grid_b,device)
    test_quant_doing_quant(x, grid_a, grid_b,device)
    test_quant_doing_quant_nvfp(x, grid_a, grid_b,device)
    test_quant_doing_quant_mxfp(x, grid_a, grid_b,device)
    test_n4_quant_doing_quant_mxfp(x, grid_a, grid_b,device)
    test_n1_matches_single_grid_mxfp(x, grid_a, grid_b,device)
    test_n2_matches_dual_mxfp(x, grid_a, grid_b,device)
    test_nvfp_matches_reference(x, grid_a, grid_b,device)
    test_mxfp_matches_reference(x, grid_a, grid_b,device)
    test_mxfp_matches_reference_n4(x, grid_a, grid_b,device)
    test_mxfp_matches_curated_reference(torch.cat([grid_a.repeat(10),grid_b.repeat(10)]).reshape(-1,32).cuda(), grid_a, grid_b, device)

    print()
    print("All N-grid tests passed.")


if __name__ == "__main__":
    main()
