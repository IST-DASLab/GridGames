from typing import Callable

import torch
from dataclasses import dataclass, field, asdict
from torch import nn
from enum import Enum
try:
    from transformers.utils.quantization_config import QuantizationConfigMixin
except ImportError:
    from transformers import QuantizationConfigMixin


class LutType(str, Enum):
    INT4 = "int4"
    FP4  = "fp4"
    NF4  = "nf4"

class ScaleType(str, Enum):
    E4M3 = "e4m3"
    E7M0 = "e7m0"

    def quantize(self, scales) -> torch.Tensor:
        match self:
            case ScaleType.E4M3: return scale_quantize_e4m3(scales)
            case ScaleType.E7M0: return scale_quantize_e7m0(scales)

    def dequantize(self, scales) -> torch.Tensor:
        match self:
            case ScaleType.E4M3: return scale_dequantize_e4m3(scales)
            case ScaleType.E7M0: return scale_dequantize_e7m0(scales)

    def max(self) -> float:
        match self:
            case ScaleType.E4M3: return 448.0
            case ScaleType.E7M0: return 2.0**63

    @property
    def dtype(self):
        match self:
            case ScaleType.E4M3: return torch.float8_e4m3fn
            case ScaleType.E7M0: return torch.uint8


@dataclass
class Po2QuantConfig(QuantizationConfigMixin):
    quant_method:    str       = "po2"
    bits:            int       = 4
    group_size:      int       = 16
    lut_types:       list[LutType]       = field(default_factory=lambda: [LutType.INT4, LutType.NF4])
    scale_type:      ScaleType       = ScaleType.E4M3
    modules_to_skip: list[str]       = field(default_factory=lambda: ["lm_head", "embed_tokens"])

    def __post_init__(self):
        assert 1 <= len(self.lut_types) <= 2, "one or two LUT types only"
        assert self.group_size in [16, 32], "group size must be 16 or 32"
        self.lut_types = [LutType(t) for t in self.lut_types]
        self.scale_type = ScaleType(self.scale_type)

    def get_luts(self) -> torch.Tensor:
        return get_luts(self.lut_types)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lut_types"] = [t.value for t in self.lut_types]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Po2QuantConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _make_lut(lut_type: LutType) -> torch.Tensor:
    match lut_type:
        case LutType.INT4:
            vals = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        case LutType.FP4:
            # E2M1: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
            vals = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
        case LutType.NF4:
            # normalized float4 from QLoRA
            vals = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534,  0.16093020141124725,  0.24611230194568634,  0.33791524171829224,
                    0.44070982933044434,  0.5626170039176941,   0.7229568362236023,   1.0]
    return torch.tensor(vals, dtype=torch.bfloat16)


def get_luts(lut_types: list[LutType]) -> torch.Tensor:  # [r, 16]
    return torch.stack([_make_lut(t) for t in lut_types])

def scale_quantize_e4m3(scale: torch.Tensor) -> torch.Tensor:
    """Round scales to the nearest e4m3 float8 representable value."""
    return scale.to(torch.float8_e4m3fn)

def scale_quantize_e7m0(scale: torch.Tensor) -> torch.Tensor:
    """Round scales up to the next power of two for e7m0."""
    return (scale.abs().log2().ceil().to(torch.int8).clamp(-64, 63) + 64).to(torch.uint8)

def scale_dequantize_e4m3(scale: torch.Tensor) -> torch.Tensor:
    """Round scales to the nearest e4m3 float8 representable value."""
    return scale.view(torch.float8_e4m3fn).to(torch.float)

def scale_dequantize_e7m0(scale: torch.Tensor) -> torch.Tensor:
    """Round scales up to the next power of two for e7m0."""
    return torch.pow(2.0, (scale.view(torch.int8) - 64).to(torch.float))

def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 [groups, gs] into uint32 [groups, gs//8], MSB-first (index 0 → bits 31:28)."""
    x = indices.reshape(*indices.shape[:-1], -1, 8).to(torch.int32)
    shifts = torch.tensor([28, 24, 20, 16, 12, 8, 4, 0], dtype=torch.int32, device=indices.device)
    return (x << shifts).sum(dim=-1).to(torch.int32).view(torch.uint32)

def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint32 [groups, 2] into uint8 [groups, 16], MSB-first."""
    x = packed.view(torch.int32).unsqueeze(-1)
    shifts = torch.tensor([28, 24, 20, 16, 12, 8, 4, 0], dtype=torch.int32, device=packed.device)
    return ((x >> shifts) & 0xF).flatten(-2).to(torch.uint8)


def compute_scales(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """
    Compute per-group scales for `x` given a LUT's value range.

    Args:
        x:    [groups, group_size] input tensor.
        lut:  [lut_size] lookup table of quantized values.

    Returns:
        scales: [groups, 1] — per-group scale factor.
    """
    lut_max = lut.max()
    lut_min = lut.min()

    group_max = x.amax(dim=-1, keepdim=True).clamp(min=0)
    group_min = x.amin(dim=-1, keepdim=True).clamp(max=0)

    scale_pos = torch.where(lut_max > 0, group_max / lut_max, torch.zeros_like(group_max))
    scale_neg = torch.where(lut_min < 0, group_min / lut_min, torch.zeros_like(group_min))

    return torch.maximum(scale_pos, scale_neg)


def quantize_to_simple_lut(
        x: torch.Tensor,
        lut: torch.Tensor,
        scaler: ScaleType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a grouped tensor to the nearest LUT entries with a per-group scale.
    Scales are quantized with the given scaler.

    Args:
        x:      [groups, group_size] input tensor.
        lut:    [lut_size] lookup table of quantized values.
        scaler: ScaleType to quantize the scales (e.g. to e4m3 or e7m0).

    Returns:
        indices: uint8 [groups, group_size] — index into `lut` for each element.
        scales:  [groups] — per-group scale factor.
    """
    scale = scaler.quantize(compute_scales(x, lut))

    x_scaled = x / scaler.dequantize(scale).clamp(min=1e-9).float()

    diff    = (x_scaled.unsqueeze(-1) - lut.view(1, 1, -1)).abs()
    indices = diff.argmin(dim=-1).to(torch.uint8)

    return indices, scale.squeeze(-1)


def dequantize_from_simple_lut(
        indices: torch.Tensor,
        scales:  torch.Tensor,
        lut:     torch.Tensor,
        scaler: ScaleType,
) -> torch.Tensor:
    """
    Reconstruct a tensor from LUT indices and per-group scales.

    Args:
        indices:  uint8 [groups, group_size] — index into `lut` for each element.
        scales:   [groups] — per-group scale factor.
        lut:      [lut_size] lookup table of quantized values.

    Returns:
        Reconstructed tensor of shape [groups, group_size].
    """
    return lut[indices.long()] * scaler.dequantize(scales).float().unsqueeze(-1)


def quantize_simple_po2(
        weight:     torch.Tensor,               # [m, k]   bfloat16
        luts:       torch.Tensor,               # [r, 16]  bfloat16
        scaler:     ScaleType,
        group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight matrix by selecting the best LUT per group of 16 elements.

    Each group is quantized against all r candidate LUTs; the one with the
    lowest MSE reconstruction error is selected.

    Args:
        weight: bfloat16 [m, k] — weight matrix to quantize.
        luts:   bfloat16 [r, 16] — r candidate LUTs.
        scaler: ScaleType to quantize per-group scales (e.g. to e4m3 or e7m0).

    Returns:
        best_indices: uint8 [groups, group_size] — 4-bit LUT indices.
        best_scales:  [groups]                   — per-group scale factor.
        best_lut_idx: uint8 [groups]             — which LUT was selected per group.
    """
    x = weight.reshape(-1, group_size)
    r = luts.shape[0]

    # Compute the worst-case (largest) raw scale across all groups and all LUTs,
    # then derive a global pre-scale to keep per-group scales within the
    # representable range of scale_quant.
    max_representable = scaler.max()
    raw_scales = torch.stack(
        [compute_scales(x, luts[i]) for i in range(r)], dim=0
    )                                                   # [r, groups, 1]
    global_scale = (raw_scales.amax() / max_representable).clamp(min=1e-10)
    if scaler=="e7m0":
        global_scale = torch.tensor(1.0).to(device=global_scale.device,dtype=global_scale.dtype)
    x = x / global_scale

    results = [quantize_to_simple_lut(x, luts[i], scaler) for i in range(r)]
    indices = [res[0] for res in results]               # each [groups, 16] uint8
    scales  = [res[1] for res in results]               # each [groups]

    recons  = [dequantize_from_simple_lut(indices[i], scales[i], luts[i], scaler) for i in range(r)]
    mse     = torch.stack(
        [((x - recons[i]) ** 2).mean(dim=-1) for i in range(r)],
        dim=0,
    )                                                   # [r, groups]

    best_lut_idx = mse.argmin(dim=0)                    # [groups]

    best_indices = torch.stack(indices, dim=0)[best_lut_idx, torch.arange(x.shape[0])]
    best_scales  = torch.stack(scales,  dim=0)[best_lut_idx, torch.arange(x.shape[0])]

    return (
        best_indices.to(torch.uint8),
        best_scales,
        best_lut_idx.to(torch.uint8),
        global_scale,
    )


def dequantize_simple_po2(
        indices:  torch.Tensor,   # uint8 [groups, group_size]
        scales:   torch.Tensor,   # [groups]
        lut_idx:  torch.Tensor,   # uint8 [groups]
        luts:     torch.Tensor,   # [r, 16]
) -> torch.Tensor:
    """
    Reconstruct a weight matrix from its quantized representation.

    Args:
        indices:  uint8 [groups, group_size] — LUT indices per group.
        scales:   [groups] — per-group scale factor.
        lut_idx:  uint8 [groups] — which LUT was selected per group.
        luts:     [r, 16] — candidate LUTs.

    Returns:
        Reconstructed tensor of shape [groups, 16].
    """
    group_size = indices.shape[-1]
    selected_luts = luts[lut_idx.long()]                # [groups, group_size]
    indices = indices.reshape(*selected_luts.shape[:-1], group_size)
    values = selected_luts.gather(1, indices.long())    # [groups, group_size]
    return values * scales.float().unsqueeze(-1)


def pack_lut_idx(scales: torch.Tensor, lut_idx: torch.Tensor, r: int) -> torch.Tensor:
    """Pack LUT index into the high bit(s) of the scale bytes. Only r=2 supported."""
    assert r == 2, f"Only r=2 supported, got r={r}"
    assert scales.element_size() == 1, f"Expected 1-byte scale dtype, got {scales.dtype}"
    raw = scales.view(torch.uint8)
    return ((raw & 0x7F) | (lut_idx.to(torch.uint8) << 7)).view(scales.dtype)


def unpack_lut_idx(packed_scales: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack LUT index from the high bit(s) of the scale bytes. Only r=2 supported."""
    assert r == 2, f"Only r=2 supported, got r={r}"
    packed_scales = packed_scales.view(torch.uint8)
    lut_idx = (packed_scales >> 7)
    scales  = (packed_scales & 0x7F)
    return scales, lut_idx


def quantize_po2(
        weight:     torch.Tensor,   # [m, k]   bfloat16
        luts:       torch.Tensor,   # [r, 16]  bfloat16
        scaler:     ScaleType,
        group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight matrix and return packed indices, scales, and LUTs.
    """
    r = luts.shape[0]

    indices, scales, lut_idx, gs = quantize_simple_po2(weight, luts, scaler, group_size)

    packed_indices = pack_uint4(indices)
    packed_scales  = pack_lut_idx(scales, lut_idx, r)
    scaled_luts = luts * gs.to(torch.bfloat16)
    return packed_indices.reshape(weight.shape[0], weight.shape[1] // 8), packed_scales.reshape(weight.shape[0], weight.shape[1] // group_size), scaled_luts


def dequantize_po2(
        packed_indices: torch.Tensor,   # uint8 [groups, 8]
        packed_scales:  torch.Tensor,   # uint8 [groups]
        luts:           torch.Tensor,   # [r, 16]  bfloat16
        scaler:         ScaleType,
) -> torch.Tensor:
    """
    Reconstruct a weight matrix from packed indices and scales.

    Args:
        packed_indices: uint8 [groups, 8] — packed 4-bit LUT indices.
        packed_scales:  uint8 [groups] — scales with LUT index in high bit(s).
        luts:           bfloat16 [r, 16] — candidate LUTs.
        scaler:         Scale quantizer to dequantize the scales.

    Returns:
        Reconstructed tensor of shape [groups, 16].
    """
    m = packed_indices.shape[0]
    k = packed_indices.shape[1] * 8
    r = luts.shape[0]

    indices         = unpack_uint4(packed_indices.flatten())
    scales, lut_idx = unpack_lut_idx(packed_scales.flatten(), r)
    scales          = scaler.dequantize(scales)

    gs = indices.numel() // scales.numel()
    indices = indices.reshape(-1, gs)
    out = dequantize_simple_po2(indices, scales.to(luts.dtype), lut_idx, luts)
    return out.reshape(m, k)
