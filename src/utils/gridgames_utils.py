import torch

from inference_lib.src.fp_quant.module.triton.PO2 import fake_quant_dual_triton, fake_quant_grid_triton




def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")



class QuantLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, activation_quantizer: torch.nn.Module = None) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.nn.Parameter(weight, requires_grad=False)  # (R, C)
        self.activation_quantizer: torch.nn.Module | None = activation_quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_quantizer is not None:
            x_fake_quant: torch.Tensor = self.activation_quantizer(x)  # (..., C)
        else:
            x_fake_quant: torch.Tensor = x  # (..., C)
        # print(x_fake_quant.dtype, self.weight.dtype)
        y: torch.Tensor = x_fake_quant @ self.weight.transpose(-2, -1)  # (..., R)
        return y

class BlockTransformQuantizerGrid(torch.nn.Module):
    """
    Activation quantizer with an optional per-block transform.

    `grids` is a list of codebooks (1 or 2 entries). With 2 grids the kernel picks the
    best per group (`fake_quant_dual_triton`); with 1 grid it falls back to
    `fake_quant_grid_triton`. The kernels themselves are not changed.

    `group_size` must be provided when `transform` is None (e.g. weights are not quantized);
    otherwise it is inferred from the transform's last dim.
    """
    def __init__(
        self,
        transform: torch.Tensor | None,
        grids: list[torch.Tensor] | None = None,
        group_size: int | None = None,
        global_scale=10.,
        scale_precision=torch.float8_e4m3fn,
    ) -> None:
        super().__init__()
        if transform is not None:
            # (C//G, G, G).
            self.register_buffer("transform", transform, persistent=True)
        else:
            self.transform = None

        if grids is None or len(grids) == 0:
            self.grid_a = None
            self.grid_b = None
        else:
            assert 1 <= len(grids) <= 2, (
                "BlockTransformQuantizerGrid supports 1 or 2 grids (kernel limit)."
            )
            self.grid_a = grids[0]
            self.grid_b = grids[1] if len(grids) == 2 else None

        if transform is not None:
            self.group_size = transform.size(-1)
        else:
            assert group_size is not None and group_size > 0, (
                "group_size must be specified when transform is None."
            )
            self.group_size = group_size

        if global_scale is not None:
            self.global_scale = global_scale
            self._track_global_scale = False
        else:
            self._track_global_scale = True
            self.global_scale = torch.tensor([float("inf")], dtype=torch.float32)
            if self.grid_a is not None:
                self.number = self.grid_a.abs().max()
                if self.grid_b is not None:
                    self.number = min(
                        self.grid_a.abs().max(),
                        self.grid_b.abs().max(),
                        -self.grid_a.min() if self.grid_a.min() <= 0 else torch.inf,
                        -self.grid_b.min() if self.grid_b.min() <= 0 else torch.inf,
                    )

        self.scale_precision = scale_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., C)  ;  transform: (C//G, G, G) or None
        """
        compute_dtype = x.dtype
        y_old = x.clone()
        y = x.to(dtype=compute_dtype, copy=False)

        if self.transform is not None:
            G = self.transform.size(-1)
            C = y.size(-1)
            assert C % G == 0, "C must be divisible by group size G"
            N = C // G

            # Reshape once: (..., N, G)
            y = y.reshape(*y.shape[:-1], N, G)

            Tt = self.transform.transpose(-2, -1).to(dtype=compute_dtype)
            # einsum keeps memory tight: result (..., N, G)
            y = torch.einsum('...ng,ngh->...nh', y, Tt)

            # Flatten back to (..., C)
            y = y.reshape(*y.shape[:-2], C)

        if self.grid_a is None and self.grid_b is None:
            return y.reshape(y_old.shape)

        if self._track_global_scale and self.scale_precision is torch.float8_e4m3fn:
            current_global_scale = ((self.number * 448) * get_reciprocal((y.abs().max())).to(torch.float32))
            if not current_global_scale:
                raise ValueError(f"Current global scale is not finite: {current_global_scale}\n")
            self.global_scale = torch.minimum(self.global_scale.to(x.device), current_global_scale)

        group_size = self.group_size
        if self.scale_precision is not torch.float8_e8m0fnu:
            if self.grid_b is not None:
                y = fake_quant_dual_triton(y, grid_a=self.grid_a, grid_b=self.grid_b, gs=group_size, scale_quant=self.scale_precision is torch.float8_e4m3fn, global_scale=self.global_scale).to(dtype=x.dtype, copy=False)
            else:
                y = fake_quant_grid_triton(y, grid=self.grid_a, gs=group_size, scale_quant=self.scale_precision is torch.float8_e4m3fn, global_scale=self.global_scale).to(dtype=x.dtype, copy=False)
        else:
            if self.grid_b is not None:
                y = fake_quant_dual_triton(y, grid_a=self.grid_a, grid_b=self.grid_b, gs=group_size, scale_quant="e8m0", global_scale=torch.tensor(1.)).to(dtype=x.dtype, copy=False)
            else:
                y = fake_quant_grid_triton(y, grid=self.grid_a, gs=group_size, scale_quant="e8m0", global_scale=torch.tensor(1.)).to(dtype=x.dtype, copy=False)
        return y

