import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128
FP8_MAX = 448.0


@triton.jit
def _silu_mul_fp8_dequant_kernel(
    y_ptr,
    out_ptr,
    H: tl.constexpr,
    SY0: tl.constexpr,
    SY1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    grp = tl.program_id(1)

    offs = tl.arange(0, BLOCK)
    col = grp * BLOCK + offs

    gate = tl.load(y_ptr + row * SY0 + col * SY1).to(tl.float32)
    up = tl.load(y_ptr + row * SY0 + (H + col) * SY1).to(tl.float32)

    act = (gate / (1.0 + tl.exp(-gate))) * up

    amax = tl.max(tl.abs(act), axis=0)
    scale = tl.maximum(amax * (1.0 / FP8_MAX), 1.0e-12)

    v = tl.maximum(tl.minimum(act / scale, FP8_MAX), -FP8_MAX)
    q = v.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.float32)
    out = q * scale

    tl.store(out_ptr + row * SO0 + col * SO1, out)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y):
        T, H2 = y.shape
        H = H2 // 2
        out = torch.empty((T, H), device=y.device, dtype=torch.float32)

        grid = (T, H // GROUP)
        _silu_mul_fp8_dequant_kernel[grid](
            y,
            out,
            H,
            y.stride(0),
            y.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK=GROUP,
            num_warps=4,
            num_stages=4,
        )
        return out