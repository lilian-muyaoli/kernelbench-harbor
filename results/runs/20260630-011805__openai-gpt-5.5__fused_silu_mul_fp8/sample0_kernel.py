import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128
FP8_MAX = 448.0


@triton.jit
def _silu_mul_fp8_dequant_kernel(y_ptr, out_ptr, H: tl.constexpr, H2: tl.constexpr, BLOCK: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = pid_g * BLOCK + tl.arange(0, BLOCK)

    gate = tl.load(y_ptr + pid_t * H2 + offs).to(tl.float32)
    up = tl.load(y_ptr + pid_t * H2 + H + offs).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-gate))
    act = gate * sig * up

    abs_act = tl.abs(act)
    amax = tl.max(abs_act, axis=0)
    scale = tl.maximum(amax / FP8_MAX, 1.0e-12)

    x = act / scale
    x = tl.minimum(tl.maximum(x, -FP8_MAX), FP8_MAX)

    q = x.to(tl.float8e4nv).to(tl.float32)
    res = q * scale

    tl.store(out_ptr + pid_t * H + offs, res)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y):
        T, H2 = y.shape
        H = H2 // 2
        out = torch.empty((T, H), device=y.device, dtype=torch.float32)

        grid = (T, triton.cdiv(H, GROUP))
        _silu_mul_fp8_dequant_kernel[grid](
            y,
            out,
            H,
            H2,
            BLOCK=GROUP,
            num_warps=4,
        )
        return out