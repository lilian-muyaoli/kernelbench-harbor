import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _s6_scan_kernel(
    u_ptr, delta_ptr, A_ptr, Bm_ptr, C_ptr, y_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
    su_b: tl.constexpr, su_l: tl.constexpr, su_d: tl.constexpr,
    sd_b: tl.constexpr, sd_l: tl.constexpr, sd_d: tl.constexpr,
    sA_d: tl.constexpr, sA_s: tl.constexpr,
    sB_b: tl.constexpr, sB_l: tl.constexpr, sB_s: tl.constexpr,
    sC_b: tl.constexpr, sC_l: tl.constexpr, sC_s: tl.constexpr,
    sy_b: tl.constexpr, sy_l: tl.constexpr, sy_d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_S)

    mask_d = offs_d < D
    mask_s = offs_s < S
    mask_ds = mask_d[:, None] & mask_s[None, :]

    A_vals = tl.load(
        A_ptr + offs_d[:, None] * sA_d + offs_s[None, :] * sA_s,
        mask=mask_ds,
        other=0.0,
    )

    h = tl.zeros((BLOCK_D, BLOCK_S), dtype=tl.float32)

    for t in range(0, L):
        u_vals = tl.load(
            u_ptr + pid_b * su_b + t * su_l + offs_d * su_d,
            mask=mask_d,
            other=0.0,
        )
        delta_vals = tl.load(
            delta_ptr + pid_b * sd_b + t * sd_l + offs_d * sd_d,
            mask=mask_d,
            other=0.0,
        )
        Bm_vals = tl.load(
            Bm_ptr + pid_b * sB_b + t * sB_l + offs_s * sB_s,
            mask=mask_s,
            other=0.0,
        )
        C_vals = tl.load(
            C_ptr + pid_b * sC_b + t * sC_l + offs_s * sC_s,
            mask=mask_s,
            other=0.0,
        )

        da_arg = (delta_vals[:, None] * A_vals).to(tl.float16).to(tl.float32)
        dA = tl.exp(da_arg).to(tl.float16).to(tl.float32)

        dB = (delta_vals[:, None] * Bm_vals[None, :]).to(tl.float16)
        dBu = (dB * u_vals[:, None]).to(tl.float16).to(tl.float32)

        h = dA * h + dBu

        y_vals = tl.sum(h * C_vals[None, :].to(tl.float32), axis=1)

        tl.store(
            y_ptr + pid_b * sy_b + t * sy_l + offs_d * sy_d,
            y_vals,
            mask=mask_d,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, delta, A, Bm, C):
        Bb, L, D = u.shape
        S = A.shape[1]
        y = torch.empty_like(u)

        BLOCK_D = 16
        BLOCK_S = triton.next_power_of_2(S)

        grid = (Bb, triton.cdiv(D, BLOCK_D))

        _s6_scan_kernel[grid](
            u, delta, A, Bm, C, y,
            Bb, L, D, S,
            u.stride(0), u.stride(1), u.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            Bm.stride(0), Bm.stride(1), Bm.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_D=BLOCK_D,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        return y