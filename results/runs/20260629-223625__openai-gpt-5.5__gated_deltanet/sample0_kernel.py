import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gated_delta_fwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, alpha_ptr, o_ptr,
    H: tl.constexpr, L: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    sqb: tl.constexpr, sqh: tl.constexpr, sql: tl.constexpr, sqd: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, skl: tl.constexpr, skd: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svl: tl.constexpr, svd: tl.constexpr,
    sbb: tl.constexpr, sbh: tl.constexpr, sbl: tl.constexpr,
    sab: tl.constexpr, sah: tl.constexpr, sal: tl.constexpr,
    sob: tl.constexpr, soh: tl.constexpr, sol: tl.constexpr, sod: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_dv = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_k = tl.arange(0, BLOCK_DK)
    offs_v = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

    mask_k = offs_k < DK
    mask_v = offs_v < DV

    S = tl.zeros((BLOCK_DK, BLOCK_DV), dtype=tl.float32)

    q_base = q_ptr + b * sqb + h * sqh
    k_base = k_ptr + b * skb + h * skh
    v_base = v_ptr + b * svb + h * svh
    beta_base = beta_ptr + b * sbb + h * sbh
    alpha_base = alpha_ptr + b * sab + h * sah
    o_base = o_ptr + b * sob + h * soh

    for t in tl.range(0, L, 1):
        kt = tl.load(k_base + t * skl + offs_k * skd, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_base + t * sql + offs_k * sqd, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_base + t * svl + offs_v * svd, mask=mask_v, other=0.0).to(tl.float32)

        a = tl.load(alpha_base + t * sal).to(tl.float32)
        be = tl.load(beta_base + t * sbl).to(tl.float32)

        kS = tl.sum(S * kt[:, None], axis=0)
        diff = vt - kS

        S = a * S + be * kt[:, None] * diff[None, :]

        out = tl.sum(S * qt[:, None], axis=0)
        tl.store(o_base + t * sol + offs_v * sod, out, mask=mask_v)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, beta, alpha):
        B, H, L, Dk = q.shape
        Dv = v.shape[-1]

        out = torch.empty((B, H, L, Dv), device=q.device, dtype=q.dtype)

        block_dk = triton.next_power_of_2(Dk)
        block_dv = 16

        grid = (B * H, triton.cdiv(Dv, block_dv))

        _gated_delta_fwd_kernel[grid](
            q, k, v, beta, alpha, out,
            H, L, Dk, Dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            alpha.stride(0), alpha.stride(1), alpha.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_DK=block_dk,
            BLOCK_DV=block_dv,
            num_warps=8,
        )

        return out