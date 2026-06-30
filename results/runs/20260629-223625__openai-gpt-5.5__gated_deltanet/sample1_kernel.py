import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gated_deltanet_fwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, alpha_ptr, out_ptr,
    sqb, sqh, sql, sqd,
    skb, skh, skl, skd,
    svb, svh, svl, svd,
    sbb, sbh, sbl,
    sab, sah, sal,
    sob, soh, sol, sod,
    H: tl.constexpr,
    L: tl.constexpr,
    Dk: tl.constexpr,
    Dv: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_k = tl.arange(0, BLOCK_DK)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_k = offs_k < Dk
    mask_v = offs_v < Dv

    q_base = b * sqb + h * sqh
    k_base = b * skb + h * skh
    v_base = b * svb + h * svh
    beta_base = b * sbb + h * sbh
    alpha_base = b * sab + h * sah
    out_base = b * sob + h * soh

    S = tl.zeros((BLOCK_DK, BLOCK_V), dtype=tl.float32)

    for t in tl.range(0, L, 1):
        kt = tl.load(
            k_ptr + k_base + t * skl + offs_k * skd,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        qt = tl.load(
            q_ptr + q_base + t * sql + offs_k * sqd,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        vt = tl.load(
            v_ptr + v_base + t * svl + offs_v * svd,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)

        a = tl.load(alpha_ptr + alpha_base + t * sal).to(tl.float32)
        beta_t = tl.load(beta_ptr + beta_base + t * sbl).to(tl.float32)

        kS = tl.sum(S * kt[:, None], axis=0)
        residual = vt - kS

        S = a * S + beta_t * kt[:, None] * residual[None, :]

        out = tl.sum(S * qt[:, None], axis=0)

        tl.store(
            out_ptr + out_base + t * sol + offs_v * sod,
            out,
            mask=mask_v,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, beta, alpha):
        B, H, L, Dk = q.shape
        Dv = v.shape[-1]

        out = torch.empty((B, H, L, Dv), device=q.device, dtype=q.dtype)

        block_dk = triton.next_power_of_2(Dk)
        block_v = 8

        grid = (B * H, triton.cdiv(Dv, block_v))

        _gated_deltanet_fwd_kernel[grid](
            q, k, v, beta, alpha, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            alpha.stride(0), alpha.stride(1), alpha.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H=H,
            L=L,
            Dk=Dk,
            Dv=Dv,
            BLOCK_DK=block_dk,
            BLOCK_V=block_v,
            num_warps=4,
            num_stages=1,
        )

        return out