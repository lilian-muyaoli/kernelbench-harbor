import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _quant_top2_moe_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    r_ptr,
    o_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    m1 = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    m2 = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    i1 = tl.full((BLOCK_M,), 0, tl.int32)
    i2 = tl.full((BLOCK_M,), 0, tl.int32)

    for e in tl.static_range(0, 8):
        rv = tl.load(r_ptr + offs_m * E + e, mask=offs_m < T, other=-float("inf")).to(tl.float32)
        gt1 = rv > m1
        gt2 = rv > m2

        old_m1 = m1
        old_i1 = i1

        m1 = tl.where(gt1, rv, m1)
        i1 = tl.where(gt1, e, i1)

        m2 = tl.where(gt1, old_m1, tl.where(gt2, rv, m2))
        i2 = tl.where(gt1, old_i1, tl.where(gt2, e, i2))

    g1 = (1.0 / (1.0 + tl.exp(m2 - m1))).to(tl.float16).to(tl.float32)
    g2 = (1.0 - (1.0 / (1.0 + tl.exp(m2 - m1)))).to(tl.float16).to(tl.float32)

    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for e in tl.static_range(0, 8):
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        k0 = 0
        while k0 < D:
            offs_k = k0 + offs_k_base
            x = tl.load(
                x_ptr + offs_m[:, None] * D + offs_k[None, :],
                mask=(offs_m[:, None] < T) & (offs_k[None, :] < D),
                other=0.0,
            ).to(tl.float16)

            q = tl.load(
                q_ptr + e * D * N + offs_k[:, None] * N + offs_n[None, :],
                mask=(offs_k[:, None] < D) & (offs_n[None, :] < N),
                other=0,
            )

            grp = k0 // GROUP_SIZE
            sc = tl.load(
                s_ptr + e * ((D + GROUP_SIZE - 1) // GROUP_SIZE) * N + grp * N + offs_n,
                mask=offs_n < N,
                other=0.0,
            ).to(tl.float32)

            w = (q.to(tl.float32) * sc[None, :]).to(tl.float16)
            acc += tl.dot(x, w, out_dtype=tl.float32)

            k0 += BLOCK_K

        coeff = tl.where(i1 == e, g1, 0.0) + tl.where(i2 == e, g2, 0.0)
        acc = acc.to(tl.float16).to(tl.float32)
        out += acc * coeff[:, None]

    tl.store(
        o_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < T) & (offs_n[None, :] < N),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qweight, scales, router_logits):
        T, D = x.shape
        E, _, N = qweight.shape
        out = torch.empty((T, N), device=x.device, dtype=x.dtype)

        _quant_top2_moe_kernel[(triton.cdiv(T, 16), triton.cdiv(N, 64))](
            x,
            qweight,
            scales,
            router_logits,
            out,
            T,
            D,
            E,
            N,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=64,
            GROUP_SIZE=GROUP,
            num_warps=4,
            num_stages=3,
        )
        return out