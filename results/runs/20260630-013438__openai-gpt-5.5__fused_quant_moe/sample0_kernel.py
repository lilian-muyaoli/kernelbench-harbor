import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _quant_moe_all_experts_kernel(
    x_ptr,
    q_ptr,
    scales_ptr,
    all_out_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_t = pid_t * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), tl.float32)

    for g in range(0, D // GROUP):
        d_base = g * GROUP
        x = tl.load(
            x_ptr + offs_t[:, None] * D + (d_base + offs_k[None, :]),
            mask=offs_t[:, None] < T,
            other=0.0,
        )

        q_i8 = tl.load(
            q_ptr + pid_e * D * N + (d_base + offs_k[:, None]) * N + offs_n[None, :],
            mask=offs_n[None, :] < N,
            other=0,
        )

        s = tl.load(
            scales_ptr + pid_e * (D // GROUP) * N + g * N + offs_n,
            mask=offs_n < N,
            other=0.0,
        )

        q = (q_i8.to(tl.float32) * s[None, :].to(tl.float32)).to(tl.float16)
        acc += tl.dot(x, q, out_dtype=tl.float32)

    tl.store(
        all_out_ptr + pid_e * T * N + offs_t[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < N),
    )


@triton.jit
def _combine_top2_kernel(
    all_out_ptr,
    topi_ptr,
    topv_ptr,
    out_ptr,
    T: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = offs_n < N

    e0 = tl.load(topi_ptr + pid_t * 2 + 0)
    e1 = tl.load(topi_ptr + pid_t * 2 + 1)
    g0 = tl.load(topv_ptr + pid_t * 2 + 0).to(tl.float32)
    g1 = tl.load(topv_ptr + pid_t * 2 + 1).to(tl.float32)

    y0 = tl.load(all_out_ptr + e0 * T * N + pid_t * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(all_out_ptr + e1 * T * N + pid_t * N + offs_n, mask=mask, other=0.0).to(tl.float32)

    out = y0 * g0 + y1 * g1
    tl.store(out_ptr + pid_t * N + offs_n, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qweight, scales, router_logits):
        T, D = x.shape
        E, _, N = qweight.shape

        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)

        all_out = torch.empty((E, T, N), device=x.device, dtype=x.dtype)

        BM = 16
        BN = 64
        BK = 128

        grid_moe = (triton.cdiv(T, BM), E, triton.cdiv(N, BN))
        _quant_moe_all_experts_kernel[grid_moe](
            x,
            qweight,
            scales,
            all_out,
            T,
            D,
            E,
            N,
            BM,
            BN,
            BK,
            num_warps=4,
            num_stages=3,
        )

        out = torch.empty((T, N), device=x.device, dtype=x.dtype)
        grid_combine = (T, triton.cdiv(N, BN))
        _combine_top2_kernel[grid_combine](
            all_out,
            topi,
            topv,
            out,
            T,
            N,
            BN,
            num_warps=4,
        )
        return out