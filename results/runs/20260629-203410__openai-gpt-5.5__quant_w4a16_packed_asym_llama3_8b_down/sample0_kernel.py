import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _int4_grouped_dequant_matmul_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    sxm: tl.constexpr,
    sxk: tl.constexpr,
    sqk: tl.constexpr,
    sqp: tl.constexpr,
    ssg: tl.constexpr,
    ssn: tl.constexpr,
    szg: tl.constexpr,
    szn: tl.constexpr,
    som: tl.constexpr,
    son: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    word_cols = offs_n // 8
    shifts = ((offs_n & 7) * 4).to(tl.int32)

    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + offs_k_base

        x_vals = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        q_vals = tl.load(
            qweight_ptr + offs_k[:, None] * sqk + word_cols[None, :] * sqp,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )

        nibbles = ((q_vals >> shifts[None, :]) & 15).to(tl.float32)

        g = k0 // GROUP
        sc = tl.load(
            scales_ptr + g * ssg + offs_n * ssn,
            mask=offs_n < N,
            other=0.0,
        )
        ze = tl.load(
            zeros_ptr + g * szg + offs_n * szn,
            mask=offs_n < N,
            other=0.0,
        )

        sub = (nibbles - ze[None, :].to(tl.float32)).to(tl.float16)
        w_vals = (sub.to(tl.float32) * sc[None, :].to(tl.float32)).to(tl.float16)

        acc += tl.dot(x_vals, w_vals, out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qweight, scales, zeros):
        M = x.shape[0]
        K = qweight.shape[0]
        N = qweight.shape[1] * 8

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = GROUP

        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))

        _int4_grouped_dequant_matmul_kernel[grid](
            x,
            qweight,
            scales,
            zeros,
            out,
            M,
            K,
            N,
            x.stride(0),
            x.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            zeros.stride(0),
            zeros.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
        return out