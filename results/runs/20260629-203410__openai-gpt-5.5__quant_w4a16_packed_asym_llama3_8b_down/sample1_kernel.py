import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _int4_group_gemm_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    y_ptr,
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
    sym: tl.constexpr,
    syn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    pack_n = offs_n // 8
    shifts = ((offs_n & 7) * 4).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k_base

        a = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        packed = tl.load(
            qweight_ptr + offs_k[:, None] * sqk + pack_n[None, :] * sqp,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )

        q = ((packed >> shifts[None, :]) & 0xF).to(tl.float32)

        group_id = k0 // GROUP
        sc = tl.load(
            scales_ptr + group_id * ssg + offs_n * ssn,
            mask=offs_n < N,
            other=0.0,
        )
        ze = tl.load(
            zeros_ptr + group_id * szg + offs_n * szn,
            mask=offs_n < N,
            other=0.0,
        )

        diff = (q.to(tl.float16) - ze[None, :].to(tl.float16)).to(tl.float16)
        b = (diff * sc[None, :].to(tl.float16)).to(tl.float16)

        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn,
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

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BM = 16
        BN = 32
        BK = 64

        grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))

        _int4_group_gemm_kernel[grid](
            x,
            qweight,
            scales,
            zeros,
            y,
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
            y.stride(0),
            y.stride(1),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            num_warps=4,
            num_stages=3,
        )
        return y