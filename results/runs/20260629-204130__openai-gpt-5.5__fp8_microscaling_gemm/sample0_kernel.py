import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK = 32


@triton.jit
def _scaled_fp8_matmul_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )

        w_fp8 = tl.load(
            w_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=offs_n[None, :] < N,
            other=0.0,
        ).to(tl.float32)

        scale = tl.load(
            s_ptr + (k0 // BLOCK) * N + offs_n,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)

        w = (w_fp8 * scale[None, :]).to(tl.float16)
        acc += tl.dot(x, w, out_dtype=tl.float32)

    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w_fp8, scales):
        M = x.shape[0]
        K = w_fp8.shape[0]
        N = w_fp8.shape[1]

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(N, 32), triton.cdiv(M, 16))
        _scaled_fp8_matmul_kernel[grid](
            x,
            w_fp8,
            scales,
            y,
            M,
            K,
            N,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=4,
        )
        return y