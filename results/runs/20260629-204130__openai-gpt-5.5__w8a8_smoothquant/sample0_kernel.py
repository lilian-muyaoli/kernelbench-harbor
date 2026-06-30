import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _int8_matmul_scale_kernel(
    x_ptr, w_ptr, x_scale_ptr, w_scale_ptr, out_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0,
        )
        b = tl.load(
            w_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )

        acc += tl.dot(a, b)

    xs = tl.load(x_scale_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    ws = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    out = acc.to(tl.float32) * xs[:, None] * ws[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_int8, w_int8, x_scale, w_scale):
        M = x_int8.shape[0]
        K = x_int8.shape[1]
        N = w_int8.shape[1]

        out = torch.empty((M, N), device=x_int8.device, dtype=torch.float16)

        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _int8_matmul_scale_kernel[grid](
            x_int8, w_int8, x_scale, w_scale, out,
            M, K, N,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4,
            num_stages=4,
        )

        return out