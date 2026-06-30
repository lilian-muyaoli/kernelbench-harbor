import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    x_ptr, qweight_ptr, scales_ptr, zeros_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    stride_outm, stride_outn,
    group_size, num_groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    offs_qn = (pid_n * (BLOCK_SIZE_N // 8)) + tl.arange(0, BLOCK_SIZE_N // 8)
    shifts = (tl.arange(0, 8) * 4)[None, None, :]
    
    for k_block in range(0, K, BLOCK_SIZE_K):
        offs_k = k_block + tl.arange(0, BLOCK_SIZE_K)
        x_mask = (offs_m[:, None] <