import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def gemm_awq_kernel(
    x_ptr, qw_ptr, scales_ptr, zeros_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_sg, stride_sn,
    stride_zg, stride_zn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # packed column index: each int32 holds 8 nibbles -> column n maps to packed col n//8
    packed_col = offs_n // 8
    nib_idx = offs_n % 8
    shift = (nib_idx * 4).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k = tl.cdiv(K, BLOCK_K)
    for k in range(num_k):
        k_base = k * BLOCK_K
        cur_k = k_base + offs_k

        # load x block
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + cur_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (cur_k[None, :] < K)
        x_blk = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # load packed weights
        qw_ptrs = qw_ptr + cur_k[:, None] * stride_qwk + packed_col[None, :] * stride_qwn
        qw_mask = (cur_k[:, None] < K) & (offs_n[None, :] < N)
        qw = tl.load(qw_ptrs, mask=qw_mask, other=0)
        w = ((qw >> shift[None, :]) & 0xF).to(tl.float32)

        # group index per k row
        g = cur_k // GROUP_SIZE
        s_ptrs = scales_ptr + g[:, None] * stride_sg + offs_n[None, :] * stride_sn
        z_ptrs = zeros_ptr + g[:, None] * stride_zg + offs_n[None, :] * stride_zn
        smask = (cur_k[:, None] < K) & (offs_n[None, :] < N)
        s = tl.load(s_ptrs, mask=smask, other=0.0).to(tl.float32)
        z = tl.load(z_ptrs, mask=smask, other=0.0).to(tl.float32)

        wdq = (w - z) * s  # (BLOCK_K, BLOCK_N)

        acc += tl.dot(x_blk, wdq)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


class ModelNew(nn.Module):
    def forward(self, x, qweight, scales, zeros):
        K = qweight.shape[0]
        N = qweight.shape[1] * 8
        M = x.shape[0]

        x = x.contiguous()
        qweight = qweight.contiguous()
        scales = scales.contiguous()
        zeros = zeros.contiguous()

        out = torch.empty((M, N), device=x.device, dtype=torch.float16)

        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        gemm_awq_kernel[grid](
            x, qweight, scales, zeros, out,
            M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP,
        )
        return out