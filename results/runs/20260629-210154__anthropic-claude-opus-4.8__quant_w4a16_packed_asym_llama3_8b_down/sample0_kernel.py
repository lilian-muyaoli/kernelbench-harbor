import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def gemm_awq_kernel(
    x_ptr, qweight_ptr, scales_ptr, zeros_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qk, stride_qn,
    stride_sg, stride_sn,
    stride_zg, stride_zn,
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # qweight column index: each packed int32 holds 8 nibbles
    # output column n -> packed col n//8, shift (n%8)*4
    qcol = offs_n // 8
    qshift = (offs_n % 8) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk
        x_block = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)

        q_ptrs = qweight_ptr + k[:, None] * stride_qk + qcol[None, :] * stride_qn
        q_packed = tl.load(q_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        w = (q_packed >> qshift[None, :]) & 0xF
        w = w.to(tl.float32)

        g = k // GROUP_SIZE
        s_ptrs = scales_ptr + g[:, None] * stride_sg + offs_n[None, :] * stride_sn
        z_ptrs = zeros_ptr + g[:, None] * stride_zg + offs_n[None, :] * stride_zn
        s = tl.load(s_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        z = tl.load(z_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0).to(tl.float32)

        wdq = (w - z) * s

        acc += tl.dot(x_block.to(tl.float32), wdq)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
            GROUP_SIZE=GROUP,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out