import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def w4a16_gemm_kernel(
    x_ptr, qweight_ptr, scales_ptr, zeros_ptr, out_ptr,
    M, N, K,
    stride_x_m, stride_x_k,
    stride_q_k, stride_q_n,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    stride_out_m, stride_out_n,
    GROUP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_packed_n = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_packed_n = offs_packed_n < (N // 8)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    shifts = tl.arange(0, 8)[None, None, :] * 4  # [1, 1, 8]

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        # Load x: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x_val = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # Load qweight: [BLOCK_SIZE_K, BLOCK_SIZE_N // 8]
        q_ptrs = qweight_ptr + offs_k[:, None] * stride_q_k + offs_packed_n[None, :] * stride_q_n
        q_val = tl.load(q_ptrs, mask=(mask_k[:, None] & mask_packed_n[None, :]), other=0)

        # Unpack qweight to [BLOCK_SIZE_K, BLOCK_SIZE_N]
        q_expanded = q_val[:, :, None]  # [BLOCK_SIZE_K, BLOCK_SIZE_N // 8, 1]
        unpacked = (q_expanded >> shifts) & 0xF
        unpacked = tl.view(unpacked, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        unpacked_fp = unpacked.to(tl.float16)

        # Load scales and zeros: [BLOCK_SIZE_N]
        group_idx = k // GROUP
        scales_ptrs = scales_ptr + group_idx * stride_scales_g + offs_n * stride_scales_n
        zeros_ptrs = zeros_ptr + group_idx * stride_zeros_g + offs_n * stride_zeros_n

        scales_val = tl.load(scales_ptrs, mask=mask_n, other=1.0)
        zeros_val = tl.load(zeros_ptrs, mask=mask_n, other=0.0)

        # Dequantize: (w - zeros) * scales
        w = (unpacked_fp - zeros_val[None, :]) * scales_val[None, :]

        # Dot product
        acc += tl.dot(x_val, w)

    # Store result
    acc = acc.to(tl.float16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qweight, scales, zeros):
        x = x.contiguous()
        qweight = qweight.contiguous()
        scales = scales.contiguous()
        zeros = zeros.contiguous()

        M, K = x.shape
        N = qweight.shape[1] * 8

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Choose block sizes dynamically based on M
        if M <= 16:
            BLOCK_SIZE_M = 16
        elif M <= 32:
            BLOCK_SIZE_M = 32
        elif M <= 64:
            BLOCK_SIZE_M = 64
        else:
            BLOCK_SIZE_M = 128

        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        GROUP = K // scales.shape[0]

        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

        w4a16_gemm_kernel[grid](
            x, qweight, scales, zeros, out,
            M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            out.stride(0), out.stride(1),
            GROUP=GROUP,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_warps=4,
        )
        return out