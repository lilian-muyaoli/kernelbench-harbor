import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128

@triton.jit
def gemm_dequant_kernel(
    x_ptr, qweight_ptr, scales_ptr, zeros_ptr, out_ptr,
    M, N, K,
    stride_x_m, stride_x_k,
    stride_q_k, stride_q_n,
    stride_scale_g, stride_scale_n,
    stride_zero_g, stride_zero_n,
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
    
    offs_m_clamped = tl.minimum(offs_m, M - 1)
    offs_n_clamped = tl.minimum(offs_n, N - 1)
    offs_n_packed_clamped = tl.minimum(offs_n // 8, (N // 8) - 1)
    
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute shifts outside the loop since they only depend on N
    shifts = (offs_n % 8) * 4
    
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        offs_k_clamped = tl.minimum(offs_k, K - 1)
        
        # Load X block
        x_ptrs = x_ptr + offs_m_clamped[:, None] * stride_x_m + offs_k_clamped[None, :] * stride_x_k
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Load packed qweight block
        q_ptrs = qweight_ptr + offs_k_clamped[:, None] * stride_q_k + offs_n_packed_clamped[None, :] * stride_q_n
        mask_q = (offs_k[:, None] < K) & ((offs_n // 8)[None, :] < (N // 8))
        q_packed = tl.load(q_ptrs, mask=mask_q, other=0)
        
        # Unpack 4-bit weights
        w = (q_packed >> shifts[None, :]) & 0xF
        
        # Load scales and zeros
        offs_g = offs_k // GROUP
        offs_g_clamped = tl.minimum(offs_g, (K // GROUP) - 1)
        
        scale_ptrs = scales_ptr + offs_g_clamped[:, None] * stride_scale_g + offs_n_clamped[None, :] * stride_scale_n
        zero_ptrs = zeros_ptr + offs_g_clamped[:, None] * stride_zero_g + offs_n_clamped[None, :] * stride_zero_n
        
        mask_sz = (offs_g[:, None] < (K // GROUP)) & (offs_n[None, :] < N)
        scale_val = tl.load(scale_ptrs, mask=mask_sz, other=0.0)
        zero_val = tl.load(zero_ptrs, mask=mask_sz, other=0.0)
        
        # Dequantize
        w_dequant = (w.to(x_val.dtype) - zero_val.to(x_val.dtype)) * scale_val.to(x_val.dtype)
        
        # Accumulate dot product
        accum = tl.dot(x_val.to(tl.float16), w_dequant.to(tl.float16), accum)
        
    out_ptrs = out_ptr + offs_m_clamped[:, None] * stride_out_m + offs_n_clamped[None, :] * stride_out_n
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accum.to(x_val.dtype), mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qweight, scales, zeros):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, K = x_2d.shape
        N = qweight.shape[1] * 8
        
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        # Choose BLOCK_SIZE_M dynamically based on M
        if M <= 16:
            BLOCK_SIZE_M = 16
        elif M <= 32:
            BLOCK_SIZE_M = 32
        elif M <= 64:
            BLOCK_SIZE_M = 64
        else:
            BLOCK_SIZE_M = 128
            
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N'])
        )
        
        gemm_dequant_kernel[grid](
            x_2d, qweight, scales, zeros, out,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            out.stride(0), out.stride(1),
            GROUP=GROUP,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            num_warps=4,
        )
        
        return out.reshape(orig_shape[:-1] + (N,))