import torch
import torch.nn as nn
import triton
import triton.language as tl

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

@triton.jit
def selective_scan_kernel(
    u_ptr, delta_ptr, A_ptr, Bm_ptr, C_ptr, y_ptr,
    B, L, D, S,
    u_stride_b, u_stride_l, u_stride_d,
    delta_stride_b, delta_stride_l, delta_stride_d,
    A_stride_d, A_stride_s,
    Bm_stride_b, Bm_stride_l, Bm_stride_s,
    C_stride_b, C_stride_l, C_stride_s,
    y_stride_b, y_stride_l, y_stride_d,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    
    # Load A: shape [BLOCK_D, BLOCK_S]
    A_ptrs = A_ptr + offs_d[:, None] * A_stride_d + offs_s[None, :] * A_stride_s
    A_mask = (offs_d[:, None] < D) & (offs_s[None, :] < S)
    a_val = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)
    
    # Initialize h: shape [BLOCK_D, BLOCK_S]
    h = tl.zeros((BLOCK_D, BLOCK_S), dtype=tl.float32)
    
    # Pre-compute base pointers for this batch
    u_base = u_ptr + pid_b * u_stride_b
    delta_base = delta_ptr + pid_b * delta_stride_b
    Bm_base = Bm_ptr + pid_b * Bm_stride_b
    C_base = C_ptr + pid_b * C_stride_b
    y_base = y_ptr + pid_b * y_stride_b
    
    for t in range(0, L):
        # Load delta: shape [BLOCK_D]
        delta_ptrs = delta_base + t * delta_stride_l + offs_d * delta_stride_d
        delta_val = tl.load(delta_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        delta_val = delta_val[:, None] # [BLOCK_D, 1]
        
        # dA = exp(delta * A)
        dA = tl.exp(delta_val * a_val) # [BLOCK_D, BLOCK_S]
        
        # Load u: shape [BLOCK_D]
        u_ptrs = u_base + t * u_stride_l + offs_d * u_stride_d
        u_val = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        u_val = u_val[:, None] # [BLOCK_D, 1]
        
        # Load Bm: shape [BLOCK_S]
        bm_ptrs = Bm_base + t * Bm_stride_l + offs_s * Bm_stride_s
        bm_val = tl.load(bm_ptrs, mask=mask_s, other=0.0).to(tl.float32)
        bm_val = bm_val[None, :] # [1, BLOCK_S]
        
        # dBu = delta * Bm * u: shape [BLOCK_D, BLOCK_S]
        dBu = (delta_val * bm_val) * u_val
        
        # h = dA * h + dBu
        h = dA * h + dBu
        
        # Load C: shape [BLOCK_S]
        c_ptrs = C_base + t * C_stride_l + offs_s * C_stride_s
        c_val = tl.load(c_ptrs, mask=mask_s, other=0.0).to(tl.float32)
        c_val = c_val[None, :] # [1, BLOCK_S]
        
        # y = sum(h * C, axis=1)
        y_val = tl.sum(h * c_val, axis=1) # [BLOCK_D]
        
        # Store y
        y_ptrs = y_base + t * y_stride_l + offs_d * y_stride_d
        tl.store(y_ptrs, y_val.to(y_ptr.dtype.element_ty), mask=mask_d)

class ModelNew(nn.Module):
    """Optimized selective state-space scan using a custom Triton kernel."""
    def forward(self, u, delta, A, Bm, C):
        B, L, D = u.shape
        S = A.shape[1]
        
        y = torch.empty_like(u)
        
        BLOCK_D = 64
        BLOCK_S = next_power_of_2(S)
        
        grid = (B, triton.cdiv(D, BLOCK_D))
        
        selective_scan_kernel[grid](
            u, delta, A, Bm, C, y,
            B, L, D, S,
            u.stride(0), u.stride(1), u.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            Bm.stride(0), Bm.stride(1), Bm.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_D=BLOCK_D,
            BLOCK_S=BLOCK_S,
            num_warps=4
        )
        
        return y