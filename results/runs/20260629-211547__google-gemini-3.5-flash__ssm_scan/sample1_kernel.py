import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def selective_scan_kernel(
    u_ptr, delta_ptr, A_ptr, Bm_ptr, C_ptr, y_ptr,
    B, L, D, S,
    stride_u_b, stride_u_l, stride_u_d,
    stride_delta_b, stride_delta_l, stride_delta_d,
    stride_A_d, stride_A_s,
    stride_Bm_b, stride_Bm_l, stride_Bm_s,
    stride_C_b, stride_C_l, stride_C_s,
    stride_y_b, stride_y_l, stride_y_d,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Compute offsets
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # Load A: shape [BLOCK_D, BLOCK_S]
    A_ptrs = A_ptr + d_offsets[:, None] * stride_A_d + s_offsets[None, :] * stride_A_s
    A_val = tl.load(A_ptrs, mask=(d_mask[:, None] & s_mask[None, :]), other=0.0).to(tl.float32)

    # Initialize state h: shape [BLOCK_D, BLOCK_S]
    h = tl.zeros((BLOCK_D, BLOCK_S), dtype=tl.float32)

    # Base pointers for this batch
    u_batch_ptr = u_ptr + pid_b * stride_u_b
    delta_batch_ptr = delta_ptr + pid_b * stride_delta_b
    Bm_batch_ptr = Bm_ptr + pid_b * stride_Bm_b
    C_batch_ptr = C_ptr + pid_b * stride_C_b
    y_batch_ptr = y_ptr + pid_b * stride_y_b

    for t in range(0, L):
        # Load u_t: shape [BLOCK_D]
        u_ptrs = u_batch_ptr + t * stride_u_l + d_offsets * stride_u_d
        u_t = tl.load(u_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Load delta_t: shape [BLOCK_D]
        delta_ptrs = delta_batch_ptr + t * stride_delta_l + d_offsets * stride_delta_d
        delta_t = tl.load(delta_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Load Bm_t: shape [BLOCK_S]
        Bm_ptrs = Bm_batch_ptr + t * stride_Bm_l + s_offsets * stride_Bm_s
        Bm_t = tl.load(Bm_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        # Load C_t: shape [BLOCK_S]
        C_ptrs = C_batch_ptr + t * stride_C_l + s_offsets * stride_C_s
        C_t = tl.load(C_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        # Compute dA: shape [BLOCK_D, BLOCK_S]
        dA = tl.exp(delta_t[:, None] * A_val)

        # Compute dBu: shape [BLOCK_D, BLOCK_S]
        delta_u = delta_t * u_t
        dBu = delta_u[:, None] * Bm_t[None, :]

        # Update state h
        h = dA * h + dBu

        # Compute y_t: shape [BLOCK_D]
        y_t_val = tl.sum(h * C_t[None, :], axis=1)

        # Store y_t
        y_ptrs = y_batch_ptr + t * stride_y_l + d_offsets * stride_y_d
        tl.store(y_ptrs, y_t_val, mask=d_mask)

class ModelNew(nn.Module):
    def forward(self, u, delta, A, Bm, C):
        B, L, D = u.shape
        S = A.shape[1]
        
        y = torch.empty_like(u)
        
        BLOCK_D = 32
        BLOCK_S = triton.next_power_of_2(S)
        
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
            num_warps=4,
        )
        return y