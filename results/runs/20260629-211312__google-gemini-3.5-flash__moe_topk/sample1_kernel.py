import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128, 'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 64, 'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 128, 'BLOCK_H': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256, 'BLOCK_H': 128}, num_warps=8),
    ],
    key=['T', 'D', 'H'],
)
@triton.jit
def moe_kernel(
    X_ptr, W_ptr, TopI_ptr, TopV_ptr, Out_ptr,
    T, D, E, H,
    stride_xt, stride_xd,
    stride_we, stride_wd, stride_wh,
    stride_topi_t, stride_topi_k,
    stride_topv_t, stride_topv_k,
    stride_out_t, stride_out_h,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Load top-2 expert indices and gate weights
    e0 = tl.load(TopI_ptr + pid_t * stride_topi_t)
    e1 = tl.load(TopI_ptr + pid_t * stride_topi_t + stride_topi_k)
    
    v0 = tl.load(TopV_ptr + pid_t * stride_topv_t).to(tl.float32)
    v1 = tl.load(TopV_ptr + pid_t * stride_topv_t + stride_topv_k).to(tl.float32)
    
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    
    acc0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    
    for d in range(0, D, BLOCK_D):
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        
        # Load X chunk
        x_ptr = X_ptr + pid_t * stride_xt + offs_d * stride_xd
        x_val = tl.load(x_ptr, mask=mask_d, other=0.0).to(tl.float32)
        
        # Load W chunk for expert 0
        w0_ptr = W_ptr + e0 * stride_we + offs_d[:, None] * stride_wd + offs_h[None, :] * stride_wh
        w0_val = tl.load(w0_ptr, mask=(mask_d[:, None] & mask_h[None, :]), other=0.0).to(tl.float32)
        
        # Load W chunk for expert 1
        w1_ptr = W_ptr + e1 * stride_we + offs_d[:, None] * stride_wd + offs_h[None, :] * stride_wh
        w1_val = tl.load(w1_ptr, mask=(mask_d[:, None] & mask_h[None, :]), other=0.0).to(tl.float32)
        
        # Elementwise-multiply and reduce along the D dimension
        acc0 += tl.sum(x_val[:, None] * w0_val, axis=0)
        acc1 += tl.sum(x_val[:, None] * w1_val, axis=0)
        
    out_val = v0 * acc0 + v1 * acc1
    
    # Store result
    out_ptr = Out_ptr + pid_t * stride_out_t + offs_h * stride_out_h
    tl.store(out_ptr, out_val.to(X_ptr.dtype_element), mask=mask_h)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, w, router_logits):
        T, D = x.shape
        E, _, H = w.shape
        
        # Compute top-2 gating weights
        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)
        
        topi = topi.to(torch.int32).contiguous()
        topv = topv.contiguous()
        
        out = torch.empty((T, H), device=x.device, dtype=x.dtype)
        
        grid = lambda META: (T, triton.cdiv(H, META['BLOCK_H']))
        
        moe_kernel[grid](
            x, w, topi, topv, out,
            T, D, E, H,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1), w.stride(2),
            topi.stride(0), topi.stride(1),
            topv.stride(0), topv.stride(1),
            out.stride(0), out.stride(1)
        )
        
        return out