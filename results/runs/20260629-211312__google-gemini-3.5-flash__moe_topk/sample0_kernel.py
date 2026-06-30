An elegant and highly optimized Triton implementation for the Top-2 Mixture-of-Experts (MoE) layer. 

Rather than computing the dense matrix multiplication for all experts (which does unnecessary computation for inactive experts), this implementation groups the active tokens by their routed expert using fast GPU sorting, and then executes a **Grouped GEMM** via a custom Triton kernel. This reduces the computational complexity by **4x** (from 8 experts to 2 experts per token) and achieves maximum memory and tensor core efficiency.

Here is the complete and self-contained code:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def moe_grouped_gemm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    sorted_t_ptr, sorted_topv_ptr,
    offsets_ptr, start_block_idx_ptr,
    T, D, H,
    stride_xt, stride_xd,
    stride_we, stride_wd, stride_wh,
    stride_outt, stride_outh,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 1. Map pid to the corresponding expert e
    e = 0
    for i in range(1, E):
        start_idx = tl.load(start_block_idx_ptr + i)
        if pid >= start_idx:
            e = i
            
    start_block = tl.load(start_block_idx_ptr + e)
    pid_in_expert = pid - start_block
    
    num_h_blocks = H // BLOCK_N
    block_m = pid_in_expert // num_h_blocks
    block_n = pid_in_expert % num_h_blocks
    
    # 2. Get the range of tokens for this expert e
    offset_start = tl.load(offsets_ptr + e)
    offset_end = tl.load(offsets_ptr + e + 1)
    
    token_offs = offset_start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_offs < offset_end
    
    # Load the actual token indices and weights
    t_indices = tl.load(sorted_t_ptr + token_offs, mask=token_mask, other=0)
    weights = tl.load(sorted_topv_ptr + token_offs, mask=token_mask, other=0.0)
    
    # 3. Compute GEMM
    h_offs = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc =