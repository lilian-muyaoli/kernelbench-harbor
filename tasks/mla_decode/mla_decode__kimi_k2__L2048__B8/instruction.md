# Optimize a GPU kernel

You are given a naive PyTorch reference module `Model` (in `reference.py`). Write an
optimized replacement `ModelNew` with the **same `forward` signature and outputs**.

## CRITICAL — how to submit
Your FINAL action MUST write the complete code (imports + Triton kernel + `class ModelNew`)
to **`/app/solution.py`**. Use a single shell command, e.g.:
```
cat > /app/solution.py << 'EOF'
<your full python code here>
EOF
```
The grader reads ONLY `/app/solution.py`. If you do not write this file, you score 0.

## Rules (your submission is rejected otherwise)
- `ModelNew()` must construct with **no arguments**; same `forward(...)` signature as `Model`.
- `ModelNew.forward` must compute the core work with at least one real `@triton.jit` kernel.
- Do NOT offload core compute to torch (`torch.matmul`, `torch.softmax`, `einsum`, `F.linear`, …)
  or vendor libs (`cublas`, `cudnn`, `flash_attn`, `scaled_dot_product_attention`).
- Must be numerically correct vs the reference (fp32 ground truth, dtype-aware tolerance).
- You may use torch only for tensor allocation, shape handling, and launching the kernel.

## You pass iff
correct **AND** faster than `torch.compile(Model, mode="max-autotune")` by the configured
threshold. Beating the compiler is the goal — a correct-but-slow kernel does not pass.

## The reference
```python
import math
import torch
import torch.nn as nn

H, DH, DC = 16, 64, 256

class Model(nn.Module):
    """MLA decode, naive decompress-then-attend. q:[B,H,DH]; c_kv:[B,L,DC] cached latent;
    W_uk:[DC,H*DH]; W_uv:[DC,H*DH]. Fast kernel must use the 'absorb' trick (attend in
    latent space; never materialize per-head K/V)."""
    def forward(self, q, c_kv, W_uk, W_uv):
        B, L, _ = c_kv.shape
        K = (c_kv @ W_uk).view(B, L, H, DH).permute(0, 2, 1, 3)
        V = (c_kv @ W_uv).view(B, L, H, DH).permute(0, 2, 1, 3)
        att = torch.einsum('bhd,bhld->bhl', q, K) / math.sqrt(DH)
        att = torch.softmax(att, dim=-1)
        return torch.einsum('bhl,bhld->bhd', att, V)

def get_inputs():
    B, L = 8, 2048
    q = torch.randn(B, H, DH, device="cuda", dtype=torch.float16)
    c_kv = torch.randn(B, L, DC, device="cuda", dtype=torch.float16) * 0.02
    W_uk = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    W_uv = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    return [q, c_kv, W_uk, W_uv]

```
