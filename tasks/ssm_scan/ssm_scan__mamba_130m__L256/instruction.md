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
import torch
import torch.nn as nn

class Model(nn.Module):
    """Mamba selective scan. u,delta:[B,L,D]; A:[D,S]; Bm,C:[B,L,S]. y:[B,L,D]."""
    def forward(self, u, delta, A, Bm, C):
        Bb, L, D = u.shape
        S = A.shape[1]
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dBu = (delta.unsqueeze(-1) * Bm.unsqueeze(2)) * u.unsqueeze(-1)
        h = torch.zeros(Bb, D, S, device=u.device, dtype=torch.float32)
        ys = []
        for t in range(L):
            h = dA[:, t].float() * h + dBu[:, t].float()
            ys.append((h * C[:, t].unsqueeze(1).float()).sum(-1))
        return torch.stack(ys, dim=1).to(u.dtype)

def get_inputs():
    B, L, D, S = 4, 256, 1536, 16
    u = torch.randn(B, L, D, device="cuda", dtype=torch.float16)
    delta = torch.nn.functional.softplus(torch.randn(B, L, D, device="cuda", dtype=torch.float16))
    A = (-torch.rand(D, S, device="cuda", dtype=torch.float16) - 0.5)
    Bm = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    C = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    return [u, delta, A, Bm, C]

```
