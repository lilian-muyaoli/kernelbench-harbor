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

TOPK = 6

class Model(nn.Module):
    """Top-6 mixture-of-experts FFN. x:[T,D]; w:[E,D,H]; router_logits:[T,E].
    out[t] = sum over its top-6 experts e of gate[t,e] * (x[t] @ w[e]).
    A fast kernel must do sparse dispatch instead of computing all E experts."""
    def forward(self, x, w, router_logits):
        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(TOPK, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)
        all_out = torch.einsum('td,edh->teh', x, w)
        H = w.shape[2]
        sel = torch.gather(all_out, 1, topi.unsqueeze(-1).expand(-1, -1, H))
        return (sel * topv.unsqueeze(-1)).sum(dim=1)

def get_inputs():
    T, D, E, H = 128, 2048, 64, 1408
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    w = (torch.randn(E, D, H, device="cuda", dtype=torch.float16) * 0.02)
    router_logits = torch.randn(T, E, device="cuda", dtype=torch.float16)
    return [x, w, router_logits]

```
