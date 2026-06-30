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
import torch.nn.functional as F

GROUP = 128

class Model(nn.Module):
    """Quantized top-2 MoE. x:[T,D]; qweight:[E,D,N] int8 (symmetric int4 in [-8,7]);
    scales:[E,D//GROUP,N] fp16; router_logits:[T,E]. Per-expert dequant (per-group along D)
    + grouped GEMM + top-2 gated combine, fused in one kernel."""
    def forward(self, x, qweight, scales, router_logits):
        T, D = x.shape
        E, _, N = qweight.shape
        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)
        G = D // GROUP
        w = (qweight.reshape(E, G, GROUP, N).to(x.dtype) * scales[:, :, None, :].to(x.dtype)).reshape(E, D, N)
        all_out = torch.einsum('td,edn->ten', x, w)
        sel = torch.gather(all_out, 1, topi.unsqueeze(-1).expand(-1, -1, N))
        return (sel * topv.unsqueeze(-1)).sum(dim=1)

def get_inputs():
    T, D, E, N = 256, 2048, 60, 1408
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    qweight = torch.randint(-8, 8, (E, D, N), device="cuda", dtype=torch.int8)
    scales = (torch.randn(E, D // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    router_logits = torch.randn(T, E, device="cuda", dtype=torch.float16)
    return [x, qweight, scales, router_logits]

```
