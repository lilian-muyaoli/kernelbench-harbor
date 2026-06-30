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

BLOCK = 32

class Model(nn.Module):
    """FP8 block-microscaled weight GEMM. x:[M,K] float16; w_fp8:[K,N] float8_e4m3fn;
    scales:[K//BLOCK,N] fp16.  w[k,n]=w_fp8[k,n]*scales[k//BLOCK,n] ; out=x@w."""
    def forward(self, x, w_fp8, scales):
        K, N = w_fp8.shape
        w = w_fp8.float().reshape(K // BLOCK, BLOCK, N) * scales[:, None, :].float()
        return torch.matmul(x, w.reshape(K, N).to(x.dtype))

def get_inputs():
    M, K, N = 16, 29568, 8192
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp8 = (torch.randn(K, N, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = (torch.rand(K // BLOCK, N, device="cuda") * 0.05 + 0.05).to(torch.float16)
    return [x, w_fp8, scales]

```
