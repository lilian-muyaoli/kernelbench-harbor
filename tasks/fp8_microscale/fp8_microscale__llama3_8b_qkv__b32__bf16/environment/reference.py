import torch
import torch.nn as nn

BLOCK = 32

class Model(nn.Module):
    """FP8 block-microscaled weight GEMM. x:[M,K] bfloat16; w_fp8:[K,N] float8_e4m3fn;
    scales:[K//BLOCK,N] fp16.  w[k,n]=w_fp8[k,n]*scales[k//BLOCK,n] ; out=x@w."""
    def forward(self, x, w_fp8, scales):
        K, N = w_fp8.shape
        w = w_fp8.float().reshape(K // BLOCK, BLOCK, N) * scales[:, None, :].float()
        return torch.matmul(x, w.reshape(K, N).to(x.dtype))

def get_inputs():
    M, K, N = 16, 4096, 6144
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_fp8 = (torch.randn(K, N, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = (torch.rand(K // BLOCK, N, device="cuda") * 0.05 + 0.05).to(torch.float16)
    return [x, w_fp8, scales]
