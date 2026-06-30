import torch
import torch.nn as nn

class Model(nn.Module):
    """W8A8 SmoothQuant linear. x_int8:[M,K] int8; w_int8:[K,N] int8;
    x_scale:[M,1] fp16 (per-token); w_scale:[1,N] fp16 (per-channel). out fp16."""
    def forward(self, x_int8, w_int8, x_scale, w_scale):
        acc = torch.matmul(x_int8.float(), w_int8.float())
        return (acc * x_scale.float() * w_scale.float()).to(torch.float16)

def get_inputs():
    M, K, N = 256, 3584, 18944
    x_int8 = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    w_int8 = torch.randint(-127, 128, (K, N), device="cuda", dtype=torch.int8)
    x_scale = (torch.rand(M, 1, device="cuda") * 0.01 + 0.001).to(torch.float16)
    w_scale = (torch.rand(1, N, device="cuda") * 0.01 + 0.001).to(torch.float16)
    return [x_int8, w_int8, x_scale, w_scale]
