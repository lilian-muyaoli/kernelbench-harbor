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
    B, L, D, S = 2, 256, 4096, 16
    u = torch.randn(B, L, D, device="cuda", dtype=torch.float16)
    delta = torch.nn.functional.softplus(torch.randn(B, L, D, device="cuda", dtype=torch.float16))
    A = (-torch.rand(D, S, device="cuda", dtype=torch.float16) - 0.5)
    Bm = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    C = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    return [u, delta, A, Bm, C]
