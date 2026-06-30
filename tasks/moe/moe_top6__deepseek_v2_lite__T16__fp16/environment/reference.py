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
    T, D, E, H = 16, 2048, 64, 1408
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    w = (torch.randn(E, D, H, device="cuda", dtype=torch.float16) * 0.02)
    router_logits = torch.randn(T, E, device="cuda", dtype=torch.float16)
    return [x, w, router_logits]
