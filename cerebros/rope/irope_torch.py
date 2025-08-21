"""
PyTorch Rotary Positional Embedding (RoPE/iRoPE) shim for Cerebros.
This is a minimal adapter; replace internals with your preferred implementation.
"""
from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if torch is None:
            raise RuntimeError("PyTorch not available")
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        # x: [batch, seq, heads, head_dim] or [seq, batch, heads, head_dim]
        if x.dim() < 3:
            return x
        seq_dim = 1 if x.dim() == 4 else 0
        t = torch.arange(seq_len or x.shape[seq_dim], device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
        cos = emb.cos()[..., None]
        sin = emb.sin()[..., None]
        # Apply rotation on last dim (head_dim)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (x_rot * sin)

