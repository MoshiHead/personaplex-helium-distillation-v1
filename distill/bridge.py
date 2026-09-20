# SPDX-License-Identifier: MIT
"""The Bridge: the only new module standing between the student temporal
transformer and the frozen teacher depth transformer.

Bridge(x) stands in for the teacher's raw temporal-transformer output (before
`out_norm`), so everything downstream -- the teacher's `out_norm`, `text_linear`,
`depformer_in` (16 per-codebook projections), `depformer`, and `linears` -- is
reused completely unmodified. See distill/init_from_teacher.py for why the
bridge maps to the teacher's 4096-dim space rather than to `depformer_dim`
(1024): the latter would require bypassing/discarding the teacher's 16 frozen
per-codebook `depformer_in` projections.
"""

import torch
import torch.nn as nn

from moshi.modules.transformer import create_norm_fn


class Bridge(nn.Module):
    """Linear(in_dim -> out_dim) + RMSNorm, new and learned."""

    def __init__(self, in_dim: int, out_dim: int, device=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False, device=device, dtype=dtype)
        self.norm = create_norm_fn("rms_norm_f32", out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))
