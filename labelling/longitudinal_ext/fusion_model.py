import torch
import torch.nn as nn


class PairedFusionNet(nn.Module):
    """encoder(prior), encoder(current) -> image features; optional MIMIC-IV tabular MLP; shared head -> [B, 5, 3].
    img_mode: 'diff' (current - prior, as in the existing model) | 'diff_cur' | 'all' (prior, current, diff).
    encoder=None gives a tabular-only network (same head, same training code)."""

    def __init__(self, encoder=None, n_tab=0, n_diseases=5, n_classes=3, img_mode="diff",
                 tab_dim=64, hidden=256, dropout=0.3, tab_input_dropout=0.1):
        super().__init__()
        assert encoder is not None or n_tab > 0, "need images and/or tabular features"
        self.encoder, self.img_mode = encoder, img_mode
        self.n_diseases, self.n_classes = n_diseases, n_classes
        d = 0
        if encoder is not None:
            mult = {"diff": 1, "diff_cur": 2, "all": 3}[img_mode]
            d += encoder.out_dim * mult
        self.tab = None
        if n_tab > 0:
            self.tab = nn.Sequential(nn.Dropout(tab_input_dropout), nn.Linear(n_tab, tab_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(tab_dim, tab_dim), nn.GELU())
            d += tab_dim
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_diseases * n_classes))

    def head_parameters(self):
        mods = [self.norm, self.head] + ([self.tab] if self.tab is not None else [])
        return [p for m in mods for p in m.parameters()]

    def forward(self, x_prev=None, x_cur=None, tab=None):
        parts = []
        if self.encoder is not None:
            f = self.encoder(torch.cat([x_prev, x_cur], 0))
            f0, f1 = f.chunk(2)
            parts.append({"diff": [f1 - f0], "diff_cur": [f1 - f0, f1], "all": [f0, f1, f1 - f0]}[self.img_mode])
            parts = [t for g in parts for t in g]
        if self.tab is not None:
            parts.append(self.tab(tab))
        z = self.norm(torch.cat(parts, dim=1))
        return self.head(z).view(-1, self.n_diseases, self.n_classes)
