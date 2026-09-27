"""A small PatchTST-style forecaster, written from scratch in PyTorch.

Idea from Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023): every feature
series is handled on its own with shared weights (channel independence), the past
window is cut into patches, each patch becomes one token, and a Transformer encoder
mixes the tokens before a linear head predicts the next values. Only the forecasting
set-up is used here, not the masked-patch pre-training.
"""
import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, d_model = x.shape
        head_dim = d_model // self.n_heads
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.n_heads, head_dim).permute(2, 0, 3, 1, 4)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(head_dim), dim=-1)
        mixed = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, tokens, d_model)
        return self.out(mixed), weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = SelfAttention(d_model, n_heads, dropout)
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, weights = self.attention(self.norm_attention(x))
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x, weights


class PatchForecaster(nn.Module):
    def __init__(self, window, horizon, patch, stride, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        if (window - patch) % stride:
            raise ValueError("patches must tile the window: (window - patch) % stride == 0")
        self.window, self.horizon, self.patch, self.stride = window, horizon, patch, stride
        self.n_patches = (window - patch) // stride + 1
        self.embed = nn.Linear(patch, d_model)
        self.position = nn.Parameter(0.02 * torch.randn(1, self.n_patches, d_model))
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, horizon)

    def forward(self, x, return_attention=False):
        """x: (batch, window, channels) -> forecast (batch, horizon, channels).

        With return_attention, also returns the attention weights as
        (batch, channels, layers, heads, patches, patches).
        """
        batch, window, channels = x.shape
        series = x.permute(0, 2, 1).reshape(batch * channels, window)
        # Only the window mean is removed: a slow drift of the level is not a surprise,
        # but a noisier, more erratic series should still give larger errors.
        level = series.mean(dim=1, keepdim=True)
        tokens = self.embed((series - level).unfold(1, self.patch, self.stride)) + self.position
        maps = []
        for layer in self.layers:
            tokens, weights = layer(tokens)
            maps.append(weights)
        forecast = self.head(self.norm(tokens).flatten(1)) + level
        forecast = forecast.view(batch, channels, self.horizon).permute(0, 2, 1)
        if not return_attention:
            return forecast
        attention = torch.stack(maps, dim=1)
        return forecast, attention.view(batch, channels, *attention.shape[1:])

    def patch_spans(self):
        """(start, stop) of every patch as offsets into the window."""
        return [(i * self.stride, i * self.stride + self.patch) for i in range(self.n_patches)]
