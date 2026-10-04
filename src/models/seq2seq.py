"""Small CNN/Transformer image-to-LaTeX model."""

from __future__ import annotations

import math

import torch
from torch import nn


class CNNEncoder(nn.Module):
    """Convert an RGB image into a sequence of visual tokens."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, d_model, 3, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = features + self._position_encoding(features)
        return features.flatten(2).transpose(1, 2)

    @staticmethod
    def _position_encoding(features: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = features.shape
        device, dtype = features.device, features.dtype
        rows = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        columns = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        channel_ids = torch.arange(channels, device=device)
        axes = (channel_ids // 2) % 2
        frequencies = channel_ids // 4
        scales = torch.exp(-math.log(10000) * frequencies.to(dtype) / max(channels // 4, 1))
        row_grid = rows[:, None].expand(height, width)
        column_grid = columns[None, :].expand(height, width)
        coordinates = torch.where(axes[:, None, None] == 0, row_grid[None], column_grid[None])
        angles = coordinates * scales[:, None, None] * math.pi
        return torch.where(channel_ids[:, None, None] % 2 == 0, angles.sin(), angles.cos()).unsqueeze(0) * 0.1


class ImageToLatex(nn.Module):
    """Autoregressive Transformer decoder over CNN image features."""

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, layers: int = 2, ff_dim: int = 256, max_length: int = 160) -> None:
        super().__init__()
        self.d_model, self.max_length = d_model, max_length
        self.encoder = CNNEncoder(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(images)
        positions = torch.arange(target.size(1), device=target.device).clamp(max=self.max_length - 1)
        embedded = self.embedding(target) * math.sqrt(self.d_model) + self.position(positions)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(target.size(1), device=target.device)
        decoded = self.decoder(embedded, memory, tgt_mask=mask)
        return self.output(decoded)

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor, sos_id: int, eos_id: int, max_length: int | None = None) -> torch.Tensor:
        memory = self.encoder(images)
        tokens = torch.full((images.size(0), 1), sos_id, dtype=torch.long, device=images.device)
        limit = max_length or self.max_length
        for _ in range(limit - 1):
            positions = torch.arange(tokens.size(1), device=images.device).clamp(max=self.max_length - 1)
            embedded = self.embedding(tokens) * math.sqrt(self.d_model) + self.position(positions)[None]
            mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=images.device)
            next_token = self.output(self.decoder(embedded, memory, tgt_mask=mask))[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if bool((next_token == eos_id).all()):
                break
        return tokens
