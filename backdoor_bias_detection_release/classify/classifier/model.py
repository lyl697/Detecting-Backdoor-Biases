"""Shared block-channel CNN-Transformer detector architecture."""

import torch
from torch import nn

class AblationBlockChannelDetector(nn.Module):
    """Original block-channel detector with an optional LFD branch."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_blocks: int,
        extra_dim: int,
        num_steps: int,
        use_lfd: bool = True,
        token_dim: int = 128,
        cnn_dim: int = 128,
        cnn_token_bins: int = 8,
        transformer_layers: int = 2,
        nhead: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if cnn_dim % nhead != 0:
            raise ValueError("--cnn_dim must be divisible by --nhead")

        self.use_lfd = bool(use_lfd)
        self.extra_dim = int(extra_dim)
        self.num_steps = int(num_steps)
        self.hidden_blocks = int(hidden_blocks)
        self.cnn_token_bins = int(cnn_token_bins)

        if not self.use_lfd and self.extra_dim <= 0:
            raise ValueError("At least one feature branch must be enabled")

        self.total_blocks = (
            (self.hidden_blocks if self.use_lfd else 0)
            + (1 if self.extra_dim > 0 else 0)
        )

        self.hidden_proj = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) if self.use_lfd else None
        )

        self.extra_proj = (
            nn.Sequential(
                nn.LayerNorm(extra_dim),
                nn.Linear(extra_dim, token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) if extra_dim > 0 else None
        )

        self.step_embedding = nn.Parameter(torch.zeros(1, num_steps, 1, token_dim))
        self.block_embedding = nn.Parameter(torch.zeros(1, 1, self.total_blocks, token_dim))

        self.cnn = nn.Sequential(
            nn.Conv2d(self.total_blocks, cnn_dim, kernel_size=(3, 7), padding=(1, 3)),
            nn.GroupNorm(num_groups=1, num_channels=cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=(3, 5), padding=(1, 2)),
            nn.GroupNorm(num_groups=1, num_channels=cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=(1, 1)),
            nn.GroupNorm(num_groups=1, num_channels=cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((num_steps, cnn_token_bins))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_dim,
            nhead=nhead,
            dim_feedforward=cnn_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cnn_dim))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cnn_dim),
            nn.Linear(cnn_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 2),
        )

    def forward(self, hidden: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        token_parts = []

        if self.use_lfd:
            token_parts.append(self.hidden_proj(hidden))

        if self.extra_proj is not None:
            token_parts.append(self.extra_proj(extra).unsqueeze(2))

        if not token_parts:
            raise RuntimeError("No feature token is enabled")

        tokens = torch.cat(token_parts, dim=2)
        tokens = tokens + self.step_embedding[:, :tokens.shape[1], :, :]
        tokens = tokens + self.block_embedding[:, :, :tokens.shape[2], :]

        # [B, step, feature_block, token_dim] -> [B, feature_block, step, token_dim]
        tokens = tokens.permute(0, 2, 1, 3)
        tokens = self.cnn(tokens)
        tokens = self.pool(tokens)

        # [B, cnn_dim, step, bins] -> [B, step*bins, cnn_dim]
        batch_size, cnn_dim = tokens.shape[0], tokens.shape[1]
        tokens = tokens.permute(0, 2, 3, 1).reshape(batch_size, -1, cnn_dim)
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.classifier(encoded[:, 0])

