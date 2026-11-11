"""Conventional 2D U-Net (four encoder levels, bottleneck, symmetric decoder)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers with dropout in between."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Transposed-convolution upsampling followed by a skip-connection ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, dropout: float = 0.20) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c, dropout)
        self.enc2 = ConvBlock(c, 2 * c, dropout)
        self.enc3 = ConvBlock(2 * c, 4 * c, dropout)
        self.enc4 = ConvBlock(4 * c, 8 * c, dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(8 * c, 16 * c, dropout)
        self.up4 = UpBlock(16 * c, 8 * c, 8 * c, dropout)
        self.up3 = UpBlock(8 * c, 4 * c, 4 * c, dropout)
        self.up2 = UpBlock(4 * c, 2 * c, 2 * c, dropout)
        self.up1 = UpBlock(2 * c, c, c, dropout)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return segmentation logits."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d = self.up4(b, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        return self.head(d)
