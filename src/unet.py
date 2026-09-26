"""U-Net architecture for the U-Net arm (Week 1, Task 5.2 onward).

Config: see configs/unet_base.yaml (model.depth=4, model.base_channels=32,
model.pretrained=false). Deliberately vanilla, from-scratch (no pretrained
encoder) — the RQ compares zero-shot SAM 3 against a from-scratch specialist,
so a pretrained backbone here would confound that comparison (planning doc,
Idea 3.2 / §1.5).

Ronneberger et al. (2015)-style: 4 down-blocks (double-conv + maxpool),
bottleneck, 4 up-blocks (transpose-conv upsample + skip-concat + double-conv).
Base width 32 channels, doubling each down-block: 32 -> 64 -> 128 -> 256 ->
512 (bottleneck), matching configs/unet_base.yaml's model.base_channels: 32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Maxpool(2) then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Transpose-conv upsample(2), concat the matching skip connection, DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Guard against off-by-one spatial mismatches from odd input sizes.
        # configs/unet_base.yaml pins input.resize: [256, 256], so this is a
        # no-op in normal use — kept only so the model doesn't silently break
        # if that config value ever changes to something not divisible by 16.
        if x.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Vanilla U-Net, from scratch, no pretrained weights (configs/unet_base.yaml).

    Parameters
    ----------
    in_channels : int
        3 for RGB Kvasir-SEG images.
    out_channels : int
        1 — a single-channel logit mask. Sigmoid is NOT applied inside forward();
        apply it at the loss (e.g. BCEWithLogitsLoss) or at inference time, not here.
    base_channels : int
        Width of the first down-block (configs/unet_base.yaml: 32). Doubles at
        each of the 4 down-blocks: base -> 2*base -> 4*base -> 8*base -> 16*base
        (bottleneck).
    depth : int
        Number of down/up-block pairs (configs/unet_base.yaml: 4). Only depth=4
        is wired up below — the encoder/decoder channel counts are hand-written
        to that spec, not generated dynamically, so a different value raises.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        if depth != 4:
            raise NotImplementedError(
                f"UNet is hand-wired for depth=4 (configs/unet_base.yaml) — got depth={depth}."
            )

        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        self.in_conv = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)  # raw logits


if __name__ == "__main__":
    # Shape sanity check — part of Task 5.2's "print model summary; confirm
    # output shape matches mask shape."  Run with: python src/unet.py
    model = UNet(in_channels=3, out_channels=1, base_channels=32, depth=4)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Trainable parameters: {n_params:,}")
    assert out.shape == (2, 1, 256, 256), "Output shape must match mask shape (B, 1, H, W)."
    print("PASS: output shape matches mask shape.")
