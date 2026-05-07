"""
model.py
--------
Monocular Depth Estimation Network
====================================
Architecture: ResNet34 Encoder  +  U-Net style Decoder

┌──────────────────────────────────────────────────────────────────────┐
│  INPUT  RGB image  (3 × H × W)                                       │
│                                                                      │
│  ENCODER  (pretrained ResNet-34)                                     │
│  ─────────────────────────────                                       │
│  Layer 0 : Conv 7×7, s=2, BN, ReLU  →  (64,  H/2,  W/2)  ← skip 0  │
│  Layer 1 : MaxPool                  →  (64,  H/4,  W/4)             │
│  Layer 2 : ResBlock × 3             →  (64,  H/4,  W/4)  ← skip 1  │
│  Layer 3 : ResBlock × 4 (s=2)       →  (128, H/8,  W/8)  ← skip 2  │
│  Layer 4 : ResBlock × 6 (s=2)       →  (256, H/16, W/16) ← skip 3  │
│  Layer 5 : ResBlock × 3 (s=2)       →  (512, H/32, W/32) ← bottleneck│
│                                                                      │
│  DECODER  (Upsampling + Skip connections)                            │
│  ─────────────────────────────────────────                           │
│  Up-block 1: bilinear ×2 + skip 3   →  (256, H/16, W/16)            │
│  Up-block 2: bilinear ×2 + skip 2   →  (128, H/8,  W/8)             │
│  Up-block 3: bilinear ×2 + skip 1   →  (64,  H/4,  W/4)             │
│  Up-block 4: bilinear ×2 + skip 0   →  (64,  H/2,  W/2)             │
│  Up-block 5: bilinear ×2             →  (32,  H,    W)               │
│                                                                      │
│  HEAD  1×1 Conv + Sigmoid × MAX_DEPTH  →  (1, H, W)  depth map      │
└──────────────────────────────────────────────────────────────────────┘

Skip connections bring fine spatial details from the encoder back into
the decoder, preventing blurry, over-smoothed depth predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


MAX_DEPTH = 10.0   # metres (NYU indoor)


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsample by 2× (bilinear), concatenate skip connection, then
    apply two Conv-BN-ReLU layers to fuse the features.

    Parameters
    ----------
    in_ch   : channels coming from below (deeper decoder)
    skip_ch : channels from the encoder skip connection
    out_ch  : output channels
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate skip
        if skip is not None:
            # Handle slight size mismatch (can happen at borders)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ─────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────

class DepthEstimationNet(nn.Module):
    """
    Monocular Depth Estimation Network.

    Parameters
    ----------
    backbone : str
        'resnet18' or 'resnet34'  (both have the same channel layout)
    pretrained : bool
        Whether to initialise the encoder with ImageNet weights.
    """

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()

        # ── Encoder (ResNet backbone) ──────────────────────────────────
        if backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        elif backbone == "resnet34":
            resnet = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18' or 'resnet34'.")

        # Decompose ResNet into named stages for easy skip-connection access
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # (64, H/2, W/2)
        self.pool = resnet.maxpool                                          # (64, H/4, W/4)
        self.enc1 = resnet.layer1                                           # (64, H/4, W/4)
        self.enc2 = resnet.layer2                                           # (128, H/8, W/8)
        self.enc3 = resnet.layer3                                           # (256, H/16, W/16)
        self.enc4 = resnet.layer4                                           # (512, H/32, W/32)

        # ── Decoder ───────────────────────────────────────────────────
        # Each DecoderBlock(in_ch, skip_ch, out_ch)
        self.dec4 = DecoderBlock(512, 256, 256)   # fuse enc3 skip
        self.dec3 = DecoderBlock(256, 128, 128)   # fuse enc2 skip
        self.dec2 = DecoderBlock(128,  64,  64)   # fuse enc1 skip
        self.dec1 = DecoderBlock( 64,  64,  64)   # fuse enc0 skip
        self.dec0 = DecoderBlock( 64,   0,  32)   # no skip – restore full res

        # ── Output head ───────────────────────────────────────────────
        # 1×1 conv to single channel, then sigmoid scaled to [0, MAX_DEPTH]
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, 3, H, W)  normalised RGB

        Returns
        -------
        depth : Tensor (B, 1, H, W)  predicted depth in metres [0, MAX_DEPTH]
        """
        # ── Encoder forward pass ──────────────────────────────────────
        s0 = self.enc0(x)       # (B, 64,  H/2,  W/2)
        s1 = self.enc1(self.pool(s0))  # (B, 64,  H/4,  W/4)
        s2 = self.enc2(s1)      # (B, 128, H/8,  W/8)
        s3 = self.enc3(s2)      # (B, 256, H/16, W/16)
        s4 = self.enc4(s3)      # (B, 512, H/32, W/32)

        # ── Decoder forward pass with skip connections ─────────────────
        d  = self.dec4(s4, s3)  # (B, 256, H/16, W/16)
        d  = self.dec3(d,  s2)  # (B, 128, H/8,  W/8)
        d  = self.dec2(d,  s1)  # (B, 64,  H/4,  W/4)
        d  = self.dec1(d,  s0)  # (B, 64,  H/2,  W/2)
        d  = self.dec0(d)       # (B, 32,  H,    W)

        # ── Output ────────────────────────────────────────────────────
        depth = self.head(d) * MAX_DEPTH   # scale sigmoid to metres

        return depth

    # ── Utility ───────────────────────────────────────────────────────
    def count_parameters(self):
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────
# Quick architecture test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DepthEstimationNet(backbone="resnet34", pretrained=False).to(device)

    total, trainable = model.count_parameters()
    print(f"\n{'='*55}")
    print(f"  Depth Estimation Network  (ResNet-34 + U-Net Decoder)")
    print(f"{'='*55}")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"{'='*55}\n")

    # Forward pass test
    dummy = torch.randn(2, 3, 240, 320).to(device)
    out   = model(dummy)
    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Depth  range : [{out.min().item():.4f}, {out.max().item():.4f}] metres")
