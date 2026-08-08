# src/models/dlinknet.py
"""D-LinkNet (Zhou et al., CVPR-W 2018) - winner of the DeepGlobe Road Extraction
challenge.

The idea in one paragraph. Roads are thin, long and connected. A plain encoder
downsamples 32x, by which point a 3-pixel-wide road is sub-pixel, and the
receptive field of the bottleneck still may not span a road that crosses the
whole tile. D-LinkNet inserts a *dilated* block at the bottleneck: parallel
convolution branches with dilation 1/2/4/8 stacked in a cascade, which widens
the receptive field to cover the full tile WITHOUT another downsample and
without extra resolution loss. The LinkNet decoder (additive skips, not
concatenation) then keeps the parameter count low.

Worth knowing for the eventual CCTA pivot: vessels are the same problem class -
thin, elongated, sparse, connectivity-critical - so this is the family of
architecture to reach for there too, not a road-specific trick.

Reference: https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html
"""
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder


class DilatedBlock(nn.Module):
    """The "D" in D-LinkNet: cascaded dilated 3x3 convs with rates 1, 2, 4, 8.

    Each branch is the running sum of all shallower branches, so the block sees
    receptive fields of 3, 7, 15 and 31 at the bottleneck stride and adds them
    together residually. At stride 32 with a 256px input that is up to ~992px of
    effective context - i.e. the whole tile.
    """

    def __init__(self, channels: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, dilation=d, padding=d)
            for d in dilations
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        acc = x
        for branch in self.branches:
            out = self.relu(branch(out))
            acc = acc + out
        return acc


class DecoderBlock(nn.Module):
    """LinkNet decoder block: squeeze to 1/4 channels, upsample 2x with a
    transposed conv, expand. The squeeze is what keeps LinkNet cheap relative to
    a U-Net decoder of the same width."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DLinkNet(nn.Module):
    """D-LinkNet with a swappable smp encoder (default ResNet-34, as published).

    Uses segmentation_models_pytorch's encoder zoo rather than a hand-rolled
    ResNet so the ImageNet weights and the feature-stride contract match every
    other model in this repo - the comparison then isolates the decoder and
    bottleneck, which is the thing being tested.
    """

    def __init__(self, encoder_name: str = "resnet34", encoder_weights="imagenet", classes: int = 1):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)

        # smp encoders return 6 feature maps at strides 1, 2, 4, 8, 16, 32.
        _, _, c4, c8, c16, c32 = self.encoder.out_channels

        self.center = DilatedBlock(c32)
        self.decoder4 = DecoderBlock(c32, c16)
        self.decoder3 = DecoderBlock(c16, c8)
        self.decoder2 = DecoderBlock(c8, c4)
        self.decoder1 = DecoderBlock(c4, c4)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(c4, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        _, _, e4, e8, e16, e32 = self.encoder(x)

        c = self.center(e32)
        d4 = self.decoder4(c) + e16      # additive skips - LinkNet, not U-Net concat
        d3 = self.decoder3(d4) + e8
        d2 = self.decoder2(d3) + e4
        d1 = self.decoder1(d2)           # stride 4 -> stride 2
        return self.head(d1)             # stride 2 -> stride 1, returns logits


if __name__ == "__main__":  # quick shape check: python -m src.models.dlinknet
    model = DLinkNet(encoder_weights=None)
    out = model(torch.randn(2, 3, 256, 256))
    print("output:", tuple(out.shape), "params:", sum(p.numel() for p in model.parameters()))
