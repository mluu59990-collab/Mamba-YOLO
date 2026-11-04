"""YOLOv12 adaptations, not the original papers' detector architectures.

AKC equations (6)-(12) and Figure 3 are implemented explicitly. Scan machinery,
LSBlock and RGBlock derive from the supplied HZAI-ZJNU/Mamba-YOLO source.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv
from .mamba_yolo import SS2D, VSSBlock, XSSBlock
from .common_utils_mbyolo import LayerNorm2d


class AKConv(nn.Module):
    """N learned sample offsets, bilinear border sampling, N-point convolution.

    Uses the released AKConv's uncentred initial sample lattice. grid_sample
    avoids duplicated boundary weights in the supplied four-corner sampler.
    Offsets start at zero (both weights AND bias).
    """
    def __init__(self, channels, points=5):
        super().__init__()
        self.points = points
        self.offset = nn.Conv2d(channels, 2 * points, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        width = round(math.sqrt(points))
        lattice = [(i // width, i % width) for i in range(points)]
        self.register_buffer("lattice", torch.tensor(lattice, dtype=torch.float32))
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, (points, 1), stride=(points, 1), bias=False),
            nn.BatchNorm2d(channels), nn.SiLU())

    def sample(self, x):
        b, c, h, w = x.shape
        # Keep coordinates and interpolation in FP32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            offsets = F.conv2d(x.float(), self.offset.weight.float(),
                               self.offset.bias.float(), padding=1).view(b, 2, self.points, h, w)
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device),
                                    torch.arange(w, device=x.device), indexing="ij")
            y = yy + self.lattice[:, 0, None, None] + offsets[:, 0]
            z = xx + self.lattice[:, 1, None, None] + offsets[:, 1]
            grid = torch.stack((2 * (z + .5) / w - 1, 2 * (y + .5) / h - 1), -1)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(b, h * self.points, w, 2)
            return F.grid_sample(x.float(), grid, padding_mode="border", align_corners=False)

    def forward(self, x):
        return self.conv(self.sample(x).to(x.dtype))


class AKCBlock(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.body = nn.Sequential(AKConv(c), Conv(c, c, 1), Conv(c, c, 1))
        self.shortcut = shortcut

    def forward(self, x):
        y = self.body(x)
        return y + x if self.shortcut else y


class CAKC(nn.Module):
    """Equations 7/8: three/four outer convolutions, residual addition."""
    def __init__(self, c, four=False):
        super().__init__()
        self.branch = nn.Sequential(Conv(c, c, 1), AKCBlock(c), Conv(c, c, 1),
                                    Conv(c, c, 1) if four else nn.Identity())
        self.skip = Conv(c, c, 1)

    def forward(self, x):
        return self.branch(x) + self.skip(x)


class AKSS2D(SS2D):
    """Figure 3(a): linear -> AKConv -> four S6 scans -> LN -> linear."""
    def __init__(self, c):
        super().__init__(d_model=c, forward_type="v2noz")
        self.conv2d = AKConv(2 * c)

    def forward(self, x):
        x = self.conv2d(self.in_proj(x))
        y = self.forward_core(x, channel_first=True, force_fp32=True)
        return self.out_proj(y.permute(0, 3, 1, 2).contiguous())


class AKCAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.akc = AKCBlock(c)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // 16, 1), 1),
                                nn.ReLU(), nn.Conv2d(max(c // 16, 1), c, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.akc(x)
        return x * self.se(x)


class AKC3Adapter(nn.Module):
    def __init__(self, c, four=False):
        super().__init__()
        self.project = Conv(c, c, 1)
        self.local = CAKC(c, four=four)
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.ssm = AKSS2D(c)
        self.attention = AKCAttention(c)

    def forward(self, x):
        x = self.project(x)
        return self.attention(self.norm2(self.ssm(self.norm1(self.local(x))) + x))


class AKC4Adapter(AKC3Adapter):
    def __init__(self, c):
        super().__init__(c, four=True)


class MambaAdapter(nn.Module):
    def __init__(self, c, neck=False):
        super().__init__()
        self.block = XSSBlock(c, c, n=1) if neck else VSSBlock(c, c)

    def forward(self, x):
        return self.block(x)
