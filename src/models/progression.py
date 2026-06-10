# model_progression.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# EfficientNet-B0 2.5D baseline (uses torchvision if available)
try:
    import torchvision
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


# -----------------------------
# 1) Simple 3D CNN (MR-ADProgNet)
# -----------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Prog3DCNN(nn.Module):
    """
    Output: scalar in [0,1] via sigmoid
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock3D(in_ch, base, k=3, s=1, p=1),
            ConvBlock3D(base, base, k=3, s=1, p=1),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock3D(base, base * 2),
            ConvBlock3D(base * 2, base * 2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock3D(base * 2, base * 4),
            ConvBlock3D(base * 4, base * 4),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock3D(base * 4, base * 8),
            ConvBlock3D(base * 8, base * 8),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base * 8, base * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base * 4, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.head(x)              # [B,1]
        x = torch.sigmoid(x).squeeze(1)  # [B]
        return x


# -----------------------------
# 2) 3D DenseNet-121 style
# -----------------------------
class _DenseLayer3D(nn.Module):
    def __init__(self, in_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter = bn_size * growth_rate
        self.norm1 = nn.BatchNorm3d(in_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_features, inter, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm3d(inter)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inter, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, x):
        new = self.conv1(self.relu1(self.norm1(x)))
        new = self.conv2(self.relu2(self.norm2(new)))
        if self.drop_rate > 0:
            new = F.dropout(new, p=self.drop_rate, training=self.training)
        return torch.cat([x, new], dim=1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, in_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        feats = in_features
        for _ in range(num_layers):
            layers.append(_DenseLayer3D(feats, growth_rate, bn_size, drop_rate))
            feats += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_features = feats

    def forward(self, x):
        return self.block(x)


class _Transition3D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet3D121(nn.Module):
    """
    DenseNet-121 config: [6, 12, 24, 16]
    Output: scalar in [0,1]
    """
    def __init__(self, in_ch=1, growth_rate=32, bn_size=4, drop_rate=0.0, init_features=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        block_config = [6, 12, 24, 16]
        num_features = init_features
        for i, nlayers in enumerate(block_config):
            block = _DenseBlock3D(nlayers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = block.out_features
            if i != len(block_config) - 1:
                out_features = num_features // 2
                trans = _Transition3D(num_features, out_features)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = out_features

        self.norm_final = nn.BatchNorm3d(num_features)

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(num_features, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.norm_final(x)
        x = self.head(x)                 # [B,1]
        x = torch.sigmoid(x).squeeze(1)  # [B]
        return x


# -----------------------------
# 3) EfficientNet-B0 (2.5D baseline)
# -----------------------------
class EfficientNetB0_2p5D(nn.Module):
    """
    Takes a 3D volume [B,1,D,H,W]
    Samples K axial slices, runs 2D EfficientNet-B0, averages embeddings, regresses to [0,1].

    This is a GREAT baseline: strong model, easy, fast, and publishable as "2.5D baseline".
    """
    def __init__(self, num_slices=16, pretrained=True):
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for EfficientNetB0_2p5D baseline.")

        self.num_slices = int(num_slices)

        net = torchvision.models.efficientnet_b0(pretrained=bool(pretrained))
        # Use feature extractor up to pooling
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_dim = 1280  # efficientnet_b0 last channel
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # x: [B,1,D,H,W]
        b, c, d, h, w = x.shape
        k = min(self.num_slices, d)
        # evenly spaced slice indices
        idx = torch.linspace(0, d - 1, steps=k).long().to(x.device)

        # collect slices: [B, K, H, W]
        slices = x[:, 0, idx, :, :]  # take channel 0
        # reshape to [B*K, 3, H, W] (repeat channels)
        slices = slices.unsqueeze(2).repeat(1, 1, 3, 1, 1).reshape(b * k, 3, h, w)

        feats = self.features(slices)
        pooled = self.pool(feats).flatten(1)  # [B*K, embed_dim]
        pooled = pooled.reshape(b, k, self.embed_dim).mean(dim=1)  # [B, embed_dim]

        out = self.head(pooled)               # [B,1]
        out = torch.sigmoid(out).squeeze(1)   # [B]
        return out


def build_progression_model(name: str):
    name = name.lower().strip()
    if name in ["prog3dcnn", "mr-adprognet", "adprog3d"]:
        return Prog3DCNN()
    if name in ["densenet3d121", "densenet-121-3d", "densenet121_3d"]:
        return DenseNet3D121()
    if name in ["efficientnet_b0_2p5d", "efficientnetb0", "effb0_2p5d"]:
        return EfficientNetB0_2p5D(num_slices=16, pretrained=True)
    raise ValueError(f"Unknown model name: {name}")
