# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Simple3DCNN(nn.Module):
    """
    Input:  [B, 1, 128, 128, 128]
    Output: logits [B] (binary classification)
    """
    def __init__(self, base_ch: int = 16, dropout: float = 0.2):
        super().__init__()
        ch = base_ch

        self.stem = nn.Sequential(
            ConvBlock3D(1, ch, stride=1),
            ConvBlock3D(ch, ch, stride=1),
        )

        self.stage1 = nn.Sequential(
            ConvBlock3D(ch, ch * 2, stride=2),   # 64^3
            ConvBlock3D(ch * 2, ch * 2, stride=1),
        )
        ch *= 2

        self.stage2 = nn.Sequential(
            ConvBlock3D(ch, ch * 2, stride=2),   # 32^3
            ConvBlock3D(ch * 2, ch * 2, stride=1),
        )
        ch *= 2

        self.stage3 = nn.Sequential(
            ConvBlock3D(ch, ch * 2, stride=2),   # 16^3
            ConvBlock3D(ch * 2, ch * 2, stride=1),
        )
        ch *= 2

        self.stage4 = nn.Sequential(
            ConvBlock3D(ch, ch * 2, stride=2),   # 8^3
            ConvBlock3D(ch * 2, ch * 2, stride=1),
        )
        ch *= 2

        self.pool = nn.AdaptiveAvgPool3d(1)  # [B, ch, 1,1,1]
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(ch, 1)           # binary logit

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        logit = self.fc(x).squeeze(1)        # [B]
        return logit


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
    Output: raw logits [B] (binary classification)
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
        logits = self.head(x).squeeze(1)                 # [B]
        return logits


def build_model(name: str, base_ch: int = 16, dropout: float = 0.2) -> nn.Module:
    name = name.lower().strip()
    if name in ["simple3dcnn", "simple3d", "cnn"]:
        return Simple3DCNN(base_ch=base_ch, dropout=dropout)
    elif name in ["densenet3d121", "densenet", "densenet3d"]:
        return DenseNet3D121(drop_rate=dropout)
    else:
        raise ValueError(f"Unknown model name: {name}")

