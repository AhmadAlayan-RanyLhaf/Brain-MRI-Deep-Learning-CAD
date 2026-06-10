# segmentation_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """
    Two successive Conv3D -> GroupNorm -> SiLU layers.
    GroupNorm is ideal for 3D medical segmentation because it is independent of batch size (batch_size=2 is common).
    """
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    """
    State-of-the-art 3D U-Net for medical segmentation.
    Preserves spatial context using Contracting and Expanding paths with Skip Connections.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, base_filters: int = 16):
        super().__init__()
        
        # Contracting Path (Encoder)
        self.inc = DoubleConv3D(in_channels, base_filters)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(base_filters, base_filters * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(base_filters * 2, base_filters * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(base_filters * 4, base_filters * 8)
        )
        
        # Expanding Path (Decoder)
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConv3D(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv3D(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv1 = DoubleConv3D(base_filters * 2, base_filters)
        
        # Final Classification Head
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        up_x3 = self.up3(x4)
        # Ensure dimensions match in case of odd voxel sizes during pooling
        if up_x3.shape != x3.shape:
            up_x3 = F.interpolate(up_x3, size=x3.shape[2:], mode="trilinear", align_corners=False)
        concat_x3 = torch.cat([x3, up_x3], dim=1)
        x3_dec = self.conv3(concat_x3)
        
        up_x2 = self.up2(x3_dec)
        if up_x2.shape != x2.shape:
            up_x2 = F.interpolate(up_x2, size=x2.shape[2:], mode="trilinear", align_corners=False)
        concat_x2 = torch.cat([x2, up_x2], dim=1)
        x2_dec = self.conv2(concat_x2)
        
        up_x1 = self.up1(x2_dec)
        if up_x1.shape != x1.shape:
            up_x1 = F.interpolate(up_x1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        concat_x1 = torch.cat([x1, up_x1], dim=1)
        x1_dec = self.conv1(concat_x1)
        
        logits = self.out_conv(x1_dec)
        return logits

if __name__ == "__main__":
    # Smoke test model forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(in_channels=1, out_channels=4).to(device)
    dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)
    dummy_output = model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Dummy output shape (logits):", dummy_output.shape) # Expected: [1, 4, 64, 64, 64]
