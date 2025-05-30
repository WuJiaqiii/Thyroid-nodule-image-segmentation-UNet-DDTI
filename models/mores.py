import torch
import torch.nn as nn
import torch.nn.functional as F

class VNet2D(nn.Module):
    """
    2D adaptation of V-Net (Milletari et al.) for segmentation.
    Uses residual blocks, strided conv for downsampling, and transpose conv for upsampling.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256]):
        super().__init__()
        # Encoding path
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc_blocks.append(self._block(ch, f))
            # downsample by factor 2
            self.down_convs.append(nn.Conv2d(f, f, kernel_size=2, stride=2, bias=False))
            ch = f
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        # Decoding path
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev_ch = features[-1] * 2
        for f in reversed(features):
            # upsample by factor 2
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2, bias=False))
            # after concatenation, channels = f(skip) + f(up) = 2f
            self.dec_blocks.append(self._block(f * 2, f))
            prev_ch = f
        # Final conv to get desired output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU()
        )

    def forward(self, x):
        skips = []
        # Encoder
        for enc, down in zip(self.enc_blocks, self.down_convs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder
        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            x = up(x)
            # pad if needed
            if x.size() != skip.size():
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        # Final conv
        return self.final_conv(x)

# 1. Attention U-Net
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True)
            ))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = f
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        rev_features = features[::-1]
        for f in rev_features:
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.attentions.append(AttentionGate(F_g=f, F_l=f, F_int=f//2))
            self.ups.append(nn.Sequential(
                nn.Conv2d(f*2, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True)
            ))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            upconv = self.ups[idx]
            conv = self.ups[idx+1]
            skip = skip_connections[idx//2]
            x = upconv(x)
            skip = self.attentions[idx//2](g=x, x=skip)
            x = torch.cat((skip, x), dim=1)
            x = conv(x)
        return self.final_conv(x)


# 2. Residual U-Net (ResU-Net)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv(x)
        out += identity
        return self.relu(out)

class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f in features:
            self.downs.append(ResidualBlock(in_channels, f))
            self.pools.append(nn.MaxPool2d(2, 2))
            in_channels = f
        self.bottleneck = ResidualBlock(features[-1], features[-1]*2)
        self.ups = nn.ModuleList()
        rev = features[::-1]
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(ResidualBlock(f*2, f))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        return self.final(x)


# 3. U-Net with ASPP (DeepLab-like)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super().__init__()
        self.convs = nn.ModuleList()
        for r in rates:
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False)
            )
        self.project = nn.Sequential(
            nn.Conv2d(len(rates)*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        x = torch.cat(res, dim=1)
        return self.project(x)

class ASPPUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        # encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True)
            ))
            self.pools.append(nn.MaxPool2d(2,2))
            in_channels = f
        self.bottleneck = ASPP(features[-1], features[-1]*2)
        # decoder
        self.ups = nn.ModuleList()
        rev = features[::-1]
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2,2))
            self.ups.append(nn.Sequential(
                nn.Conv2d(f*2, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True)
            ))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        return self.final(x)


# 4. TransUNet (simplified)
from torchvision.transforms import Resize
from einops import rearrange

# 4. TransUNet (修正版)
class TransEncoder(nn.Module):
    def __init__(self, in_channels=512, dim=256, heads=8, depth=4, spatial_size=32):
        super().__init__()
        # 1x1 投影，保留空间维度
        self.patchify = nn.Conv2d(in_channels, dim, kernel_size=1, bias=False)
        # 位置编码，与 spatial_size^2 对应
        self.pos_emb = nn.Parameter(torch.randn(1, spatial_size*spatial_size, dim))
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=0.1)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: [B, C, H, W], H=W=spatial_size
        x = self.patchify(x)  # [B, dim, H, W]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c') + self.pos_emb
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x, (H, W)

class TransUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512],
                 trans_dim=256, heads=8, depth=4):
        super().__init__()
        # CNN 编码
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(nn.Sequential(
                nn.Conv2d(ch, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True)
            ))
            self.pools.append(nn.MaxPool2d(2,2))
            ch = f
        # Transformer Bottleneck
        self.trans = TransEncoder(in_channels=features[-1], dim=trans_dim,
                                   heads=heads, depth=depth, spatial_size=512//(2**len(features)))
        self.trans_proj = nn.Linear(trans_dim, features[-1])
        # 解码
        self.ups = nn.ModuleList()
        for f in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2,2))
            self.ups.append(nn.Sequential(
                nn.Conv2d(f*2, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True)
            ))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skip.append(x); x = pool(x)
        # Transformer
        trans_out, (H, W) = self.trans(x)
        x = self.trans_proj(trans_out)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        skip = skip[::-1]
        # 解码
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat([skip[i//2], x], dim=1)
            x = self.ups[i+1](x)
        return self.final(x)
