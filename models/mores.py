import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # 编码器（下采样）
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 中间层
        self.middle = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(512, 1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        
        # 解码器（上采样）
        self.decoder3 = self.upconv_block(1024, 256)
        self.decoder2 = self.upconv_block(512, 128)
        self.decoder1 = self.upconv_block(256, 64)

        # 输出层
        self.final = nn.Sequential(
            self.conv_block(128, 64),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def conv_block(self, in_channels, out_channels):
        """卷积块：包含卷积层、ReLU激活和批归一化"""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        """上采样块：包括转置卷积和卷积块"""
        block = nn.Sequential(
            self.conv_block(in_channels, in_channels // 2),
            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=2, stride=2),
        )
        return block

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)                         # torch.Size([2, 64, 512, 512])
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))     # torch.Size([2, 128, 256, 256])
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))     # torch.Size([2, 256, 128, 128])
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))     # torch.Size([2, 512, 64, 64])

        # 中间层
        dec4 = self.middle(enc4)                        # torch.Size([2, 512, 64, 64])

        # 解码器
        dec4 = torch.cat([dec4, enc4], dim=1)           # torch.Size([2, 1024, 64, 64])
        dec3 = self.decoder3(dec4)                      # torch.Size([2, 256, 128, 128])
        dec3 = torch.cat([dec3, enc3], dim=1)           # torch.Size([2, 512, 128, 128])
        dec2 = self.decoder2(dec3)                      # torch.Size([2, 128, 256, 256])
        dec2 = torch.cat([dec2, enc2], dim=1)           # torch.Size([2, 256, 256, 256])
        dec1 = self.decoder1(dec2)                      # torch.Size([2, 64, 512, 512])
        dec1 = torch.cat([dec1, enc1], dim=1)           # torch.Size([2, 128, 512, 512])
        
        # 输出
        return self.final(dec1)

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


######## 
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block to recalibrate channel-wise feature responses."""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        # Squeeze: Global Average Pooling to get channel descriptors
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: Two fully-connected layers (implemented as 1x1 convolutions for convenience)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Squeeze: aggregate spatial information into a channel-wise vector
        z = self.avg_pool(x)
        # First excitation: reduce channels then ReLU
        z = self.relu(self.fc1(z))
        # Second excitation: expand back to original channels and apply sigmoid
        z = self.sigmoid(self.fc2(z))
        # Re-calibrate: scale the input features by the learned channel weights
        return x * z

class ConvBlock(nn.Module):
    """
    Convolutional block with a specified number of conv layers, each followed by 
    BatchNorm, ReLU, and Dropout. Incorporates a residual connection.
    """
    def __init__(self, in_channels, out_channels, num_convs, dropout_rate):
        super(ConvBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout_rate)
        # Define convolutional layers (3x3 convs)
        for i in range(num_convs):
            conv_in = in_channels if i == 0 else out_channels
            conv_out = out_channels
            self.convs.append(nn.Conv2d(conv_in, conv_out, kernel_size=3, stride=1, padding=1))
            self.bns.append(nn.BatchNorm2d(conv_out))
        # Residual projection (1x1 conv) if in/out channels differ
        self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        # Forward through each Conv-BN-ReLU-Dropout sequence
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            x = self.drop(x)
        # Add residual connection
        if self.res_proj is not None:
            residual = self.res_proj(residual)
        x = x + residual
        return x

class ImprovedVNet(nn.Module):
    """
    Improved V-Net model for thyroid nodule segmentation.
    Uses a three-branch encoder fusion structure with SE modules and a U-Net style decoder.
    """
    def __init__(self, in_channels=1, num_classes=1, base_num_filters=64, dropout_rate=0.05, se_reduction=4):
        super(ImprovedVNet, self).__init__()
        self.num_branches = 3  # Three encoder branches for triple fusion
        self.in_channels = in_channels

        # Define number of filters at each encoder level (doubles each downsampling)
        filters = [base_num_filters * (2 ** i) for i in range(5)]  # e.g., [16, 32, 64, 128, 256] if base_num_filters=16
        
        # Encoder blocks and SE modules for each branch
        self.enc_blocks = nn.ModuleList([nn.ModuleList() for _ in range(self.num_branches)])
        self.enc_ses    = nn.ModuleList([nn.ModuleList() for _ in range(self.num_branches)])
        self.down_convs = nn.ModuleList([nn.ModuleList() for _ in range(self.num_branches)])
        enc_conv_counts = [2, 2, 3, 3, 3]  # conv layers in encoder blocks 1-5

        # Build encoder branches
        for b in range(self.num_branches):
            for i in range(5):
                # Determine in/out channels for this encoder block
                if i == 0:
                    in_ch = in_channels
                    out_ch = filters[0]
                else:
                    in_ch = filters[i]    # after downsampling, channel count = filters[i]
                    out_ch = filters[i]
                # Encoder conv block
                self.enc_blocks[b].append(ConvBlock(in_ch, out_ch, enc_conv_counts[i], dropout_rate))
                # SE module after encoder block
                self.enc_ses[b].append(SEBlock(out_ch, reduction=se_reduction))
                # Downsampling conv (except after last encoder block)
                if i < 4:
                    self.down_convs[b].append(nn.Conv2d(out_ch, filters[i+1], kernel_size=3, stride=2, padding=1))
        
        # Decoder layers (shared, one decoder for all encoder branches)
        # Transposed conv layers for upsampling (halves channels, doubles spatial size)
        self.up6 = nn.ConvTranspose2d(filters[4] * self.num_branches, filters[3], kernel_size=2, stride=2)
        self.up7 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.up8 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up9 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        # Decoder conv blocks (with skip fusion). conv layers: block6&7=3 convs, block8&9=2 convs
        self.dec_blocks = nn.ModuleList([
            ConvBlock(filters[3] + filters[3] * self.num_branches, filters[3], num_convs=3, dropout_rate=dropout_rate),  # block6
            ConvBlock(filters[2] + filters[2] * self.num_branches, filters[2], num_convs=3, dropout_rate=dropout_rate),  # block7
            ConvBlock(filters[1] + filters[1] * self.num_branches, filters[1], num_convs=2, dropout_rate=dropout_rate),  # block8
            ConvBlock(filters[0] + filters[0] * self.num_branches, filters[0], num_convs=2, dropout_rate=dropout_rate)   # block9
        ])
        # SE module in the last decoder block (prior to output)
        self.dec_se_final = SEBlock(filters[0], reduction=se_reduction)
        # Final 1x1 convolution to output segmentation map
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        assert x.shape[1] == self.in_channels, f"Expected input with {self.in_channels} channel(s)"
        # Encoder forward pass for each branch (collect skip connections)
        enc_features = [[None] * 5 for _ in range(self.num_branches)]
        for b in range(self.num_branches):
            e = x  # input to this branch
            for i in range(5):
                # Conv block + SE in encoder
                e = self.enc_blocks[b][i](e)
                e = self.enc_ses[b][i](e)
                enc_features[b][i] = e  # store skip feature
                if i < 4:  # downsample to next encoder block
                    e = self.down_convs[b][i](e)
        # Fuse encoder branch outputs at the bottom layer (concatenate along channels)
        fused_bottom = torch.cat([enc_features[b][4] for b in range(self.num_branches)], dim=1)
        d = fused_bottom
        # Decoder with skip connections from all branches
        # Decoder block 6 -> skip from encoder block 4, block 7 -> skip from encoder 3, etc.
        d = self.up6(d)  # upsample from bottom (32->64)
        # Fuse skip from encoder level 4 (index 3 in 0-based) across all branches
        skip4 = torch.cat([enc_features[b][3] for b in range(self.num_branches)], dim=1)
        d = torch.cat([d, skip4], dim=1)
        d = self.dec_blocks[0](d)  # decoder block6 convs (with residual)
        d = self.up7(d)  # 64->128
        skip3 = torch.cat([enc_features[b][2] for b in range(self.num_branches)], dim=1)
        d = torch.cat([d, skip3], dim=1)
        d = self.dec_blocks[1](d)  # decoder block7
        d = self.up8(d)  # 128->256
        skip2 = torch.cat([enc_features[b][1] for b in range(self.num_branches)], dim=1)
        d = torch.cat([d, skip2], dim=1)
        d = self.dec_blocks[2](d)  # decoder block8
        d = self.up9(d)  # 256->512 (original size)
        skip1 = torch.cat([enc_features[b][0] for b in range(self.num_branches)], dim=1)
        d = torch.cat([d, skip1], dim=1)
        d = self.dec_blocks[3](d)  # decoder block9
        # Apply SE in last decoder block and produce final output
        d = self.dec_se_final(d)
        out = self.final_conv(d)
        return out
