import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------------
# 1. UNet
# ----------------------------------
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 64,
                 depth: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        channels = [base_filters * (2**i) for i in range(depth)]
        for ch in channels:
            self.encoders.append(self._block(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2,2))
            prev_ch = ch
        # Bottleneck
        self.bottleneck = self._block(prev_ch, prev_ch*2)
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = channels[::-1]
        prev_ch = channels[-1]*2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.decoders.append(self._block(prev_ch, ch))
            prev_ch = ch
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.final_conv(x)

# ----------------------------------
# 2. ResUNet
# ----------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class ResUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 64,
                 depth: int = 5):
        super().__init__()
        self.base_filters = base_filters
        self.depth = depth
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        channels = [base_filters * (2**i) for i in range(depth)]
        for ch in channels:
            self.encoders.append(ResidualBlock(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2,2))
            prev_ch = ch
        # Bottleneck
        self.bottleneck = ResidualBlock(prev_ch, prev_ch*2)
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = channels[::-1]
        prev_ch = channels[-1]*2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, 2,2))
            self.decoders.append(ResidualBlock(prev_ch, ch))
            prev_ch = ch
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        skips=[]
        for enc,pool in zip(self.encoders,self.pools):
            x=enc(x)
            skips.append(x)
            x=pool(x)
        x=self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x=up(x)
            if x.shape!=skip.shape:
                x=F.interpolate(x,size=skip.shape[2:],mode='bilinear',align_corners=False)
            x=torch.cat([skip,x],dim=1)
            x=dec(x)
        return self.final_conv(x)

# ----------------------------------
# 3. ASPPUNet
# ----------------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1,6,12,18]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(dilations)*out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))

class ASPPUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 64,
                 depth: int = 5,
                 aspp_dilations: list = [1,6,12,18]):
        super().__init__()
        self.base_filters = base_filters
        self.depth = depth
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        channels = [base_filters * (2**i) for i in range(depth)]
        for ch in channels:
            self.encoders.append(self._block(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2,2))
            prev_ch = ch
        # Bottleneck ASPP
        self.aspp = ASPP(channels[-1], channels[-1]*2, dilations=aspp_dilations)
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = channels[::-1]
        prev_ch = channels[-1]*2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, 2,2))
            self.decoders.append(self._block(prev_ch, ch))
            prev_ch = ch
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips=[]
        for enc,pool in zip(self.encoders,self.pools):
            x=enc(x); skips.append(x); x=pool(x)
        x=self.aspp(x)
        for up,dec,skip in zip(self.upconvs,self.decoders,reversed(skips)):
            x=up(x)
            if x.shape!=skip.shape:
                x=F.interpolate(x,size=skip.shape[2:],mode='bilinear',align_corners=False)
            x=torch.cat([skip,x],dim=1)
            x=dec(x)
        return self.final_conv(x)

# ----------------------------------
# 4. AttentionUNet
# ----------------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1=self.W_g(g)
        x1=self.W_x(x)
        psi=self.relu(g1+x1)
        psi=self.psi(psi)
        return x*psi

class AttentionUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 64,
                 depth: int = 5):
        super().__init__()
        self.base_filters = base_filters
        self.depth = depth
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        channels = [base_filters * (2**i) for i in range(depth)]
        for ch in channels:
            self.encoders.append(self._block(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2,2))
            prev_ch = ch
        # Bottleneck
        self.bottleneck = self._block(prev_ch, prev_ch*2)
        # Decoder
        self.upconvs = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = channels[::-1]
        prev_ch = channels[-1]*2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, 2,2))
            self.attn_gates.append(AttentionGate(F_g=ch, F_l=ch, F_int=ch//2))
            self.decoders.append(self._block(prev_ch, ch))
            prev_ch = ch
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips=[]
        for enc,pool in zip(self.encoders,self.pools):
            x=enc(x); skips.append(x); x=pool(x)
        x=self.bottleneck(x)
        for up, gate, dec, skip in zip(self.upconvs, self.attn_gates, self.decoders, reversed(skips)):
            x=up(x)
            if x.shape!=skip.shape:
                x=F.interpolate(x,size=skip.shape[2:],mode='bilinear',align_corners=False)
            skip_att=gate(g=x, x=skip)
            x=torch.cat([skip_att, x], dim=1)
            x=dec(x)
        return self.final_conv(x)

# ----------------------------------
# 5. TransUNet
# ----------------------------------
class TransEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 spatial_size: int):
        super().__init__()
        self.patchify = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(1, spatial_size*spatial_size, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4, dropout=0.1)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.patchify(x)
        x = rearrange(x, 'b c h w -> b (h w) c') + self.pos_emb
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x, (H, W)

class TransUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 64,
                 depth: int = 5,
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 embed_dim: int = 256,
                 image_size: int = 512):
        super().__init__()
        self.base_filters = base_filters
        self.depth = depth

        # --- CNN 编码器 ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        channels = [base_filters * (2 ** i) for i in range(depth)]
        for ch in channels:
            self.encoders.append(self._block(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2, 2))
            prev_ch = ch

        # --- Transformer Bottleneck ---
        spatial_size = image_size // (2 ** depth)
        self.trans = TransEncoder(
            in_channels=channels[-1],
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            spatial_size=spatial_size
        )
        self.trans_proj = nn.Linear(embed_dim, channels[-1])

        # --- Decoder ---
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = channels[::-1]
        prev_ch = channels[-1]  # Transformer 输出投影后通道
        for ch in rev_channels:
            # 上采样：prev_ch -> ch
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, 2, 2))
            # 解码块：in_ch=ch(skip)+ch(up) = 2*ch
            self.decoders.append(self._block(ch * 2, ch))
            prev_ch = ch

        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        # 编码
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Transformer
        x, (H, W) = self.trans(x)
        x = self.trans_proj(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        # 解码
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)   # 合并 skip 特征
            x = dec(x)

        return self.final_conv(x)

# ----------------------------------
# 6. VNet2D
# ----------------------------------
class VNet2D(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 16,
                 depth: int = 5):
        super().__init__()
        self.base_filters = base_filters
        self.depth = depth
        # Build encoder blocks and downsample
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        prev_ch = in_channels
        self.features = [base_filters * (2**i) for i in range(depth)]
        for f in self.features:
            self.enc_blocks.append(self._block(prev_ch, f))
            self.down_convs.append(nn.Conv2d(f, f, kernel_size=2, stride=2, bias=False))
            prev_ch = f
        # Bottleneck
        self.bottleneck = self._block(self.features[-1], self.features[-1]*2)
        # Build decoder blocks and upsample
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev_ch = self.features[-1]*2
        for f in reversed(self.features):
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2, bias=False))
            self.dec_blocks.append(self._block(prev_ch, f))
            prev_ch = f
        # Final conv
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, x):
        skips = []
        for enc, down in zip(self.enc_blocks, self.down_convs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.final_conv(x)

# ----------------------------------
# 7. ImprovedVNet
# ----------------------------------
class ImprovedVNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 16,
                 depth: int = 5,
                 use_attention: bool = True,
                 deep_supervision: bool = False):
        super().__init__()
        self.base_filters = base_filters
        self.depth = depth
        self.use_attention = use_attention
        self.deep_supervision = deep_supervision
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        prev_ch = in_channels
        self.features = [base_filters * (2**i) for i in range(depth)]
        for f in self.features:
            self.enc_blocks.append(self._block(prev_ch, f))
            self.down_convs.append(nn.Conv2d(f, f, 2,2, bias=False))
            prev_ch = f
        # Bottleneck
        self.bottleneck = self._block(self.features[-1], self.features[-1]*2)
        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.attn_gates = nn.ModuleList() if use_attention else None
        prev_ch = self.features[-1]*2
        for f in reversed(self.features):
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, f, 2,2, bias=False))
            if use_attention:
                self.attn_gates.append(AttentionGate(F_g=f, F_l=f, F_int=f//2))
            self.dec_blocks.append(self._block(prev_ch, f))
            prev_ch = f
        # Deep supervision heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([nn.Conv2d(f, out_channels, 1) for f in self.features])
        # Final conv
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips=[]
        for enc,down in zip(self.enc_blocks,self.down_convs):
            x=enc(x); skips.append(x); x=down(x)
        x=self.bottleneck(x)
        ds_outs=[]
        for i,(up,dec) in enumerate(zip(self.up_convs,self.dec_blocks)):
            x=up(x)
            skip=skips[-1-i]
            if self.use_attention:
                gate=self.attn_gates[i]
                skip=gate(g=x,x=skip)
            if x.shape!=skip.shape:
                x=F.interpolate(x,size=skip.shape[2:],mode='bilinear',align_corners=False)
            x=torch.cat([skip,x],dim=1)
            x=dec(x)
            if self.deep_supervision:
                ds_outs.append(self.ds_heads[i](x))
        out=self.final_conv(x)
        if self.deep_supervision:
            return out, ds_outs
        return out
