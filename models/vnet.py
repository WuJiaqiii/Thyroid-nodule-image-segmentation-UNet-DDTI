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


if __name__ == '__main__':
    unet = ImprovedVNet()
    x = torch.zeros(size=(2, 1, 512, 512))
    print(x.shape, unet(x).shape)   # torch.Size([2, 1, 512, 512]) torch.Size([2, 1, 512, 512])
    