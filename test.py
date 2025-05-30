from models.model import UNet
from models.mores import *
from models.vnet import *

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = [UNet(in_channels=1, out_channels=1)
,VNet2D(in_channels=1, out_channels=1)
,ImprovedVNet(in_channels=1, num_classes=1)     
,TransUNet(in_channels=1, out_channels=1)   
,ResUNet(in_channels=1, out_channels=1)
,ASPPUNet(in_channels=1, out_channels=1)
,AttentionUNet(in_channels=1, out_channels=1)]

for m in model:
    print(count_params(m))