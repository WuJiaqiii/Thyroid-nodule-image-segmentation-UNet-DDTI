import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as nd

class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: raw network outputs, shape (N, 1, H, W)
        # targets: float mask in {0,1}, same shape
        probs = torch.sigmoid(logits)
        # flatten
        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1).float()
        # intersection and union
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N,1,H,W), raw outputs
        # targets: (N,1,H,W), float in {0,1}
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        # TP, FP, FN
        TP = (probs_flat * targets_flat).sum()
        FP = ((probs_flat) * (1 - targets_flat)).sum()
        FN = ((1 - probs_flat) * targets_flat).sum()
        ti = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = (1 - ti) ** self.gamma
        return loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # logits, targets: (N,1,H,W)
        probs = torch.sigmoid(logits)
        targets_np = targets.detach().cpu().numpy().astype(np.uint8)  # for distance transform
        batch_size, _, H, W = targets.shape
        loss = 0.0
        for b in range(batch_size):
            gt = targets_np[b,0]
            # 1) 对正例mask外部做距离变换
            dt = nd.distance_transform_edt(1 - gt)
            # 转回 tensor
            dist_map = torch.from_numpy(dt).to(logits.device).float()
            # 2) boundary loss 累加
            loss += torch.mean(torch.abs(probs[b,0] - targets[b,0]) * dist_map)
        return loss / batch_size

class CompositeLoss(nn.Module):
    def __init__(self, λ_ft=1.0, λ_b=0.5, λ_bce=0.0, λ_dice=0.0):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.bl = BoundaryLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.λ_ft, self.λ_b, self.λ_bce, self.λ_dice = λ_ft, λ_b, λ_bce, λ_dice

    def forward(self, logits, targets):
        loss = self.λ_ft * self.ft(logits, targets) + self.λ_b * self.bl(logits, targets)
        if self.λ_bce > 0:
            loss = loss + self.λ_bce * self.bce(logits, targets)
        if self.λ_dice > 0:
            loss = loss + self.λ_dice * self.dice(logits, targets)
        return loss