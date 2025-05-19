import torch
import torch.nn as nn

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