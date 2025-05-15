import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) / 
                    (pred_flat.sum() + target_flat.sum() + self.smooth))

class FocalLoss(nn.Module):
    """Focal loss for classification tasks."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross entropy loss for segmentation tasks."""
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pred, target, weight=self.weights)

class CombinedLoss(nn.Module):
    """Combined loss for segmentation tasks."""
    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1.0
    ):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = WeightedCrossEntropyLoss(weights=weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss for enhancement tasks."""
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
        
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.device == img1.device:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel
            
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

def get_loss_function(
    task: str,
    weights: Optional[torch.Tensor] = None,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> nn.Module:
    """Get appropriate loss function based on task."""
    if task == "segmentation":
        return CombinedLoss(weights=weights, dice_weight=dice_weight, ce_weight=ce_weight)
    elif task == "classification":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif task == "enhancement":
        return SSIMLoss()
    else:
        raise ValueError(f"Unknown task type: {task}") 