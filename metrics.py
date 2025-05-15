import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

def focal_loss(gamma: float = 2.0, alpha: float = 4.0):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    
    Args:
        gamma: Focusing parameter
        alpha: Weighting factor
    """
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-9
        pred = torch.clamp(pred, epsilon, 1 - epsilon)
        ce_loss = -target * torch.log(pred)
        weight = target * torch.pow(1 - pred, gamma)
        fl = alpha * weight * ce_loss
        return fl.mean()
    return loss_fn

def weighted_categorical_crossentropy(weights: Optional[torch.Tensor] = None):
    """Weighted categorical crossentropy loss.
    
    Args:
        weights: Class weights tensor
    """
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_pixel_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        if weights is not None:
            weight_mask = torch.maximum(
                torch.max(weights[None, None, None] * target, dim=-1)[0],
                torch.ones_like(target[..., 0])
            )
            per_pixel_loss = per_pixel_loss * weight_mask[..., None]
        return per_pixel_loss.mean()
    return loss_fn

def image_categorical_cross_entropy(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Image categorical cross entropy loss.
    
    Args:
        pred: Prediction tensor of shape (batch_size, height, width, num_classes)
        target: Target tensor of shape (batch_size, height, width, num_classes)
        weights: Optional class weights tensor
    """
    per_pixel_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    if weights is not None:
        weight_mask = torch.maximum(
            torch.max(weights[None, None, None] * target, dim=-1)[0],
            torch.ones_like(target[..., 0])
        )
        per_pixel_loss = per_pixel_loss * weight_mask[..., None]
    return per_pixel_loss.mean()

def class_tversky(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Class-wise Tversky coefficient.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        smooth: Smoothing factor
    """
    pred = pred.permute(3, 1, 2, 0)  # Move class dimension to front
    target = target.permute(3, 1, 2, 0)
    
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    
    true_pos = torch.sum(pred_flat * target_flat, dim=1)
    false_neg = torch.sum(target_flat * (1 - pred_flat), dim=1)
    false_pos = torch.sum((1 - target_flat) * pred_flat, dim=1)
    
    alpha = 0.2
    beta = 0.8
    return (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def focal_tversky_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Focal Tversky loss.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    """
    pt_1 = class_tversky(pred, target)
    gamma = 1.3
    return torch.sum(torch.pow(1 - pt_1, gamma))

def generalized_dice_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized Dice coefficient.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    """
    axes = tuple(range(1, len(pred.shape) - 1))
    n_cl = pred.shape[-1]
    
    w = torch.sum(target, dim=axes)
    w = 1 / (w ** 2 + 1e-6)
    
    numerator = target * pred
    numerator = w * torch.sum(numerator, dim=axes)
    numerator = torch.sum(numerator)
    
    denominator = target + pred
    denominator = w * torch.sum(denominator, dim=axes)
    denominator = torch.sum(denominator)
    
    return 2 * numerator / denominator

def generalized_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized Dice loss.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    """
    return 1 - generalized_dice_coeff(pred, target)

def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss calculation.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        epsilon: Small constant for numerical stability
    """
    axes = tuple(range(1, len(pred.shape) - 1))
    numerator = 2. * torch.sum(pred * target, dim=axes)
    denominator = torch.sum(pred * pred + target * target, dim=axes)
    return 1. - torch.mean(numerator / (denominator + epsilon))

def seg_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric_name: str,
    metric_type: str = 'standard',
    drop_last: bool = True,
    mean_per_class: bool = False,
    verbose: bool = False
) -> torch.Tensor:
    """Compute segmentation metrics.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        metric_name: Metric to compute ('iou' or 'dice')
        metric_type: Type of metric ('standard', 'soft', or 'naive')
        drop_last: Whether to drop last class
        mean_per_class: Whether to return per-class metrics
        verbose: Whether to print intermediate results
    """
    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')
    
    num_classes = pred.shape[-1]
    
    if not flag_soft:
        pred = F.one_hot(pred.argmax(dim=-1), num_classes).float()
        target = F.one_hot(target.argmax(dim=-1), num_classes).float()
    
    axes = (1, 2)
    intersection = torch.sum(torch.abs(target * pred), dim=axes)
    mask_sum = torch.sum(torch.abs(target), dim=axes) + torch.sum(torch.abs(pred), dim=axes)
    union = mask_sum - intersection
    
    smooth = 0.001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    
    metric = {'iou': iou, 'dice': dice}[metric_name]
    mask = (union != 0).float()
    
    if drop_last:
        metric = metric[:, :-1]
        mask = mask[:, :-1]
    
    if verbose:
        print('intersection, union')
        print(intersection.cpu().numpy(), union.cpu().numpy())
        print((intersection / union).cpu().numpy())
    
    if flag_naive_mean:
        return metric.mean()
    
    class_count = torch.sum(mask, dim=0)
    non_zero = class_count > 0
    non_zero_sum = torch.sum(metric * mask, dim=0)[non_zero]
    non_zero_count = class_count[non_zero]
    
    if verbose:
        print('Counts of inputs with class present, metrics for non-absent classes')
        print(class_count.cpu().numpy(), (non_zero_sum / non_zero_count).cpu().numpy())
    
    return torch.mean(non_zero_sum / non_zero_count)

def mean_iou(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """Compute mean IoU.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        **kwargs: Additional arguments for seg_metrics
    """
    return seg_metrics(pred, target, metric_name='iou', **kwargs)

def jaccard_distance_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 100.0) -> torch.Tensor:
    """Jaccard distance loss.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        smooth: Smoothing factor
    """
    intersection = torch.sum(torch.abs(target * pred), dim=-1)
    sum_ = torch.sum(torch.abs(target) + torch.abs(pred), dim=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

class MetricTracker:
    """Base class for tracking metrics."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.values = []
        
    def update(self, value: float):
        self.values.append(value)
        
    def compute(self) -> float:
        return np.mean(self.values)

class Accuracy(MetricTracker):
    """Accuracy metric for classification tasks."""
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.argmax(dim=1)
        correct = (pred == target).float().mean()
        super().update(correct.item())

class IoU(MetricTracker):
    """Intersection over Union metric for segmentation tasks."""
    def __init__(self, n_classes: int, ignore_index: Optional[int] = None):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.argmax(dim=1)
        intersection = torch.zeros(self.n_classes, device=pred.device)
        union = torch.zeros(self.n_classes, device=pred.device)
        
        for cls in range(self.n_classes):
            if cls == self.ignore_index:
                continue
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            intersection[cls] = (pred_mask & target_mask).sum()
            union[cls] = (pred_mask | target_mask).sum()
            
        iou = intersection / (union + 1e-8)
        if self.ignore_index is not None:
            iou = iou[iou != self.ignore_index]
        super().update(iou.mean().item())

class DiceScore(MetricTracker):
    """Dice coefficient metric for segmentation tasks."""
    def __init__(self, n_classes: int, ignore_index: Optional[int] = None):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.argmax(dim=1)
        dice = torch.zeros(self.n_classes, device=pred.device)
        
        for cls in range(self.n_classes):
            if cls == self.ignore_index:
                continue
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            intersection = (pred_mask & target_mask).sum()
            dice[cls] = (2. * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-8)
            
        if self.ignore_index is not None:
            dice = dice[dice != self.ignore_index]
        super().update(dice.mean().item())

class PSNR(MetricTracker):
    """Peak Signal-to-Noise Ratio metric for enhancement tasks."""
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        super().update(psnr.item())

class SSIM(MetricTracker):
    """Structural Similarity Index metric for enhancement tasks."""
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
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
        
    def update(self, img1: torch.Tensor, img2: torch.Tensor):
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
        super().update(ssim_map.mean().item())

def get_metrics(task: str, n_classes: Optional[int] = None) -> Dict[str, MetricTracker]:
    """Get appropriate metrics based on task."""
    metrics = {}
    
    if task == "segmentation":
        metrics.update({
            "iou": IoU(n_classes=n_classes),
            "dice": DiceScore(n_classes=n_classes)
        })
    elif task == "classification":
        metrics.update({
            "accuracy": Accuracy()
        })
    elif task == "enhancement":
        metrics.update({
            "psnr": PSNR(),
            "ssim": SSIM()
        })
    else:
        raise ValueError(f"Unknown task type: {task}")
        
    return metrics

def compute_metrics(
    metrics: Dict[str, MetricTracker],
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """Compute all metrics for given predictions and targets."""
    results = {}
    for name, metric in metrics.items():
        metric.update(pred, target)
        results[name] = metric.compute()
    return results
