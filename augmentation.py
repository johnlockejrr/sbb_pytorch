import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from typing import Dict, Any, Optional

def get_training_augmentations(
    image_size: int = 224,
    scale: float = 0.5,
    rotation: int = 30,
    brightness: float = 0.2,
    contrast: float = 0.2,
    p: float = 0.5
) -> A.Compose:
    """Get training augmentations."""
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(scale, 1.0)),
        A.RandomRotate90(p=p),
        A.Flip(p=p),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=rotation, p=p),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=p),
            A.GridDistortion(p=p),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=p),
        ], p=p),
        A.OneOf([
            A.GaussNoise(p=p),
            A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=p),
            A.HueSaturationValue(p=p),
        ], p=p),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentations(image_size: int = 224) -> A.Compose:
    """Get validation augmentations."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_enhancement_augmentations(
    image_size: int = 224,
    brightness_range: tuple = (0.8, 1.2),
    contrast_range: tuple = (0.8, 1.2),
    noise_std: float = 0.1,
    blur_limit: int = 3,
    p: float = 0.5
) -> A.Compose:
    """Get augmentations specific to image enhancement tasks."""
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_range,
            contrast_limit=contrast_range,
            p=p
        ),
        A.GaussNoise(var_limit=(noise_std, noise_std), p=p),
        A.GaussianBlur(blur_limit=blur_limit, p=p),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class AugmentationConfig:
    """Configuration class for augmentations."""
    def __init__(
        self,
        image_size: int = 224,
        scale: float = 0.5,
        rotation: int = 30,
        brightness: float = 0.2,
        contrast: float = 0.2,
        noise_std: float = 0.1,
        blur_limit: int = 3,
        p: float = 0.5
    ):
        self.image_size = image_size
        self.scale = scale
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.blur_limit = blur_limit
        self.p = p

def get_augmentations(
    config: AugmentationConfig,
    task: str = "segmentation",
    is_training: bool = True
) -> A.Compose:
    """Get appropriate augmentations based on task and mode."""
    if task == "enhancement":
        return get_enhancement_augmentations(
            image_size=config.image_size,
            brightness_range=(1 - config.brightness, 1 + config.brightness),
            contrast_range=(1 - config.contrast, 1 + config.contrast),
            noise_std=config.noise_std,
            blur_limit=config.blur_limit,
            p=config.p
        )
    elif is_training:
        return get_training_augmentations(
            image_size=config.image_size,
            scale=config.scale,
            rotation=config.rotation,
            brightness=config.brightness,
            contrast=config.contrast,
            p=config.p
        )
    else:
        return get_validation_augmentations(image_size=config.image_size) 