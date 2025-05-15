import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
import os
from tqdm import tqdm
import torch.nn.functional as F

def get_dirs_or_files(input_data: Union[str, Path]) -> Tuple[Path, Path]:
    """Get image and label directories from input path."""
    input_data = Path(input_data)
    image_input = input_data / 'images'
    labels_input = input_data / 'labels'
    
    # Check if directories exist
    assert image_input.is_dir(), f"{image_input} is not a directory"
    assert labels_input.is_dir(), f"{labels_input} is not a directory"
    
    return image_input, labels_input

def resize_image(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image to target dimensions."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

def otsu_copy(img: np.ndarray) -> np.ndarray:
    """Apply Otsu thresholding to image."""
    img_r = np.zeros(img.shape)
    img1 = img[:, :, 0]
    img2 = img[:, :, 1]
    img3 = img[:, :, 2]
    _, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_r[:, :, 0] = threshold1
    img_r[:, :, 1] = threshold1
    img_r[:, :, 2] = threshold1
    return img_r

def do_brightening(img_path: str, factor: float) -> np.ndarray:
    """Apply brightness adjustment to image."""
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img = img * factor
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def do_degrading(img: np.ndarray, scale: float) -> np.ndarray:
    """Apply degradation to image."""
    # Add noise
    noise = np.random.normal(0, scale, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    # Add blur
    kernel_size = int(scale * 5) * 2 + 1
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

def rotation_not_90_func(img: np.ndarray, mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rotation to image and mask."""
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    mask_rotated = cv2.warpAffine(mask, rotation_matrix, (width, height))
    return img_rotated, mask_rotated

def return_shuffled_channels(img: np.ndarray, shuffle_index: List[int]) -> np.ndarray:
    """Shuffle image channels."""
    return img[:, :, shuffle_index]

def scale_image_for_no_patch(img: np.ndarray, mask: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Scale image and mask."""
    height, width = img.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    img_scaled = cv2.resize(img, (new_width, new_height))
    mask_scaled = cv2.resize(mask, (new_width, new_height))
    return img_scaled, mask_scaled

def shift_image_and_label(img: np.ndarray, mask: np.ndarray, shift_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Shift image and mask."""
    height, width = img.shape[:2]
    if shift_type == 'xpos':
        shift = (width // 4, 0)
    elif shift_type == 'xmin':
        shift = (-width // 4, 0)
    elif shift_type == 'ypos':
        shift = (0, height // 4)
    elif shift_type == 'ymin':
        shift = (0, -height // 4)
    elif shift_type == 'xypos':
        shift = (width // 4, height // 4)
    else:  # xymin
        shift = (-width // 4, -height // 4)
    
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img_shifted = cv2.warpAffine(img, matrix, (width, height))
    mask_shifted = cv2.warpAffine(mask, matrix, (width, height))
    return img_shifted, mask_shifted

def preprocess_and_augment_data(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Dict,
    task: str = "segmentation"
) -> None:
    """
    Preprocess and augment data before training.
    
    Args:
        input_dir: Directory containing input data
        output_dir: Directory to save preprocessed data
        config: Configuration dictionary
        task: Task type (segmentation, enhancement, etc.)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output subdirectories
    train_img_dir = output_dir / 'train' / 'images'
    train_mask_dir = output_dir / 'train' / 'labels'
    eval_img_dir = output_dir / 'eval' / 'images'
    eval_mask_dir = output_dir / 'eval' / 'labels'
    
    for d in [train_img_dir, train_mask_dir, eval_img_dir, eval_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get image and label directories
    image_dir, mask_dir = get_dirs_or_files(input_dir)
    
    # Get all image files (both jpg and png)
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Preprocessing images"):
        img_name = img_file.stem
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
            
        mask_file = mask_dir / f"{img_name}.png"
        if not mask_file.exists():
            print(f"Warning: No mask found for {img_name}")
            continue
            
        mask = cv2.imread(str(mask_file))
        if mask is None:
            print(f"Warning: Could not read mask {mask_file}")
            continue
        
        # Resize to target dimensions
        img = resize_image(img, config['input_height'], config['input_width'])
        mask = resize_image(mask, config['input_height'], config['input_width'])
        
        # Save original image and mask
        cv2.imwrite(str(train_img_dir / f"{img_name}.png"), img)
        cv2.imwrite(str(train_mask_dir / f"{img_name}.png"), mask)
        
        if config.get('augmentation', False):
            # Apply augmentations
            if config.get('flip_aug', False):
                for flip_code in config.get('flip_index', [0, 1, -1]):
                    img_flipped = cv2.flip(img, flip_code)
                    mask_flipped = cv2.flip(mask, flip_code)
                    cv2.imwrite(str(train_img_dir / f"{img_name}_flip_{flip_code}.png"), img_flipped)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_flip_{flip_code}.png"), mask_flipped)
            
            if config.get('blur_aug', False):
                for kernel_size in config.get('blur_k', [3, 5]):
                    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                    cv2.imwrite(str(train_img_dir / f"{img_name}_blur_{kernel_size}.png"), img_blurred)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_blur_{kernel_size}.png"), mask)
            
            if config.get('brightening', False):
                for factor in config.get('brightness', [1.2, 1.5]):
                    img_bright = do_brightening(str(img_file), factor)
                    img_bright = resize_image(img_bright, config['input_height'], config['input_width'])
                    cv2.imwrite(str(train_img_dir / f"{img_name}_bright_{factor}.png"), img_bright)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_bright_{factor}.png"), mask)
            
            if config.get('binarization', False):
                img_bin = otsu_copy(img)
                cv2.imwrite(str(train_img_dir / f"{img_name}_bin.png"), img_bin)
                cv2.imwrite(str(train_mask_dir / f"{img_name}_bin.png"), mask)
            
            if config.get('degrading', False):
                for scale in config.get('degrade_scales', [0.1, 0.2]):
                    img_degraded = do_degrading(img, scale)
                    cv2.imwrite(str(train_img_dir / f"{img_name}_degrade_{scale}.png"), img_degraded)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_degrade_{scale}.png"), mask)
            
            if config.get('rotation_not_90', False):
                for angle in config.get('thetha', [30, 60, 120, 150]):
                    img_rot, mask_rot = rotation_not_90_func(img, mask, angle)
                    cv2.imwrite(str(train_img_dir / f"{img_name}_rot_{angle}.png"), img_rot)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_rot_{angle}.png"), mask_rot)
            
            if config.get('channels_shuffling', False):
                for shuffle_idx in config.get('shuffle_indexes', [[1, 0, 2], [2, 1, 0]]):
                    img_shuffled = return_shuffled_channels(img, shuffle_idx)
                    cv2.imwrite(str(train_img_dir / f"{img_name}_shuffle_{shuffle_idx}.png"), img_shuffled)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_shuffle_{shuffle_idx}.png"), mask)
            
            if config.get('scaling', False):
                for scale in config.get('scales', [0.8, 0.9, 1.1, 1.2]):
                    img_scaled, mask_scaled = scale_image_for_no_patch(img, mask, scale)
                    cv2.imwrite(str(train_img_dir / f"{img_name}_scale_{scale}.png"), img_scaled)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_scale_{scale}.png"), mask_scaled)
            
            if config.get('shifting', False):
                for shift_type in ['xpos', 'xmin', 'ypos', 'ymin', 'xypos', 'xymin']:
                    img_shifted, mask_shifted = shift_image_and_label(img, mask, shift_type)
                    cv2.imwrite(str(train_img_dir / f"{img_name}_shift_{shift_type}.png"), img_shifted)
                    cv2.imwrite(str(train_mask_dir / f"{img_name}_shift_{shift_type}.png"), mask_shifted)

class SegmentationDataset(Dataset):
    """Dataset class for image segmentation tasks."""
    
    def __init__(self, data_dir: Union[str, Path], transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (448, 448), num_classes: int = 2):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing 'images' and 'labels' subdirectories
            transform: Optional transform to apply to images and masks
            image_size: Target size for images and masks
            num_classes: Number of classes in segmentation masks
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'labels'
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Get list of image files (both jpg and png)
        self.image_files = sorted(
            list(self.image_dir.glob('*.jpg')) + 
            list(self.image_dir.glob('*.png'))
        )
        
        if not self.image_files:
            raise ValueError(f"No image files found in {self.image_dir}")
            
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Load mask
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image {image_path.name}")
            
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Resize image and mask to target dimensions
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to class indices (Long tensor)
        mask = mask.long()
        
        # Ensure consistent dimensions
        if image.shape[1:] != self.image_size:
            # Add channel dimension if missing
            if len(image.shape) == 2:
                image = image.unsqueeze(0)
            image = F.interpolate(image.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)
        
        if mask.shape[1:] != self.image_size:
            # Ensure mask is 2D (H, W) for CrossEntropyLoss
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            # Add batch and channel dimensions for interpolation
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                               size=self.image_size, 
                               mode='nearest').squeeze(0).squeeze(0).long()
        
        # Ensure mask is 2D (H, W) for CrossEntropyLoss
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        return image, mask

class DataModule:
    """Data module for handling training and validation datasets."""
    
    def __init__(self,
                 train_dir: Union[str, Path],
                 val_dir: Optional[Union[str, Path]] = None,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 image_size: Tuple[int, int] = (448, 448),
                 num_classes: int = 2,
                 config: Optional[Dict] = None,
                 task: str = "segmentation"):
        """
        Initialize the data module.
        
        Args:
            train_dir: Directory containing training data ('images' and 'labels' subdirectories)
            val_dir: Directory containing validation data ('images' and 'labels' subdirectories)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            image_size: Target size for images and masks
            num_classes: Number of classes in segmentation masks
            config: Configuration dictionary for preprocessing and augmentation
            task: Task type (segmentation, enhancement, etc.)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Preprocess and augment data if config is provided
        if config is not None and not config.get('data_is_provided', False):
            preprocess_and_augment_data(train_dir, config['dir_output'], config, task)
            train_dir = Path(config['dir_output']) / 'train'
            if val_dir:
                val_dir = Path(config['dir_output']) / 'eval'
        
        # Training transforms
        self.train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transforms
        self.val_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Create datasets
        self.train_dataset = SegmentationDataset(
            data_dir=train_dir,
            transform=self.train_transform,
            image_size=image_size,
            num_classes=num_classes
        )
        
        if val_dir:
            self.val_dataset = SegmentationDataset(
                data_dir=val_dir,
                transform=self.val_transform,
                image_size=image_size,
                num_classes=num_classes
            )
        else:
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# Example usage:
if __name__ == "__main__":
    from config_loader import ConfigLoader
    
    # Load configuration
    config_loader = ConfigLoader(
        config_path="config_params.json",
        page2label_path="custom_config_page2label.json",
        scales_path="scales_enhancement.json"
    )
    
    # Get model configuration
    model_config = config_loader.load_config()
    
    # Create data module
    data_module = DataModule(
        train_dir=model_config.dir_train,
        val_dir=model_config.dir_eval,
        batch_size=8,
        num_workers=4,
        image_size=(model_config.input_height, model_config.input_width),
        num_classes=model_config.n_classes
    )
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Test dataloaders
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        if batch_idx == 2:  # Test only first 3 batches
            break 