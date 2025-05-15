import sys
import os
import numpy as np
import warnings
import cv2
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import click
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
from models_pytorch import ResNet50UNet, ResNet50Classifier, MachineBasedReadingOrderModel
from config_loader import ConfigLoader, TaskType
from dataset import resize_image

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

class SBBPredict:
    def __init__(self, image: str, model_dir: str, task: str, config_params: Dict, 
                 patches: bool = False, save: Optional[str] = None, 
                 save_layout: Optional[str] = None, ground_truth: Optional[str] = None,
                 xml_file: Optional[str] = None, out: Optional[str] = None, 
                 min_area: Optional[float] = None):
        self.image = image
        self.patches = patches
        self.save = save
        self.save_layout = save_layout
        self.model_dir = model_dir
        self.ground_truth = ground_truth
        self.task = task
        self.config_params = config_params
        self.xml_file = xml_file
        self.out = out
        self.min_area = float(min_area) if min_area else 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configuration
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

    def load_model(self) -> nn.Module:
        """Load the appropriate model based on task."""
        print(f"Loading model for task: {self.task}")
        print(f"Model directory: {self.model_dir}")
        print(f"Config parameters: {self.config_params}")
        
        if self.task == "classification":
            model = ResNet50Classifier(
                n_classes=len(self.config_params['classification_classes_name']),
                input_height=self.config_params['input_height'],
                input_width=self.config_params['input_width']
            )
        elif self.task == "reading_order":
            model = MachineBasedReadingOrderModel(
                n_classes=2,  # Binary classification for reading order
                input_height=self.config_params['input_height'],
                input_width=self.config_params['input_width']
            )
        else:  # segmentation, binarization, enhancement
            model = ResNet50UNet(
                n_classes=self.config_params['n_classes'],
                input_height=self.config_params['input_height'],
                input_width=self.config_params['input_width'],
                task=TaskType(self.task)
            )
        
        # Load model weights
        model_path = os.path.join(self.model_dir, 'model.pt')
        print(f"Loading model weights from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print("Model loaded successfully!")
        return model

    def color_images(self, seg: np.ndarray) -> np.ndarray:
        """Color the segmentation output."""
        if len(np.shape(seg)) == 3:
            seg = seg[:,:,0]
            
        seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(np.uint8)
        
        # Define colors for each class
        colors = {
            0: [255, 255, 255],  # Background (white)
            1: [255, 0, 0],      # Class 1 (red)
            2: [0, 0, 255]       # Class 2 (blue)
        }
        
        # Color each class
        for class_idx, color in colors.items():
            mask = (seg == class_idx)
            seg_img[mask] = color
            
        return seg_img

    def visualize_model_output(self, prediction: np.ndarray, img: np.ndarray, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Visualize model predictions."""
        if task == "binarization":
            prediction = prediction * -1
            prediction = prediction + 1
            added_image = prediction * 255
            layout_only = None
        else:
            # Create a colored version of the prediction
            layout_only = self.color_images(prediction)
            
            # Resize original image to match prediction size
            img = resize_image(img, layout_only.shape[0], layout_only.shape[1])
            
            # Convert to float for proper blending
            img = img.astype(np.float32)
            layout_only = layout_only.astype(np.float32)
            
            # Blend the images
            added_image = cv2.addWeighted(img, 0.7, layout_only, 0.3, 0)
            added_image = added_image.astype(np.uint8)
            
        return added_image, layout_only

    def calculate_iou(self, Yi: np.ndarray, y_predi: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        IoUs = []
        Nclass = np.unique(Yi)
        for c in Nclass:
            TP = np.sum((Yi == c) & (y_predi == c))
            FP = np.sum((Yi != c) & (y_predi == c))
            FN = np.sum((Yi == c) & (y_predi != c))
            IoU = TP / float(TP + FP + FN)
            if self.config_params['n_classes'] > 2:
                print(f"class {c:02.0f}: #TP={TP:6.0f}, #FP={FP:6.0f}, #FN={FN:5.0f}, IoU={IoU:4.3f}")
            IoUs.append(IoU)
        
        if self.config_params['n_classes'] > 2:
            mIoU = np.mean(IoUs)
            print("_________________")
            print(f"Mean IoU: {mIoU:4.3f}")
            return mIoU
        elif self.config_params['n_classes'] == 2:
            mIoU = IoUs[1]
            print("_________________")
            print(f"IoU: {mIoU:4.3f}")
            return mIoU

    def predict(self) -> np.ndarray:
        """Run inference on the input image."""
        if self.task == 'classification':
            # Load and preprocess image
            img = cv2.imread(self.image, 0)
            img = img / 255.0
            img = resize_image(img, self.config_params['input_height'], self.config_params['input_width'])
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                pred_class = torch.argmax(output, dim=1).item()
            
            print(f"Predicted Class: {self.config_params['classification_classes_name'][str(pred_class)]}")
            return None
            
        elif self.task == 'reading_order':
            # Reading order prediction logic here
            # This would need to be implemented based on your specific requirements
            pass
            
        else:  # segmentation, binarization, enhancement
            print(f"Loading image: {self.image}")
            img = cv2.imread(self.image)
            if img is None:
                raise ValueError(f"Could not read image: {self.image}")
            print(f"Image loaded successfully. Shape: {img.shape}")
            self.img_org = np.copy(img)
            
            if self.patches:
                # Implement patch-based prediction
                # This would need to be implemented based on your specific requirements
                pass
            else:
                # Single image prediction
                # Normalize image to [0, 1] range
                img = img.astype(np.float32) / 255.0
                
                # Apply ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std
                
                img = resize_image(img, self.config_params['input_height'], self.config_params['input_width'])
                print(f"Resized image shape: {img.shape}")
                
                # Convert to tensor
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                print(f"Input tensor shape: {img_tensor.shape}")
                print(f"Input tensor min/max: {img_tensor.min().item():.3f}/{img_tensor.max().item():.3f}")
                
                # Ensure model is in eval mode
                self.model.eval()
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(img_tensor)
                    print(f"Model output shape: {output.shape}")
                    print(f"Model output min/max: {output.min().item():.3f}/{output.max().item():.3f}")
                    
                    # Get probabilities
                    probs = torch.softmax(output, dim=1)
                    print(f"Probabilities min/max: {probs.min().item():.3f}/{probs.max().item():.3f}")
                    
                    if self.task == 'enhancement':
                        seg = output[0].cpu().numpy().transpose(1, 2, 0)
                        seg = seg * 255
                    else:  # segmentation or binarization
                        seg = torch.argmax(output, dim=1)[0].cpu().numpy()
                        print(f"Unique values in prediction: {np.unique(seg)}")
                        print(f"Class distribution: {np.bincount(seg.flatten())}")
                        seg = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
                
                prediction_true = seg.astype(int)
                prediction_true = resize_image(prediction_true, self.img_org.shape[0], self.img_org.shape[1])
                print(f"Final prediction shape: {prediction_true.shape}")
                print(f"Unique values in final prediction: {np.unique(prediction_true)}")
                return prediction_true

    def run(self) -> None:
        """Run the complete inference pipeline."""
        res = self.predict()
        
        if self.task in ['classification', 'reading_order']:
            return
            
        if self.task == 'enhancement':
            if self.save:
                cv2.imwrite(self.save, res)
        else:
            img_seg_overlayed, only_layout = self.visualize_model_output(res, self.img_org, self.task)
            if self.save:
                cv2.imwrite(self.save, img_seg_overlayed)
            if self.save_layout:
                cv2.imwrite(self.save_layout, only_layout)
                
        if self.ground_truth:
            gt_img = cv2.imread(self.ground_truth)
            self.calculate_iou(gt_img[:,:,0], res[:,:,0])

@click.command()
@click.option(
    "--image",
    "-i",
    help="image filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="output directory where xml with detected reading order will be written.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--patches/--no-patches",
    "-p/-nop",
    is_flag=True,
    help="if this parameter set to true, this tool will try to do inference in patches.",
)
@click.option(
    "--save",
    "-s",
    help="save prediction as a png file in current folder.",
)
@click.option(
    "--save_layout",
    "-sl",
    help="save layout prediction only as a png file in current folder.",
)
@click.option(
    "--model",
    "-m",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--ground_truth",
    "-gt",
    help="ground truth directory if you want to see the iou of prediction.",
)
@click.option(
    "--xml_file",
    "-xml",
    help="xml file with layout coordinates that reading order detection will be implemented on. The result will be written in the same xml file.",
)
@click.option(
    "--min_area",
    "-min",
    help="min area size of regions considered for reading order detection. The default value is zero and means that all text regions are considered for reading order.",
)
def main(image: str, model: str, patches: bool, save: Optional[str], save_layout: Optional[str],
         ground_truth: Optional[str], xml_file: Optional[str], out: Optional[str], min_area: Optional[str]) -> None:
    """Main entry point for inference."""
    # Load model configuration
    with open(os.path.join(model, 'config.json')) as f:
        config_params_model = json.load(f)
    
    task = config_params_model['task']
    if task not in ['classification', 'reading_order'] and not save:
        print("Error: You used one of segmentation or binarization task but not set -s, you need a filename to save visualized output with -s")
        sys.exit(1)
    
    predictor = SBBPredict(
        image=image,
        model_dir=model,
        task=task,
        config_params=config_params_model,
        patches=patches,
        save=save,
        save_layout=save_layout,
        ground_truth=ground_truth,
        xml_file=xml_file,
        out=out,
        min_area=min_area
    )
    predictor.run()

if __name__ == "__main__":
    main() 
