import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config_loader import ConfigLoader
from models_pytorch import create_model
from dataset import DataModule
from training import Trainer, IoU, DiceScore, Accuracy
from initialization import initialize_weights, WeightRegularizer
from model_utils import save_model, load_model, save_checkpoint

# Set CUDA matmul precision for better performance
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

class SaveWeightsAfterSteps:
    """Callback to save model weights after specified number of steps."""
    def __init__(self, save_interval: int, save_path: Union[str, Path], config: Dict):
        self.save_interval = save_interval
        self.save_path = Path(save_path)
        self.step_count = 0
        self.config = config
        self.save_path.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
        self.step_count += 1
        if self.step_count % self.save_interval == 0:
            save_path = self.save_path / f"model_step_{self.step_count}"
            save_path.mkdir(exist_ok=True)
            
            # Save model state
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=None,  # Current loss not available in batch end
                save_dir=save_path,
                config=self.config
            )
            print(f"Saved model at step {self.step_count} to {save_path}")

def get_dirs_or_files(input_data: Union[str, Path]) -> Tuple[Path, Path]:
    """Get image and label directories from input path."""
    input_data = Path(input_data)
    image_input = input_data / 'images'
    labels_input = input_data / 'labels'
    
    # Check if directories exist
    assert image_input.is_dir(), f"{image_input} is not a directory"
    assert labels_input.is_dir(), f"{labels_input} is not a directory"
    
    return image_input, labels_input

def setup_logging(output_dir: Union[str, Path]) -> logging.Logger:
    """Set up logging configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = output_dir / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_loss_function(task: str, num_classes: int, weighted: bool = False, weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Get appropriate loss function based on task."""
    if task == "segmentation":
        if num_classes > 1:
            if weighted and weights is not None:
                return nn.CrossEntropyLoss(weight=weights)
            return nn.CrossEntropyLoss()
        else:
            return nn.BCEWithLogitsLoss()
    elif task == "classification":
        if weighted and weights is not None:
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()
    elif task == "enhancement":
        return nn.MSELoss()
    elif task == "reading_order":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task: {task}")

def get_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Get optimizer based on configuration."""
    if config.get('optimizer', 'adam').lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-6)
        )
    elif config.get('optimizer', 'adam').lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-6)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.get('optimizer')}")

def calculate_class_weights(data_dir: Union[str, Path], num_classes: int) -> torch.Tensor:
    """Calculate class weights for weighted loss."""
    data_dir = Path(data_dir)
    weights = torch.zeros(num_classes)
    
    for mask_file in data_dir.glob('*.png'):
        try:
            mask = torch.from_numpy(np.array(Image.open(mask_file)))
            for c in range(num_classes):
                weights[c] += (mask == c).sum().item()
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
            continue
    
    weights = 1.0 / (weights + 1e-6)  # Add small epsilon to avoid division by zero
    weights = weights / weights.sum()
    weights = weights / weights.min()
    weights = weights / weights.sum()
    
    return weights

class SegmentationModel(pl.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, config: Dict):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        self.save_hyperparameters(ignore=['model', 'criterion'])
        
        # Initialize metrics
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.train_iou = IoU(num_classes=config['n_classes'])
        self.val_iou = IoU(num_classes=config['n_classes'])
        self.train_dice = DiceScore(num_classes=config['n_classes'])
        self.val_dice = DiceScore(num_classes=config['n_classes'])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Convert outputs to predictions for metrics
        predictions = outputs.argmax(dim=1)
        
        # Update metrics
        self.train_accuracy.update(predictions, masks)
        self.train_iou.update(predictions, masks)
        self.train_dice.update(predictions, masks)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy.compute(), prog_bar=True)
        self.log('train_iou', self.train_iou.compute(), prog_bar=True)
        self.log('train_dice', self.train_dice.compute(), prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Convert outputs to predictions for metrics
        predictions = outputs.argmax(dim=1)
        
        # Update metrics
        self.val_accuracy.update(predictions, masks)
        self.val_iou.update(predictions, masks)
        self.val_dice.update(predictions, masks)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy.compute(), prog_bar=True)
        self.log('val_iou', self.val_iou.compute(), prog_bar=True)
        self.log('val_dice', self.val_dice.compute(), prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # Reset metrics at the end of each epoch
        self.train_accuracy.reset()
        self.train_iou.reset()
        self.train_dice.reset()
    
    def on_validation_epoch_end(self):
        # Reset metrics at the end of each epoch
        self.val_accuracy.reset()
        self.val_iou.reset()
        self.val_dice.reset()
    
    def configure_optimizers(self):
        if self.config.get('optimizer', 'adam').lower() == 'adam':
            return optim.Adam(
                self.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-6)
            )
        elif self.config.get('optimizer', 'adam').lower() == 'sgd':
            return optim.SGD(
                self.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-6)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.get('optimizer')}")

def get_arg_with_priority(args: argparse.Namespace, config: Dict, arg_name: str, default_value: any) -> any:
    """
    Get argument value with priority order:
    1. Command line argument (if provided)
    2. Config file value (if exists)
    3. Default value
    """
    # Check if command line argument was provided (not None)
    if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
        return getattr(args, arg_name)
    
    # Check if value exists in config
    if arg_name in config:
        return config[arg_name]
    
    # Return default value
    if arg_name == 'batch_size':
        return 16  # Default batch size for RTX 3060
    elif arg_name == 'num_workers':
        return 4  # Default number of workers
    return default_value

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting training...")
    
    # Load configuration
    config_loader = ConfigLoader(
        config_path=args.config,
        page2label_path=args.page2label
    )
    config = config_loader.load_config()
    logger.info(f"Loaded configuration: {config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parameters with priority
    batch_size = get_arg_with_priority(args, config.__dict__, 'batch_size', 8)
    num_workers = get_arg_with_priority(args, config.__dict__, 'num_workers', 4)
    epochs = get_arg_with_priority(args, config.__dict__, 'n_epochs', 100)
    patience = get_arg_with_priority(args, config.__dict__, 'patience', 10)
    device = get_arg_with_priority(args, config.__dict__, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using parameters: batch_size={batch_size}, num_workers={num_workers}, "
                f"epochs={epochs}, patience={patience}, device={device}")
    
    # Save configuration with resolved values
    config_dict = config.__dict__.copy()
    config_dict.update({
        'batch_size': batch_size,
        'num_workers': num_workers,
        'n_epochs': epochs,
        'patience': patience,
        'device': device
    })
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model: {model.__class__.__name__}")
    
    # Initialize weights
    initialize_weights(model)
    logger.info("Initialized model weights")
    
    # Calculate class weights if needed
    weights = None
    if config.weighted_loss:
        weights = calculate_class_weights(config.dir_train, config.n_classes)
        logger.info(f"Calculated class weights: {weights}")
    
    # Get loss function
    criterion = get_loss_function(config.task, config.n_classes, config.weighted_loss, weights)
    logger.info(f"Using loss function: {criterion.__class__.__name__}")
    
    # Create Lightning model
    lightning_model = SegmentationModel(model, criterion, config_dict)
    logger.info("Created Lightning model")
    
    # Create data module
    data_module = DataModule(
        train_dir=config.dir_train,
        val_dir=config.dir_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=(config.input_height, config.input_width),
        num_classes=config.n_classes,
        config=config_dict,
        task=config.task
    )
    logger.info("Created data module")
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=device,
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir,
                filename='model-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                verbose=True
            ),
            ModelCheckpoint(
                dirpath=output_dir,
                filename='model-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_acc',
                mode='max',
                verbose=True
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='min',
                verbose=True,
                min_delta=0.0
            ),
            EarlyStopping(
                monitor='val_acc',
                patience=patience,
                mode='max',
                verbose=True,
                min_delta=0.0
            )
        ],
        logger=TensorBoardLogger(output_dir, name='lightning_logs'),
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1
    )
    
    # Train model
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=args.resume)
    else:
        trainer.fit(lightning_model, train_loader, val_loader)
    
    # Save final model
    final_model_path = output_dir / 'model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': lightning_model.optimizers().state_dict(),
        'epoch': trainer.current_epoch,
        'val_loss': trainer.callback_metrics['val_loss'].item(),
        'val_acc': trainer.callback_metrics['val_acc'].item()
    }, final_model_path)
    logger.info(f"Training completed and final model saved to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--page2label', type=str, required=True, help='Path to page2label config file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config if provided)')
    parser.add_argument('--num-workers', type=int, help='Number of workers (overrides config if provided)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config if provided)')
    parser.add_argument('--patience', type=int, help='Early stopping patience (overrides config if provided)')
    parser.add_argument('--device', type=str, help='Device to use (overrides config if provided)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()

    
