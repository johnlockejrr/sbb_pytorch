import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Union, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import json
from abc import ABC, abstractmethod

class Metric(ABC):
    """Abstract base class for metrics."""
    
    @abstractmethod
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute metric value."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass

class Accuracy(Metric):
    """Accuracy metric for classification tasks."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset(self) -> None:
        self.correct = 0
        self.total = 0

class IoU(Metric):
    """Intersection over Union metric for segmentation tasks."""
    
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # Create confusion matrix
        mask = (targets != self.ignore_index) if self.ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += ((predictions == i) & (targets == j) & mask).sum().item()
    
    def compute(self) -> float:
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1) - intersection
        iou = intersection / (union + 1e-8)
        return iou.mean().item()
    
    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

class DiceScore(Metric):
    """Dice coefficient metric for segmentation tasks."""
    
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # Convert to one-hot encoding for multi-class
        pred_one_hot = torch.zeros((predictions.size(0), self.num_classes, *predictions.size()[1:]), 
                                 device=predictions.device)
        pred_one_hot.scatter_(1, predictions.unsqueeze(1), 1)
        
        target_one_hot = torch.zeros((targets.size(0), self.num_classes, *targets.size()[1:]), 
                                   device=targets.device)
        target_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        self.intersection += (pred_one_hot * target_one_hot).sum().item()
        self.union += pred_one_hot.sum().item() + target_one_hot.sum().item()
    
    def compute(self) -> float:
        return (2. * self.intersection + self.smooth) / (self.union + self.smooth)
    
    def reset(self) -> None:
        self.intersection = 0
        self.union = 0

class Callback(ABC):
    """Abstract base class for callbacks."""
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, float]) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Dict[str, float]) -> None:
        """Called at the beginning of each batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Dict[str, float]) -> None:
        """Called at the end of each batch."""
        pass

class ModelCheckpoint(Callback):
    """Callback to save model checkpoints."""
    
    def __init__(self,
                 filepath: Union[str, Path],
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = False):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        if self.mode == 'min':
            is_better = current_value < self.best_value
        else:
            is_better = current_value > self.best_value
        
        if is_better:
            self.best_value = current_value
            if self.save_best_only:
                self.save_model(epoch, logs)
    
    def save_model(self, epoch: int, logs: Dict[str, float]) -> None:
        """Save model state."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if self.save_weights_only:
            torch.save(self.model.state_dict(), self.filepath)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'logs': logs
            }, self.filepath)

class Trainer:
    """Training manager class."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 metrics: Optional[List[Metric]] = None,
                 callbacks: Optional[List[Callback]] = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        
        # Set model to device
        self.model.to(device)
    
    def train_step(self, batch):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        logs = {}
        
        # Reset metrics
        for metric in self.metrics:
            metric.reset()
        
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc='Training')):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for metric in self.metrics:
                metric.update(outputs, targets)
            
            # Callbacks
            logs = {'loss': loss.item()}
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, logs)
        
        # Compute final metrics
        logs['loss'] = epoch_loss / len(dataloader)
        for metric in self.metrics:
            logs[metric.__class__.__name__.lower()] = metric.compute()
        
        return logs
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        logs = {}
        
        # Reset metrics
        for metric in self.metrics:
            metric.reset()
        
        # Validation loop
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc='Validation')):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                for metric in self.metrics:
                    metric.update(outputs, targets)
        
        # Compute final metrics
        logs['val_loss'] = val_loss / len(dataloader)
        for metric in self.metrics:
            logs[f'val_{metric.__class__.__name__.lower()}'] = metric.compute()
        
        return logs
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 100) -> Dict[str, List[float]]:
        """Train the model."""
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Callbacks
            logs = {}
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, logs)
            
            # Training
            train_logs = self.train_epoch(train_loader)
            history['loss'].append(train_logs['loss'])
            
            # Validation
            if val_loader is not None:
                val_logs = self.validate(val_loader)
                history['val_loss'].append(val_logs['val_loss'])
                logs.update(val_logs)
            
            logs.update(train_logs)
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_logs['loss']:.4f}")
            if val_loader is not None:
                print(f"Val Loss: {val_logs['val_loss']:.4f}")
        
        return history

# Example usage:
if __name__ == "__main__":
    from models_pytorch import create_model
    from config_loader import ConfigLoader
    
    # Load configuration
    config_loader = ConfigLoader(
        config_path="config_params.json",
        page2label_path="custom_config_page2label.json",
        scales_path="scales_enhancement.json"
    )
    
    # Get model configuration
    model_config = config_loader.load_config()
    
    # Create model
    model = create_model(model_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        metrics=[Accuracy(), IoU(num_classes=model_config.n_classes)],
        callbacks=[
            ModelCheckpoint('checkpoints/model.pth'),
        ]
    )
    
    # Train model
    history = trainer.train(
        train_loader=None,  # Add your DataLoader here
        val_loader=None,    # Add your DataLoader here
        epochs=100
    ) 