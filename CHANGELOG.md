# SBB PyTorch Changelog

This document provides a comprehensive overview of the changes made when porting the SBB (Staatsbibliothek zu Berlin) pixelwise segmentation project from TensorFlow 2.12.1 to PyTorch.

## Overview

The original `sbb_pixelwise_segmentation` project was a TensorFlow-based document analysis system for pixelwise segmentation. The `sbb_pixelwise_segmentation_pytorch` directory contains a complete PyTorch refactor that not only ports the existing functionality but also adds significant improvements and new features.

## Major Architectural Changes

### 1. Framework Migration
- **From**: TensorFlow 2.12.1 with Keras
- **To**: PyTorch 2.0+ with PyTorch Lightning
- **Impact**: Complete rewrite of all model definitions, training loops, and data handling

### 2. Model Architecture Overhaul

#### Original TensorFlow Models (`models.py`)
- Custom ResNet50 implementation with manual layer definitions
- Two variants: `resnet50_unet()` and `resnet50_unet_light()`
- Manual ResNet50 blocks (identity_block, conv_block)
- Custom UNet decoder implementation
- Hardcoded pretrained weights loading

#### New PyTorch Models (`models_pytorch.py`)
- **ResNet50UNet**: Uses `torchvision.models.resnet50` with pretrained ImageNet weights
- **ResNet50UNetLight**: Lighter version with reduced decoder channels
- **ResNet50Classifier**: For classification tasks
- **MachineBasedReadingOrderModel**: For reading order detection
- **ViTResNet50UNet**: Vision Transformer + ResNet50 hybrid
- **ViTResNet50UNetTransformerBeforeCNN**: Alternative ViT architecture
- **EnhancementModel**: For image enhancement with attention mechanisms

#### Key Improvements:
- Modern PyTorch architecture using `torchvision.models`
- Better weight initialization
- More flexible model creation with factory pattern
- Support for multiple task types (segmentation, classification, enhancement, reading order)
- Attention mechanisms in enhancement models

### 3. Training Pipeline Revolution

#### Original TensorFlow Training (`train.py`)
- Sacred experiment framework
- Manual data generation with `data_gen()`
- Basic model compilation and training
- Simple epoch loop with `model.fit_generator()`
- Manual model saving

#### New PyTorch Training (`train.py`)
- **PyTorch Lightning Integration**: Modern training framework
- **Advanced Callbacks**: ModelCheckpoint, EarlyStopping
- **TensorBoard Logging**: Built-in experiment tracking
- **Command Line Interface**: argparse-based configuration
- **Resume Training**: Checkpoint-based training resumption
- **Advanced Metrics**: IoU, Dice Score, Accuracy tracking
- **Better Error Handling**: Comprehensive logging and validation

#### Key Improvements:
- Automatic mixed precision support
- Better GPU memory management
- Distributed training ready
- Comprehensive logging and monitoring
- Model checkpointing and early stopping
- More robust error handling

### 4. Configuration System Overhaul

#### Original Configuration (`config_params.json`)
```json
{
    "n_classes": 3,
    "n_epochs": 2,
    "input_height": 448,
    "input_width": 672,
    "weight_decay": 1e-6,
    "n_batch": 2,
    "learning_rate": 1e-4,
    "patches": true,
    "pretraining": true,
    "augmentation": false,
    // ... basic parameters
}
```

#### New Configuration System
- **Multiple Config Files**: `config_params.json`, `config_train_textline.json`, `custom_config_page2label.json`, `scales_enhancement.json`
- **Structured Configuration**: `config_loader.py` with dataclasses and enums
- **Type Safety**: Enum-based configuration validation
- **Modular Design**: Separate configs for different tasks and use cases

#### New Configuration Features:
- **Backbone Types**: transformer, resnet, nontransformer
- **Task Types**: segmentation, classification, enhancement, reading_order
- **Use Cases**: textline, layout, printspace, glyph, word
- **Advanced Augmentation**: RGB background/foreground, channel shuffling, degradation
- **Transformer Configuration**: Patch sizes, projection dimensions, attention heads
- **Device Configuration**: CUDA/CPU selection

### 5. Data Handling and Augmentation

#### Original Data Pipeline (`utils.py`)
- Basic image loading and preprocessing
- Simple augmentation functions
- Manual patch generation
- Basic data flow management

#### New Data Pipeline (`dataset.py`)
- **PyTorch Dataset**: Proper `torch.utils.data.Dataset` implementation
- **DataLoader Integration**: Efficient data loading with multiprocessing
- **Albumentations**: Professional augmentation library
- **Advanced Augmentations**:
  - RGB background/foreground replacement
  - Channel shuffling
  - Image degradation
  - Brightness adjustment
  - Elastic transformations
  - Advanced rotation and scaling

#### Key Improvements:
- Better memory efficiency
- Parallel data loading
- More sophisticated augmentations
- Proper tensor handling
- Better error handling for missing files

### 6. Loss Functions and Metrics

#### Original Loss Functions (`metrics.py`)
- Basic soft dice loss
- Weighted categorical crossentropy
- Simple accuracy metrics

#### New Loss Functions (`losses.py`)
- **DiceLoss**: Improved dice coefficient implementation
- **FocalLoss**: For handling class imbalance
- **WeightedCrossEntropyLoss**: Better weight handling
- **CombinedLoss**: Dice + CrossEntropy combination
- **SSIMLoss**: Structural similarity for enhancement tasks

#### New Metrics (`metrics.py`)
- **IoU (Intersection over Union)**: Proper segmentation metric
- **DiceScore**: Dice coefficient metric
- **Accuracy**: Classification accuracy
- **Abstract Metric Base Class**: Extensible metric system

### 7. Inference Pipeline

#### Original Inference
- Basic model loading and prediction
- Simple visualization
- Limited output options

#### New Inference (`inference_pytorch.py`)
- **Multi-task Support**: Segmentation, classification, reading order
- **Batch Processing**: Efficient batch inference
- **Advanced Visualization**: Better color mapping and output
- **XML Integration**: Reading order detection with XML output
- **Ground Truth Comparison**: IoU calculation with ground truth
- **Command Line Interface**: Click-based CLI

### 8. New Features and Capabilities

#### Completely New Features:
1. **Vision Transformer Support**: ViT-ResNet50 hybrid models
2. **Reading Order Detection**: Machine-based reading order model
3. **Image Enhancement**: Dedicated enhancement models with attention
4. **Multi-task Framework**: Support for multiple document analysis tasks
5. **Advanced Augmentation**: Professional-grade data augmentation
6. **Model Checkpointing**: Automatic model saving and resumption
7. **Experiment Tracking**: TensorBoard integration
8. **Configuration Validation**: Type-safe configuration system

#### Enhanced Features:
1. **Better Model Architecture**: Modern PyTorch implementations
2. **Improved Training**: PyTorch Lightning integration
3. **Advanced Metrics**: Comprehensive evaluation metrics
4. **Better Data Handling**: Efficient data loading and preprocessing
5. **Enhanced Visualization**: Better output visualization
6. **Robust Error Handling**: Comprehensive error handling and logging

### 9. Dependencies and Requirements

#### Original Dependencies (`requirements.txt`)
```
tensorflow == 2.12.1
sacred
opencv-python-headless
seaborn
tqdm
imutils
numpy
scipy
```

#### New Dependencies (`requirements.txt`)
```
numpy>=1.21.0
opencv-python-headless>=4.5.0
albumentations>=1.3.0
Pillow>=9.0.0
tqdm>=4.65.0
scikit-learn>=1.0.0
scipy>=1.7.0
seaborn>=0.12.0
shapely>=2.0.0
click>=8.0.0
sacred>=0.8.0
# PyTorch and PyTorch Lightning (version-specific installation)
```

### 10. File Structure Changes

#### New Files Added:
- `config_loader.py`: Configuration management system
- `dataset.py`: PyTorch dataset and data loading
- `losses.py`: Advanced loss functions
- `training.py`: Training utilities and callbacks
- `inference_pytorch.py`: PyTorch inference pipeline
- `model_utils.py`: Model utilities and checkpointing
- `initialization.py`: Weight initialization utilities
- `augmentation.py`: Advanced augmentation functions
- `gt_gen_utils.py`: Ground truth generation utilities
- `generate_gt_for_training.py`: Training data generation
- `custom_config_page2label.json`: Page-to-label configuration
- `config_train_textline.json`: Textline training configuration
- `scales_enhancement.json`: Enhancement scales configuration
- `train.md`: Training documentation

#### Modified Files:
- `models.py` → `models_pytorch.py`: Complete rewrite
- `train.py`: Complete rewrite with PyTorch Lightning
- `utils.py`: Enhanced with new utility functions
- `metrics.py`: Enhanced with new metrics
- `config_params.json`: Enhanced configuration
- `requirements.txt`: Updated dependencies

### 11. Performance Improvements

#### Training Performance:
- **Faster Data Loading**: PyTorch DataLoader with multiprocessing
- **Memory Efficiency**: Better GPU memory management
- **Mixed Precision**: Automatic mixed precision training
- **Distributed Training**: Ready for multi-GPU training

#### Model Performance:
- **Better Initialization**: Improved weight initialization
- **Modern Architecture**: Using torchvision pretrained models
- **Attention Mechanisms**: Enhanced model capabilities
- **Multi-task Learning**: Support for multiple related tasks

### 12. Code Quality Improvements

#### Architecture:
- **Object-Oriented Design**: Better class structure
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Robust error handling and validation
- **Documentation**: Better code documentation

#### Maintainability:
- **Modular Design**: Separated concerns
- **Configuration Management**: Centralized configuration
- **Logging**: Comprehensive logging system
- **Testing**: Better structure for unit testing

## Migration Guide

### For Users:
1. **Installation**: Use the new requirements.txt with PyTorch
2. **Configuration**: Use the new configuration system with multiple config files
3. **Training**: Use the new command-line interface with PyTorch Lightning
4. **Inference**: Use the new inference script with enhanced features

### For Developers:
1. **Model Development**: Use the new model factory pattern
2. **Data Pipeline**: Use the new Dataset and DataLoader classes
3. **Training**: Use PyTorch Lightning for new training scripts
4. **Configuration**: Use the new configuration loader system

## Breaking Changes

1. **Framework**: Complete migration from TensorFlow to PyTorch
2. **API**: New command-line interface and configuration system
3. **Model Loading**: Different model checkpoint format
4. **Data Format**: Enhanced data preprocessing pipeline
5. **Configuration**: New configuration file structure

## Future Enhancements

The PyTorch version provides a solid foundation for future enhancements:
- **Multi-GPU Training**: Distributed training support
- **Model Optimization**: Quantization and pruning
- **Advanced Architectures**: More transformer-based models
- **Real-time Inference**: Optimized inference pipeline
- **Web Interface**: REST API and web interface
- **Cloud Integration**: Cloud training and deployment

## Conclusion

The PyTorch refactor represents a significant improvement over the original TensorFlow implementation, providing:
- Modern deep learning framework
- Better performance and scalability
- Enhanced features and capabilities
- Improved code quality and maintainability
- Future-ready architecture

This migration not only preserves all original functionality but also adds substantial new capabilities and improvements, making it a comprehensive upgrade to the SBB document analysis system.
