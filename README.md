# SBB PyTorch

This project is a PyTorch implementation of the original SBB (Staatsbibliothek zu Berlin) document analysis system, ported from TensorFlow 2.12.1. The system provides tools for document image analysis, including segmentation, classification, and reading order detection.

## Changelog

### Major Changes
- Ported entire codebase from TensorFlow 2.12.1 to PyTorch
- Replaced TensorFlow-specific layers and operations with PyTorch equivalents
- Implemented custom PyTorch models:
  - ResNet50UNet for segmentation
  - ResNet50Classifier for classification
  - MachineBasedReadingOrderModel for reading order detection
- Added PyTorch Lightning integration for better training management
- Implemented custom metrics (IoU, Dice Score, Accuracy)
- Added support for model checkpointing and early stopping
- Improved logging and visualization capabilities

### Model Architecture Changes
- Replaced TensorFlow's ResNet50 implementation with PyTorch's torchvision.models
- Implemented custom UNet decoder in PyTorch
- Added support for multi-class segmentation
- Improved model initialization and weight loading

### Training Pipeline Changes
- Implemented PyTorch DataLoader for efficient data loading
- Added support for data augmentation
- Implemented custom loss functions for different tasks
- Added support for weighted loss for imbalanced datasets
- Improved training monitoring with TensorBoard integration

### Inference Pipeline Changes
- Implemented PyTorch-based inference pipeline
- Added support for batch processing
- Improved visualization of predictions
- Added support for model checkpoint loading

## Installation

1. Clone the repository:
```bash
git clone https://github.com/johnlockejrr/sbb_pytorch.git
cd sbb_pytorch
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv sbb_pytorch-env
source sbb_pytorch-env/bin/activate  # On Windows: sbb_pytorch-env\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

To train a model, use the following command:

```bash
python train.py \
    --config config_train_textline.json \
    --page2label page2label.json \
    --output-dir runs/experiment_name
```

Optional arguments:
- `--batch-size`: Override batch size from config
- `--num-workers`: Override number of workers from config
- `--epochs`: Override number of epochs from config
- `--patience`: Override early stopping patience from config
- `--device`: Override device from config (e.g., 'cuda' or 'cpu')
- `--resume`: Path to checkpoint to resume training

Example with all options:
```bash
python train.py \
    --config config_train_textline.json \
    --page2label page2label.json \
    --output-dir runs/experiment_name \
    --batch-size 8 \
    --num-workers 4 \
    --epochs 100 \
    --patience 10 \
    --device cuda \
    --resume runs/experiment_name/checkpoints/last.ckpt
```

### Inference

To run inference on an image:

```bash
python inference_pytorch.py \
    -i path/to/image.jpg \
    -s prediction.png \
    -m runs/experiment_name
```

Optional arguments:
- `--patches/--no-patches`: Enable/disable patch-based inference
- `--save-layout`: Save layout prediction only
- `--ground-truth`: Path to ground truth for IoU calculation
- `--xml-file`: Path to XML file for reading order detection
- `--min-area`: Minimum area for reading order detection

Example with all options:
```bash
python inference_pytorch.py \
    -i path/to/image.jpg \
    -s prediction.png \
    -m runs/experiment_name \
    --patches \
    --save-layout layout.png \
    --ground-truth ground_truth.png \
    --xml-file layout.xml \
    --min-area 100
```

## Configuration

The project uses JSON configuration files for model and training settings. Key configuration parameters include:

- `backbone_type`: Model backbone type
- `task`: Task type (segmentation, classification, reading_order)
- `n_classes`: Number of classes
- `input_height/width`: Input image dimensions
- `batch_size`: Training batch size
- `n_epochs`: Number of training epochs
- `patience`: Early stopping patience
- `device`: Training device (cuda/cpu)

## Model Checkpoints

The training script saves model checkpoints in the following format:
- `model-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.ckpt`: Checkpoint with validation metrics
- `model.pt`: Final model weights

## Logging

Training logs are saved in:
- TensorBoard logs: `runs/experiment_name/lightning_logs/`
- Training log file: `runs/experiment_name/train_YYYYMMDD_HHMMSS.log`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## How to cite
If you find this tool useful in your work, please consider citing our paper:

```bibtex
@inproceedings{hip23rezanezhad,
  title     = {Document Layout Analysis with Deep Learning and Heuristics},
  author    = {Rezanezhad, Vahid and Baierer, Konstantin and Gerber, Mike and Labusch, Kai and Neudecker, Clemens},
  booktitle = {Proceedings of the 7th International Workshop on Historical Document Imaging and Processing {HIP} 2023,
               San José, CA, USA, August 25-26, 2023},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  year      = {2023},
  pages     = {73--78},
  url       = {https://doi.org/10.1145/3604951.3605513}
}
```