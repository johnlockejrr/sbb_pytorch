# SBB PyTorch Technical Changelog

## API Changes and Code Migrations

### 1. Import Changes

#### TensorFlow → PyTorch Imports
```python
# OLD (TensorFlow)
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import load_model

# NEW (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
```

### 2. Model Architecture Changes

#### ResNet50 Implementation

**OLD (Custom TensorFlow Implementation):**
```python
def resnet50_unet(n_classes, input_height=224, input_width=224, weight_decay=1e-6, pretraining=False):
    # Manual ResNet50 implementation
    img_input = Input(shape=(input_height, input_width, 3))
    
    # Custom conv_block and identity_block functions
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # ... manual layer definitions
    
    # Custom UNet decoder
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(v512_2048)
    o = (concatenate([o, v512_1024], axis=MERGE_AXIS))
    # ... manual decoder implementation
```

**NEW (PyTorch with torchvision):**
```python
class ResNet50UNet(nn.Module):
    def __init__(self, n_classes: int, input_height: int = 224, input_width: int = 224, 
                 task: TaskType = TaskType.SEGMENTATION, weight_decay: float = 1e-6):
        super().__init__()
        # Use pretrained ResNet50 from torchvision
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Encoder using torchvision layers
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool),
            nn.Sequential(resnet50.layer1),
            nn.Sequential(resnet50.layer2),
            nn.Sequential(resnet50.layer3),
            nn.Sequential(resnet50.layer4)
        ])
        
        # Modern PyTorch decoder
        self.decoder = nn.ModuleList([...])
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            for _ in range(5)
        ])
```

#### Key Architectural Differences:
- **Weight Loading**: `model.load_weights()` → `model.load_state_dict()`
- **Layer Definition**: Manual Keras layers → `nn.Module` classes
- **Forward Pass**: Functional API → `forward()` method
- **Device Handling**: Automatic → Explicit `.to(device)`

### 3. Training Loop Changes

#### OLD (TensorFlow with Sacred):
```python
@ex.automain
def run(n_classes, n_epochs, input_height, input_width, ...):
    # Manual configuration
    model = resnet50_unet(n_classes, input_height, input_width, weight_decay, pretraining)
    
    # Manual compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    
    # Manual data generation
    train_gen = data_gen(dir_flow_train_imgs, dir_flow_train_labels, batch_size=n_batch, ...)
    val_gen = data_gen(dir_flow_eval_imgs, dir_flow_eval_labels, batch_size=n_batch, ...)
    
    # Manual training loop
    for i in tqdm(range(index_start, n_epochs + index_start)):
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(os.listdir(dir_flow_train_imgs)) / n_batch) - 1,
            validation_data=val_gen,
            validation_steps=1,
            epochs=1)
        model.save(dir_output + '/' + 'model_' + str(i))
```

#### NEW (PyTorch Lightning):
```python
class SegmentationModel(pl.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, config: Dict):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        
        # Initialize metrics
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.train_iou = IoU(num_classes=config['n_classes'])
        self.val_iou = IoU(num_classes=config['n_classes'])
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        predictions = outputs.argmax(dim=1)
        self.train_accuracy.update(predictions, masks)
        self.train_iou.update(predictions, masks)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy.compute(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config['learning_rate'])

# Training with PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=epochs,
    accelerator=device,
    callbacks=[
        ModelCheckpoint(dirpath=output_dir, filename='model-{epoch:02d}-{val_loss:.2f}'),
        EarlyStopping(monitor='val_loss', patience=patience)
    ],
    logger=TensorBoardLogger(output_dir, name='lightning_logs')
)
trainer.fit(lightning_model, train_loader, val_loader)
```

### 4. Data Handling Changes

#### OLD (Manual Data Generation):
```python
def data_gen(image_folder, mask_folder, batch_size, input_height, input_width, n_classes):
    # Manual data loading and preprocessing
    while True:
        for i in range(0, len(images), batch_size):
            batch_images = []
            batch_masks = []
            for j in range(i, min(i + batch_size, len(images))):
                # Manual image loading and preprocessing
                img = cv2.imread(image_folder + '/' + images[j])
                img = cv2.resize(img, (input_width, input_height))
                # ... manual preprocessing
```

#### NEW (PyTorch Dataset):
```python
class SegmentationDataset(Dataset):
    def __init__(self, data_dir: Union[str, Path], transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (448, 448), num_classes: int = 2):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'labels'
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.image_files = sorted(
            list(self.image_dir.glob('*.jpg')) + 
            list(self.image_dir.glob('*.png'))
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()

# DataLoader usage
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)
```

### 5. Loss Function Changes

#### OLD (TensorFlow/Keras):
```python
def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    return 1.00 - K.mean(numerator / (denominator + epsilon))

def weighted_categorical_crossentropy(weights=None):
    def loss(y_true, y_pred):
        labels_floats = tf.cast(y_true, tf.float32)
        per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_floats, logits=y_pred)
        # ... weight handling
        return tf.reduce_mean(per_pixel_loss)
    return loss
```

#### NEW (PyTorch):
```python
class DiceLoss(nn.Module):
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

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pred, target, weight=self.weights)
```

### 6. Configuration System Changes

#### OLD (Single JSON with Sacred):
```python
@ex.config
def config_params():
    n_classes = None
    n_epochs = 1
    input_height = 224 * 1
    input_width = 224 * 1
    # ... basic parameters
```

#### NEW (Structured Configuration):
```python
@dataclass
class ModelConfig:
    backbone_type: BackboneType
    task: TaskType
    n_classes: int
    input_height: int
    input_width: int
    patches: bool
    pretraining: bool
    augmentation: bool
    data_is_provided: bool
    light_model: bool = False
    transformer: Optional[TransformerConfig] = None

class ConfigLoader:
    def load_config(self) -> ModelConfig:
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        transformer_config = None
        if self.config.get('backbone_type') == BackboneType.TRANSFORMER:
            transformer_config = TransformerConfig(
                num_patches_xy=self.config['transformer_num_patches_xy'],
                patchsize_x=self.config['transformer_patchsize_x'],
                # ... structured config loading
            )
        
        return ModelConfig(
            backbone_type=BackboneType(self.config['backbone_type']),
            task=TaskType(self.config['task']),
            # ... type-safe config creation
        )
```

### 7. Metrics Implementation Changes

#### OLD (TensorFlow/Keras):
```python
def seg_metrics(y_true, y_pred, metric_name, metric_type='standard', ...):
    # Manual metric calculation with Keras backend
    intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    union = mask_sum - intersection
    # ... complex metric calculation
```

#### NEW (PyTorch with Class-based Metrics):
```python
class IoU(Metric):
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
```

### 8. Inference Pipeline Changes

#### OLD (Basic Inference):
```python
# Manual model loading and prediction
model = load_model(model_path, compile=True)
prediction = model.predict(image)
```

#### NEW (Structured Inference):
```python
class SBBPredict:
    def __init__(self, image: str, model_dir: str, task: str, config_params: Dict, ...):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self) -> nn.Module:
        if self.task == "classification":
            model = ResNet50Classifier(...)
        elif self.task == "reading_order":
            model = MachineBasedReadingOrderModel(...)
        else:
            model = ResNet50UNet(...)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model
    
    def predict(self):
        with torch.no_grad():
            # Batch processing and prediction
            # Advanced visualization and output handling
```

### 9. New Features - Technical Implementation

#### Vision Transformer Integration:
```python
class ViTResNet50UNet(nn.Module):
    def __init__(self, n_classes: int, patch_size_x: int, patch_size_y: int, ...):
        # Patches and Transformer
        self.patches = Patches(patch_size_x, patch_size_y)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads)
            for _ in range(transformer_layers)
        ])
        
        # ResNet50 encoder
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.ModuleList([...])
        
        # UNet decoder
        self.decoder = nn.ModuleList([...])
```

#### Advanced Augmentation:
```python
# Albumentations integration
self.train_transform = A.Compose([
    A.Resize(image_size[0], image_size[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### 10. Performance Optimizations

#### Memory Management:
```python
# OLD: Manual memory management
configuration()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# NEW: PyTorch automatic memory management
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
```

#### Data Loading:
```python
# OLD: Manual data generation
train_gen = data_gen(dir_flow_train_imgs, dir_flow_train_labels, ...)

# NEW: Efficient DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True  # Faster GPU transfer
)
```

### 11. Error Handling and Logging

#### OLD (Basic Error Handling):
```python
try:
    label_obj = cv2.imread(dir_flow_train_labels + '/' + obj)
    # ... processing
except:
    pass  # Silent failure
```

#### NEW (Comprehensive Error Handling):
```python
def setup_logging(output_dir: Union[str, Path]) -> logging.Logger:
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
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
```

### 12. Type Safety and Modern Python

#### Type Hints:
```python
# OLD: No type hints
def resnet50_unet(n_classes, input_height=224, input_width=224, weight_decay=1e-6, pretraining=False):

# NEW: Comprehensive type hints
def create_model(config: 'ModelConfig') -> nn.Module:
def get_loss_function(task: str, num_classes: int, weighted: bool = False, 
                     weights: Optional[torch.Tensor] = None) -> nn.Module:
```

#### Dataclasses and Enums:
```python
@dataclass
class ModelConfig:
    backbone_type: BackboneType
    task: TaskType
    n_classes: int
    # ... structured configuration

class BackboneType(str, Enum):
    TRANSFORMER = "transformer"
    RESNET = "resnet"
    NONTRANSFORMER = "nontransformer"
```

## Breaking Changes Summary

1. **Framework**: Complete migration from TensorFlow to PyTorch
2. **API**: Function-based → Class-based architecture
3. **Data Flow**: Manual generators → PyTorch DataLoader
4. **Configuration**: Single JSON → Multi-file structured config
5. **Training**: Manual loops → PyTorch Lightning
6. **Model Loading**: `.h5` files → `.pt` checkpoints
7. **Device Handling**: Automatic → Explicit device management
8. **Error Handling**: Silent failures → Comprehensive logging

## Migration Checklist

- [ ] Update imports from TensorFlow to PyTorch
- [ ] Convert model definitions to `nn.Module` classes
- [ ] Replace Keras layers with PyTorch equivalents
- [ ] Update data loading to use `Dataset` and `DataLoader`
- [ ] Convert loss functions to `nn.Module` classes
- [ ] Update training loop to use PyTorch Lightning
- [ ] Migrate configuration to structured format
- [ ] Update model saving/loading to PyTorch format
- [ ] Add proper error handling and logging
- [ ] Update inference pipeline for new model format
