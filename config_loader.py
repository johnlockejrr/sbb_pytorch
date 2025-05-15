import json
from pathlib import Path
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from enum import Enum

class BackboneType(str, Enum):
    TRANSFORMER = "transformer"
    RESNET = "resnet"
    NONTRANSFORMER = "nontransformer"

class TaskType(str, Enum):
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    ENHANCEMENT = "enhancement"

class UseCase(str, Enum):
    TEXTLINE = "textline"
    LAYOUT = "layout"
    PRINTSPACE = "printspace"
    GLYPH = "glyph"
    WORD = "word"

@dataclass
class ScalesConfig:
    scales: List[float]

@dataclass
class TransformerConfig:
    num_patches_xy: List[int]
    patchsize_x: int
    patchsize_y: int
    projection_dim: int
    mlp_head_units: List[int]
    layers: int
    num_heads: int
    cnn_first: bool

@dataclass
class AugmentationConfig:
    flip_aug: bool
    blur_aug: bool
    scaling: bool
    adding_rgb_background: bool
    adding_rgb_foreground: bool
    add_red_textlines: bool
    channels_shuffling: bool
    degrading: bool
    brightening: bool
    binarization: bool
    scaling_bluring: bool
    scaling_binarization: bool
    scaling_flip: bool
    rotation: bool
    rotation_not_90: bool

@dataclass
class AugmentationParams:
    blur_k: List[str]
    scales: List[float]
    brightness: List[float]
    degrade_scales: List[float]
    flip_index: List[int]
    shuffle_indexes: List[List[int]]
    thetha: List[int]
    number_of_backgrounds_per_image: int

@dataclass
class TrainingConfig:
    n_epochs: int
    n_batch: int
    learning_rate: float
    weight_decay: float
    continue_training: bool
    index_start: int
    dir_of_start_model: str
    weighted_loss: bool
    is_loss_soft_dice: bool

@dataclass
class DirectoryConfig:
    dir_train: str
    dir_eval: str
    dir_output: str
    dir_rgb_backgrounds: str
    dir_rgb_foregrounds: str
    dir_img_bin: str

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
    weighted_loss: bool = False
    dir_train: str = ""
    dir_eval: str = ""
    dir_output: str = ""
    transformer: Optional[TransformerConfig] = None

@dataclass
class Page2LabelConfig:
    use_case: UseCase
    textregions: Dict[str, int]
    imageregion: int
    separatorregion: int
    graphicregions: Dict[str, int]
    columns_width: Dict[str, int]

class ConfigLoader:
    def __init__(self, 
                 config_path: Union[str, Path], 
                 page2label_path: Optional[Union[str, Path]] = None,
                 scales_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path)
        self.page2label_path = Path(page2label_path) if page2label_path else None
        self.scales_path = Path(scales_path) if scales_path else None
        self.config: Dict = {}
        self.page2label_config: Optional[Dict] = None
        self.scales_config: Optional[Dict] = None
        
    def load_config(self) -> ModelConfig:
        """Load and validate the main configuration file."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        # Create transformer config if backbone is transformer
        transformer_config = None
        if self.config.get('backbone_type') == BackboneType.TRANSFORMER:
            transformer_config = TransformerConfig(
                num_patches_xy=self.config['transformer_num_patches_xy'],
                patchsize_x=self.config['transformer_patchsize_x'],
                patchsize_y=self.config['transformer_patchsize_y'],
                projection_dim=self.config['transformer_projection_dim'],
                mlp_head_units=self.config['transformer_mlp_head_units'],
                layers=self.config['transformer_layers'],
                num_heads=self.config['transformer_num_heads'],
                cnn_first=self.config['transformer_cnn_first']
            )
            
        return ModelConfig(
            backbone_type=BackboneType(self.config['backbone_type']),
            task=TaskType(self.config['task']),
            n_classes=self.config['n_classes'],
            input_height=self.config['input_height'],
            input_width=self.config['input_width'],
            patches=self.config['patches'],
            pretraining=self.config['pretraining'],
            augmentation=self.config['augmentation'],
            data_is_provided=self.config['data_is_provided'],
            light_model=self.config['light_model'],
            weighted_loss=self.config['weighted_loss'],
            dir_train=self.config['dir_train'],
            dir_eval=self.config['dir_eval'],
            dir_output=self.config['dir_output'],
            transformer=transformer_config
        )
    
    def load_page2label(self) -> Page2LabelConfig:
        """Load and validate the page2label configuration file."""
        if not self.page2label_path:
            raise ValueError("Page2Label config path not provided")
            
        with open(self.page2label_path, 'r') as f:
            self.page2label_config = json.load(f)
            
        return Page2LabelConfig(
            use_case=UseCase(self.page2label_config['use_case']),
            textregions=self.page2label_config['textregions'],
            imageregion=self.page2label_config['imageregion'],
            separatorregion=self.page2label_config['separatorregion'],
            graphicregions=self.page2label_config['graphicregions'],
            columns_width=self.page2label_config['columns_width']
        )
    
    def load_scales(self) -> ScalesConfig:
        """Load and validate the scales configuration file."""
        if not self.scales_path:
            raise ValueError("Scales config path not provided")
            
        with open(self.scales_path, 'r') as f:
            self.scales_config = json.load(f)
            
        return ScalesConfig(
            scales=self.scales_config['scales']
        )
    
    def get_augmentation_config(self) -> AugmentationConfig:
        """Get augmentation configuration."""
        return AugmentationConfig(
            flip_aug=self.config['flip_aug'],
            blur_aug=self.config['blur_aug'],
            scaling=self.config['scaling'],
            adding_rgb_background=self.config['adding_rgb_background'],
            adding_rgb_foreground=self.config['adding_rgb_foreground'],
            add_red_textlines=self.config['add_red_textlines'],
            channels_shuffling=self.config['channels_shuffling'],
            degrading=self.config['degrading'],
            brightening=self.config['brightening'],
            binarization=self.config['binarization'],
            scaling_bluring=self.config['scaling_bluring'],
            scaling_binarization=self.config['scaling_binarization'],
            scaling_flip=self.config['scaling_flip'],
            rotation=self.config['rotation'],
            rotation_not_90=self.config['rotation_not_90']
        )
    
    def get_augmentation_params(self) -> AugmentationParams:
        """Get augmentation parameters."""
        return AugmentationParams(
            blur_k=self.config['blur_k'],
            scales=self.config['scales'],
            brightness=self.config['brightness'],
            degrade_scales=self.config['degrade_scales'],
            flip_index=self.config['flip_index'],
            shuffle_indexes=self.config['shuffle_indexes'],
            thetha=self.config['thetha'],
            number_of_backgrounds_per_image=self.config['number_of_backgrounds_per_image']
        )
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        return TrainingConfig(
            n_epochs=self.config['n_epochs'],
            n_batch=self.config['n_batch'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            continue_training=self.config['continue_training'],
            index_start=self.config['index_start'],
            dir_of_start_model=self.config['dir_of_start_model'],
            weighted_loss=self.config['weighted_loss'],
            is_loss_soft_dice=self.config['is_loss_soft_dice']
        )
    
    def get_directory_config(self) -> DirectoryConfig:
        """Get directory configuration."""
        return DirectoryConfig(
            dir_train=self.config['dir_train'],
            dir_eval=self.config['dir_eval'],
            dir_output=self.config['dir_output'],
            dir_rgb_backgrounds=self.config['dir_rgb_backgrounds'],
            dir_rgb_foregrounds=self.config['dir_rgb_foregrounds'],
            dir_img_bin=self.config['dir_img_bin']
        )

    def validate_configs(self):
        """Validate all loaded configurations."""
        # Validate model config
        model_config = self.load_config()
        if model_config.task == TaskType.ENHANCEMENT and model_config.n_classes != 3:
            raise ValueError("For enhancement task, n_classes must be 3")
            
        if model_config.task == TaskType.SEGMENTATION and model_config.patches:
            if not model_config.transformer:
                raise ValueError("Transformer config required for segmentation with patches")
                
        # Validate page2label config if present
        if self.page2label_path:
            page2label_config = self.load_page2label()
            if page2label_config.use_case == UseCase.TEXTLINE:
                if len(page2label_config.textregions) != 1:
                    raise ValueError("For textline use case, only one text region should be defined")
                    
        # Validate scales config if present
        if self.scales_path:
            scales_config = self.load_scales()
            if not all(0 < scale < 1 for scale in scales_config.scales):
                raise ValueError("All scales must be between 0 and 1")

# Example usage:
if __name__ == "__main__":
    config_loader = ConfigLoader(
        config_path="config_params.json",
        page2label_path="custom_config_page2label.json",
        scales_path="scales_enhancement.json"
    )
    
    # Validate all configs
    config_loader.validate_configs()
    
    # Load main config
    model_config = config_loader.load_config()
    print(f"Model config: {model_config}")
    
    # Load page2label config
    page2label_config = config_loader.load_page2label()
    print(f"Page2Label config: {page2label_config}")
    
    # Load scales config
    scales_config = config_loader.load_scales()
    print(f"Scales config: {scales_config}")
    
    # Get other configs
    aug_config = config_loader.get_augmentation_config()
    aug_params = config_loader.get_augmentation_params()
    train_config = config_loader.get_training_config()
    dir_config = config_loader.get_directory_config() 