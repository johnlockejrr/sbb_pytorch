import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional
from config_loader import TaskType, BackboneType

class Patches(nn.Module):
    """PyTorch implementation of the Patches layer."""
    def __init__(self, patch_size_x: int, patch_size_y: int):
        super().__init__()
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size_y, self.patch_size_y).unfold(3, self.patch_size_x, self.patch_size_x)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * self.patch_size_x * self.patch_size_y)
        return patches

class PatchEncoder(nn.Module):
    """PyTorch implementation of the PatchEncoder layer."""
    def __init__(self, num_patches: int, projection_dim: int):
        super().__init__()
        self.projection = nn.Linear(projection_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(0, self.position_embedding.num_embeddings, device=x.device)
        encoded = self.projection(x) + self.position_embedding(positions)
        return encoded

class TransformerBlock(nn.Module):
    """PyTorch implementation of the TransformerBlock."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ResNet50UNet(nn.Module):
    """PyTorch implementation of ResNet50-UNet."""
    def __init__(self, n_classes: int, input_height: int = 224, input_width: int = 224, 
                 task: TaskType = TaskType.SEGMENTATION, weight_decay: float = 1e-6):
        super().__init__()
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool),  # f1: 64
            nn.Sequential(resnet50.layer1),  # f2: 256
            nn.Sequential(resnet50.layer2),  # f3: 512
            nn.Sequential(resnet50.layer3),  # f4: 1024
            nn.Sequential(resnet50.layer4)   # f5: 2048
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            # First decoder block (no skip connection)
            nn.Sequential(
                nn.Conv2d(2048, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ),
            # Second decoder block (skip from f4: 1024)
            nn.Sequential(
                nn.Conv2d(1536, 512, 3, padding=1),  # 1024 from upsampled + 512 from skip
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # Third decoder block (skip from f3: 512)
            nn.Sequential(
                nn.Conv2d(768, 256, 3, padding=1),  # 512 from upsampled + 256 from skip
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # Fourth decoder block (skip from f2: 256)
            nn.Sequential(
                nn.Conv2d(320, 128, 3, padding=1),  # 256 from upsampled + 64 from skip
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # Fifth decoder block (skip from f1: 64)
            nn.Sequential(
                nn.Conv2d(192, 64, 3, padding=1),  # 128 from upsampled + 64 from skip
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            for _ in range(5)
        ])
        
        # Final layers
        self.final_conv = nn.Conv2d(64, n_classes, 1)
        self.task = task
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
            
        # Decoder
        x = features[-1]  # Start with the last encoder output (2048 channels)
        for i, (decoder_layer, upsample) in enumerate(zip(self.decoder, self.upsample)):
            # First upsample
            x = upsample(x)
            
            # Then concatenate with skip connection if available
            if i < len(features) - 1:
                skip_feature = features[-(i+2)]
                # Ensure skip feature has the same spatial dimensions
                if skip_feature.shape[2:] != x.shape[2:]:
                    skip_feature = F.interpolate(skip_feature, size=x.shape[2:], mode='bilinear', align_corners=True)
                
                # For the first decoder block, we don't concatenate
                if i == 0:
                    x = decoder_layer(x)
                else:
                    x = torch.cat([x, skip_feature], dim=1)
                    x = decoder_layer(x)
            else:
                # For the last block, we need to concatenate with the input
                # Upsample the first encoder output to match spatial dimensions
                first_encoder_output = features[0]
                if first_encoder_output.shape[2:] != x.shape[2:]:
                    first_encoder_output = F.interpolate(
                        first_encoder_output, 
                        size=x.shape[2:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                x = torch.cat([x, first_encoder_output], dim=1)
                x = decoder_layer(x)
            
        # Final layer - output raw logits
        x = self.final_conv(x)
        
        # Only apply activation for non-segmentation tasks
        if self.task != TaskType.SEGMENTATION:
            x = torch.sigmoid(x)
            
        return x

class ResNet50UNetLight(nn.Module):
    """PyTorch implementation of a lighter version of ResNet50-UNet."""
    def __init__(self, n_classes: int, input_height: int = 224, input_width: int = 224, 
                 task: TaskType = TaskType.SEGMENTATION, weight_decay: float = 1e-6):
        super().__init__()
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool),  # f1
            nn.Sequential(resnet50.layer1),  # f2
            nn.Sequential(resnet50.layer2),  # f3
            nn.Sequential(resnet50.layer3),  # f4
            nn.Sequential(resnet50.layer4)   # f5
        ])
        
        # Decoder with reduced channels
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, 1),  # Reduced from 1024
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),  # Reduced from 512
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),  # Reduced from 256
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),  # Reduced from 128
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),  # Reduced from 64
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            for _ in range(5)
        ])
        
        # Final layers
        self.final_conv = nn.Conv2d(32, n_classes, 1)
        self.task = task
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
            
        # Decoder
        for i, (decoder_layer, upsample) in enumerate(zip(self.decoder, self.upsample)):
            x = upsample(x)
            if i < len(features) - 1:
                x = torch.cat([x, features[-(i+2)]], dim=1)
            x = decoder_layer(x)
            
        # Final layer
        x = self.final_conv(x)
        
        if self.task == TaskType.SEGMENTATION:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
            
        return x

class ResNet50Classifier(nn.Module):
    """PyTorch implementation of ResNet50 classifier."""
    def __init__(self, n_classes: int, input_height: int = 224, input_width: int = 224, 
                 weight_decay: float = 1e-6):
        super().__init__()
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the classification head
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MachineBasedReadingOrderModel(nn.Module):
    """PyTorch implementation of machine-based reading order model."""
    def __init__(self, n_classes: int, input_height: int = 224, input_width: int = 224, 
                 weight_decay: float = 1e-6):
        super().__init__()
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the classification head
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ViTResNet50UNet(nn.Module):
    """PyTorch implementation of ViT-ResNet50-UNet."""
    def __init__(self, n_classes: int, patch_size_x: int, patch_size_y: int, num_patches: int,
                 mlp_head_units: List[int] = [128, 64], transformer_layers: int = 8,
                 num_heads: int = 4, projection_dim: int = 64, input_height: int = 224,
                 input_width: int = 224, task: TaskType = TaskType.SEGMENTATION, weight_decay: float = 1e-6):
        super().__init__()
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool),  # f1
            nn.Sequential(resnet50.layer1),  # f2
            nn.Sequential(resnet50.layer2),  # f3
            nn.Sequential(resnet50.layer3),  # f4
            nn.Sequential(resnet50.layer4)   # f5
        ])
        
        # Patches and Transformer
        self.patches = Patches(patch_size_x, patch_size_y)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads)
            for _ in range(transformer_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            for _ in range(5)
        ])
        
        # Final layers
        self.final_conv = nn.Conv2d(64, n_classes, 1)
        self.task = task
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
            
        # Transformer processing
        patches = self.patches(x)
        encoded_patches = self.patch_encoder(patches)
        for transformer_block in self.transformer_blocks:
            encoded_patches = transformer_block(encoded_patches)
            
        # Reshape transformer output back to spatial dimensions
        B, C, H, W = x.shape
        x = encoded_patches.view(B, H//self.patches.patch_size_y, W//self.patches.patch_size_x, -1)
        x = x.permute(0, 3, 1, 2)
            
        # Decoder
        for i, (decoder_layer, upsample) in enumerate(zip(self.decoder, self.upsample)):
            x = upsample(x)
            if i < len(features) - 1:
                x = torch.cat([x, features[-(i+2)]], dim=1)
            x = decoder_layer(x)
            
        # Final layer
        x = self.final_conv(x)
        
        if self.task == TaskType.SEGMENTATION:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
            
        return x

class ViTResNet50UNetTransformerBeforeCNN(nn.Module):
    """PyTorch implementation of ViT-ResNet50-UNet with transformer before CNN."""
    def __init__(self, n_classes: int, patch_size_x: int, patch_size_y: int, num_patches: int,
                 mlp_head_units: List[int] = [128, 64], transformer_layers: int = 8,
                 num_heads: int = 4, projection_dim: int = 64, input_height: int = 224,
                 input_width: int = 224, task: TaskType = TaskType.SEGMENTATION, weight_decay: float = 1e-6):
        super().__init__()
        # Patches and Transformer
        self.patches = Patches(patch_size_x, patch_size_y)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads)
            for _ in range(transformer_layers)
        ])
        
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool),  # f1
            nn.Sequential(resnet50.layer1),  # f2
            nn.Sequential(resnet50.layer2),  # f3
            nn.Sequential(resnet50.layer3),  # f4
            nn.Sequential(resnet50.layer4)   # f5
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            for _ in range(5)
        ])
        
        # Final layers
        self.final_conv = nn.Conv2d(64, n_classes, 1)
        self.task = task
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract and encode patches
        patches = self.patches(x)
        encoded_patches = self.patch_encoder(patches)
        
        # Transformer processing
        for transformer_block in self.transformer_blocks:
            encoded_patches = transformer_block(encoded_patches)
            
        # Reshape transformer output back to spatial dimensions
        B, C, H, W = x.shape
        x = encoded_patches.view(B, H//self.patches.patch_size_y, W//self.patches.patch_size_x, -1)
        x = x.permute(0, 3, 1, 2)
        
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
            
        # Decoder
        for i, (decoder_layer, upsample) in enumerate(zip(self.decoder, self.upsample)):
            x = upsample(x)
            if i < len(features) - 1:
                x = torch.cat([x, features[-(i+2)]], dim=1)
            x = decoder_layer(x)
            
        # Final layer
        x = self.final_conv(x)
        
        if self.task == TaskType.SEGMENTATION:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
            
        return x

class EnhancementModel(nn.Module):
    """PyTorch implementation of the enhancement model for binarization."""
    def __init__(self, input_height: int = 224, input_width: int = 224, 
                 weight_decay: float = 1e-6):
        super().__init__()
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool),  # f1
            nn.Sequential(resnet50.layer1),  # f2
            nn.Sequential(resnet50.layer2),  # f3
            nn.Sequential(resnet50.layer3),  # f4
            nn.Sequential(resnet50.layer4)   # f5
        ])
        
        # Decoder with attention
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                AttentionBlock(1024)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                AttentionBlock(512)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                AttentionBlock(256)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                AttentionBlock(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                AttentionBlock(64)
            )
        ])
        
        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            for _ in range(5)
        ])
        
        # Final layers for RGB output
        self.final_conv = nn.Conv2d(64, 3, 1)
        self.final_activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
            
        # Decoder
        for i, (decoder_layer, upsample) in enumerate(zip(self.decoder, self.upsample)):
            x = upsample(x)
            if i < len(features) - 1:
                x = torch.cat([x, features[-(i+2)]], dim=1)
            x = decoder_layer(x)
            
        # Final layer
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x

class AttentionBlock(nn.Module):
    """Attention block for the enhancement model."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Spatial attention
        spatial_weights = self.spatial_attention(torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1))
        x = x * spatial_weights
        
        return x

def create_model(config: 'ModelConfig') -> nn.Module:
    """Factory function to create the appropriate model based on configuration."""
    if config.task == TaskType.SEGMENTATION:
        if config.backbone_type == BackboneType.TRANSFORMER:
            if config.transformer.cnn_first:
                return ViTResNet50UNet(
                    n_classes=config.n_classes,
                    patch_size_x=config.transformer.patchsize_x,
                    patch_size_y=config.transformer.patchsize_y,
                    num_patches=config.transformer.num_patches_xy[0] * config.transformer.num_patches_xy[1],
                    mlp_head_units=config.transformer.mlp_head_units,
                    transformer_layers=config.transformer.layers,
                    num_heads=config.transformer.num_heads,
                    projection_dim=config.transformer.projection_dim,
                    input_height=config.input_height,
                    input_width=config.input_width,
                    task=TaskType.SEGMENTATION
                )
            else:
                return ViTResNet50UNetTransformerBeforeCNN(
                    n_classes=config.n_classes,
                    patch_size_x=config.transformer.patchsize_x,
                    patch_size_y=config.transformer.patchsize_y,
                    num_patches=config.transformer.num_patches_xy[0] * config.transformer.num_patches_xy[1],
                    mlp_head_units=config.transformer.mlp_head_units,
                    transformer_layers=config.transformer.layers,
                    num_heads=config.transformer.num_heads,
                    projection_dim=config.transformer.projection_dim,
                    input_height=config.input_height,
                    input_width=config.input_width,
                    task=TaskType.SEGMENTATION
                )
        elif config.backbone_type in [BackboneType.RESNET, BackboneType.NONTRANSFORMER]:
            if config.light_model:
                return ResNet50UNetLight(
                    n_classes=config.n_classes,
                    input_height=config.input_height,
                    input_width=config.input_width,
                    task=TaskType.SEGMENTATION
                )
            else:
                return ResNet50UNet(
                    n_classes=config.n_classes,
                    input_height=config.input_height,
                    input_width=config.input_width,
                    task=TaskType.SEGMENTATION
                )
    elif config.task == TaskType.CLASSIFICATION:
        return ResNet50Classifier(
            n_classes=config.n_classes,
            input_height=config.input_height,
            input_width=config.input_width
        )
    elif config.task == TaskType.ENHANCEMENT:
        return EnhancementModel(
            input_height=config.input_height,
            input_width=config.input_width
        )
    elif config.task == TaskType.READING_ORDER:
        return MachineBasedReadingOrderModel(
            n_classes=config.n_classes,
            input_height=config.input_height,
            input_width=config.input_width
        )
    else:
        raise ValueError(f"Unknown task type: {config.task}")

# Example usage:
if __name__ == "__main__":
    from config_loader import ConfigLoader, ModelConfig
    
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
    
    # Test forward pass
    x = torch.randn(1, 3, model_config.input_height, model_config.input_width)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 