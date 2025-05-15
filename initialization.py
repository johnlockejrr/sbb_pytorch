import torch
import torch.nn as nn
import math
from typing import Optional, Union, Callable

def kaiming_init(module: nn.Module,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0.0) -> None:
    """
    Initialize weights using Kaiming initialization.
    
    Args:
        module: PyTorch module to initialize
        mode: 'fan_in' or 'fan_out'
        nonlinearity: Nonlinearity function name
        bias: Bias initialization value
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def xavier_init(module: nn.Module,
                gain: float = 1.0,
                bias: float = 0.0) -> None:
    """
    Initialize weights using Xavier/Glorot initialization.
    
    Args:
        module: PyTorch module to initialize
        gain: Scaling factor
        bias: Bias initialization value
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def orthogonal_init(module: nn.Module,
                   gain: float = 1.0,
                   bias: float = 0.0) -> None:
    """
    Initialize weights using orthogonal initialization.
    
    Args:
        module: PyTorch module to initialize
        gain: Scaling factor
        bias: Bias initialization value
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def initialize_weights(model: nn.Module,
                      init_fn: Callable = kaiming_init,
                      **kwargs) -> None:
    """
    Initialize weights for all modules in a model.
    
    Args:
        model: PyTorch model to initialize
        init_fn: Initialization function to use
        **kwargs: Additional arguments for the initialization function
    """
    for module in model.modules():
        init_fn(module, **kwargs)

class WeightRegularizer:
    """Class for applying weight regularization techniques."""
    
    @staticmethod
    def l2_regularization(model: nn.Module,
                         weight_decay: float = 1e-4) -> torch.Tensor:
        """
        Calculate L2 regularization loss.
        
        Args:
            model: PyTorch model
            weight_decay: Weight decay factor
            
        Returns:
            L2 regularization loss
        """
        l2_loss = torch.tensor(0., device=next(model.parameters()).device)
        for param in model.parameters():
            l2_loss += torch.norm(param, p=2)
        return weight_decay * l2_loss
    
    @staticmethod
    def l1_regularization(model: nn.Module,
                         weight_decay: float = 1e-4) -> torch.Tensor:
        """
        Calculate L1 regularization loss.
        
        Args:
            model: PyTorch model
            weight_decay: Weight decay factor
            
        Returns:
            L1 regularization loss
        """
        l1_loss = torch.tensor(0., device=next(model.parameters()).device)
        for param in model.parameters():
            l1_loss += torch.norm(param, p=1)
        return weight_decay * l1_loss
    
    @staticmethod
    def elastic_net_regularization(model: nn.Module,
                                 l1_ratio: float = 0.5,
                                 weight_decay: float = 1e-4) -> torch.Tensor:
        """
        Calculate elastic net regularization loss.
        
        Args:
            model: PyTorch model
            l1_ratio: Ratio of L1 to L2 regularization
            weight_decay: Weight decay factor
            
        Returns:
            Elastic net regularization loss
        """
        l1_loss = WeightRegularizer.l1_regularization(model, weight_decay)
        l2_loss = WeightRegularizer.l2_regularization(model, weight_decay)
        return l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

class GradientClipper:
    """Class for gradient clipping techniques."""
    
    @staticmethod
    def clip_grad_norm(model: nn.Module,
                      max_norm: float = 1.0,
                      norm_type: float = 2.0) -> float:
        """
        Clip gradients by norm.
        
        Args:
            model: PyTorch model
            max_norm: Maximum norm value
            norm_type: Type of norm to use
            
        Returns:
            Total norm of all parameters
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type
        )
    
    @staticmethod
    def clip_grad_value(model: nn.Module,
                       clip_value: float = 1.0) -> None:
        """
        Clip gradients by value.
        
        Args:
            model: PyTorch model
            clip_value: Maximum absolute value
        """
        torch.nn.utils.clip_grad_value_(
            model.parameters(),
            clip_value=clip_value
        )

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
    
    # Initialize weights
    initialize_weights(model, init_fn=kaiming_init)
    
    # Test regularization
    l2_loss = WeightRegularizer.l2_regularization(model)
    l1_loss = WeightRegularizer.l1_regularization(model)
    elastic_net_loss = WeightRegularizer.elastic_net_regularization(model)
    
    print(f"L2 Loss: {l2_loss.item():.4f}")
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"Elastic Net Loss: {elastic_net_loss.item():.4f}")
    
    # Test gradient clipping
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    
    # Clip gradients
    total_norm = GradientClipper.clip_grad_norm(model)
    print(f"Total gradient norm: {total_norm.item():.4f}") 