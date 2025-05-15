import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import os

def save_model(model: nn.Module, 
               save_dir: Union[str, Path], 
               model_name: str,
               config: Optional[Dict[str, Any]] = None,
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None,
               loss: Optional[float] = None) -> None:
    """
    Save model state, configuration, and training state.
    
    Args:
        model: PyTorch model to save
        save_dir: Directory to save the model
        model_name: Name of the model file
        config: Model configuration dictionary
        optimizer: Optimizer state
        epoch: Current training epoch
        loss: Current loss value
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare state dictionary
    state = {
        'model_state': model.state_dict(),
        'config': config,
        'epoch': epoch,
        'loss': loss
    }
    
    if optimizer is not None:
        state['optimizer_state'] = optimizer.state_dict()
    
    # Save model state
    model_path = save_dir / f"{model_name}.pth"
    torch.save(state, model_path)
    
    # Save configuration separately for easy access
    if config is not None:
        config_path = save_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

def load_model(model: nn.Module,
               load_path: Union[str, Path],
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model state and training state.
    
    Args:
        model: PyTorch model to load state into
        load_path: Path to the saved model file
        optimizer: Optimizer to load state into
        device: Device to load the model onto
        
    Returns:
        Dictionary containing the loaded state information
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"No model found at {load_path}")
    
    # Load state
    state = torch.load(load_path, map_location=device)
    
    # Load model state
    model.load_state_dict(state['model_state'])
    if device is not None:
        model = model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state' in state:
        optimizer.load_state_dict(state['optimizer_state'])
    
    return {
        'config': state.get('config'),
        'epoch': state.get('epoch'),
        'loss': state.get('loss')
    }

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load model configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"No configuration found at {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint if found, None otherwise
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None
    
    return max(checkpoints, key=os.path.getctime)

def save_checkpoint(model: nn.Module,
                   save_dir: Union[str, Path],
                   model_name: str,
                   epoch: int,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   loss: Optional[float] = None,
                   config: Optional[Dict[str, Any]] = None,
                   keep_last_n: int = 5) -> None:
    """
    Save a checkpoint and manage old checkpoints.
    
    Args:
        model: PyTorch model to save
        save_dir: Directory to save checkpoints
        model_name: Base name for the checkpoint
        epoch: Current epoch number
        optimizer: Optimizer state
        loss: Current loss value
        config: Model configuration
        keep_last_n: Number of most recent checkpoints to keep
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save new checkpoint
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    save_model(model, save_dir, checkpoint_name, config, optimizer, epoch, loss)
    
    # Remove old checkpoints
    checkpoints = sorted(save_dir.glob(f"{model_name}_epoch_*.pth"))
    if len(checkpoints) > keep_last_n:
        for checkpoint in checkpoints[:-keep_last_n]:
            checkpoint.unlink()

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
    
    # Save model
    save_model(
        model=model,
        save_dir="checkpoints",
        model_name="test_model",
        config=model_config.__dict__,
        epoch=0,
        loss=0.0
    )
    
    # Load model
    loaded_state = load_model(
        model=model,
        load_path="checkpoints/test_model.pth"
    )
    
    print(f"Loaded model from epoch {loaded_state['epoch']} with loss {loaded_state['loss']}") 