"""
Utility functions for the person re-identification project.
"""

import torch
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: str, 
                   filename: str = 'checkpoint.pth', is_best: bool = False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(filepath, best_filepath)
        print(f'Best model saved to {best_filepath}')


def load_checkpoint(checkpoint_path: str):
    """Load model checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'No checkpoint found at {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f'Checkpoint loaded from {checkpoint_path}')
    return checkpoint


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f'Training curves saved to {save_path}')
    else:
        plt.show()


def create_lr_scheduler(optimizer, scheduler_type='step', **kwargs):
    """Create learning rate scheduler."""
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 50)
        eta_min = kwargs.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'plateau':
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def normalize_features(features, axis=1):
    """Normalize features along specified axis."""
    norm = np.linalg.norm(features, axis=axis, keepdims=True)
    norm = np.maximum(norm, 1e-8)  # Avoid division by zero
    return features / norm


def compute_accuracy(predictions, targets, topk=(1,)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = predictions.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    
    return results


def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model summary including parameter count."""
    total_params = count_parameters(model)
    print(f'Model: {model.__class__.__name__}')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Input size: {input_size}')
    
    # Try to compute model size
    dummy_input = torch.randn(1, *input_size)
    try:
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                output_shape = [o.shape for o in output]
            else:
                output_shape = output.shape
            print(f'Output shape: {output_shape}')
    except Exception as e:
        print(f'Could not compute output shape: {e}')


def create_config_dict(**kwargs):
    """Create a configuration dictionary with default values."""
    default_config = {
        'model_name': 'resnet50',
        'num_classes': 1000,
        'feature_dim': 2048,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'num_epochs': 100,
        'scheduler_type': 'step',
        'step_size': 30,
        'gamma': 0.1,
        'triplet_weight': 1.0,
        'ce_weight': 1.0,
        'margin': 0.3,
        'input_size': (224, 224),
        'seed': 42
    }
    
    # Update with provided arguments
    default_config.update(kwargs)
    return default_config
