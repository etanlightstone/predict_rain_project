import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class RainPredictionModel(nn.Module):
    """
    Configurable neural network for rain prediction using weather features.
    
    Args:
        input_dim (int): Number of input features (default: 6 for weather features)
        hidden_dims (List[int]): List of hidden layer dimensions
        dropout_rate (float): Dropout rate for regularization
        activation (str): Activation function type ('relu', 'tanh', 'leaky_relu')
        use_batch_norm (bool): Whether to use batch normalization
        output_dim (int): Number of output classes (default: 1 for binary classification)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        output_dim: int = 1
    ):
        super(RainPredictionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.output_dim = output_dim
        
        # Define activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation_fn(x)
            x = self.dropouts[i](x)
        
        # Output layer (no activation for binary classification with BCEWithLogitsLoss)
        x = self.output_layer(x)
        
        return x
    
    def get_config(self) -> dict:
        """Return model configuration as dictionary."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'output_dim': self.output_dim
        }
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> RainPredictionModel:
    """
    Factory function to create model from configuration dictionary.
    
    Args:
        config (dict): Model configuration parameters
        
    Returns:
        RainPredictionModel: Configured model instance
    """
    return RainPredictionModel(**config)


# Example configurations for different model architectures
MODEL_CONFIGS = {
    'small': {
        'input_dim': 6,
        'hidden_dims': [32],
        'dropout_rate': 0.2,
        'activation': 'relu',
        'use_batch_norm': False,
        'output_dim': 1
    },
    'medium': {
        'input_dim': 6,
        'hidden_dims': [64, 32],
        'dropout_rate': 0.3,
        'activation': 'relu',
        'use_batch_norm': True,
        'output_dim': 1
    },
    'large': {
        'input_dim': 6,
        'hidden_dims': [128, 64, 32],
        'dropout_rate': 0.4,
        'activation': 'relu',
        'use_batch_norm': True,
        'output_dim': 1
    },
    'deep': {
        'input_dim': 6,
        'hidden_dims': [64, 64, 64, 32],
        'dropout_rate': 0.3,
        'activation': 'leaky_relu',
        'use_batch_norm': True,
        'output_dim': 1
    }
}


if __name__ == "__main__":
    # Test model creation and forward pass
    for name, config in MODEL_CONFIGS.items():
        print(f"\n=== Testing {name.upper()} model ===")
        model = create_model(config)
        print(f"Model config: {model.get_config()}")
        print(f"Total parameters: {model.count_parameters():,}")
        
        # Test forward pass
        batch_size = 32
        dummy_input = torch.randn(batch_size, config['input_dim'])
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]") 