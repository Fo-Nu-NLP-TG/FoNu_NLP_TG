"""
Layer Normalization Implementation Examples

This file contains various implementations of Layer Normalization,
from basic to advanced, with detailed annotations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================
# Basic NumPy Implementation (for understanding)
# =============================================

class LayerNormNumPy:
    """
    A simple NumPy implementation of Layer Normalization.
    This is for educational purposes to understand the core algorithm.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize Layer Normalization
        
        Args:
            normalized_shape: The shape of the input features to be normalized
            eps: A small constant for numerical stability
        """
        # Initialize learnable parameters
        self.gamma = np.ones(normalized_shape)  # Scale parameter (initialized to 1)
        self.beta = np.zeros(normalized_shape)  # Shift parameter (initialized to 0)
        self.eps = eps
        
        # For storing intermediate values (useful for backward pass)
        self.input = None
        self.normalized = None
        self.mean = None
        self.var = None
    
    def forward(self, x):
        """
        Forward pass of Layer Normalization
        
        Args:
            x: Input tensor of shape (batch_size, ..., normalized_shape)
            
        Returns:
            Normalized output
        """
        # Store input for backward pass
        self.input = x
        
        # Step 1: Calculate mean along the last dimension
        self.mean = np.mean(x, axis=-1, keepdims=True)
        
        # Step 2: Calculate variance along the last dimension
        self.var = np.var(x, axis=-1, keepdims=True)
        
        # Step 3: Normalize the input
        self.normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        
        # Step 4: Scale and shift with learnable parameters
        return self.gamma * self.normalized + self.beta
    
    def visualize_normalization(self, x):
        """
        Visualize the effect of layer normalization on the input distribution
        
        Args:
            x: Input tensor
        """
        # Forward pass to get normalized values
        normalized_x = self.forward(x)
        
        # Plot histograms of original and normalized values
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(x.flatten(), bins=50, alpha=0.7)
        plt.title('Original Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(normalized_x.flatten(), bins=50, alpha=0.7)
        plt.title('Normalized Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()


# =============================================
# PyTorch Implementation
# =============================================

class LayerNormTorch(nn.Module):
    """
    A PyTorch implementation of Layer Normalization.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize Layer Normalization
        
        Args:
            normalized_shape: The shape of the input features to be normalized
            eps: A small constant for numerical stability
        """
        super(LayerNormTorch, self).__init__()
        
        # Create learnable parameters as PyTorch Parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        """
        Forward pass of Layer Normalization
        
        Args:
            x: Input tensor of shape (batch_size, ..., normalized_shape)
            
        Returns:
            Normalized output
        """
        # Step 1: Calculate mean along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        
        # Step 2: Calculate variance along the last dimension
        # unbiased=False means we're dividing by n, not n-1
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Step 3: Normalize the input
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Step 4: Scale and shift with learnable parameters
        return self.gamma * x_normalized + self.beta


# =============================================
# Advanced Variants
# =============================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    A simplified version of Layer Normalization that only normalizes by
    the root mean square of the activations, without centering.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        # Calculate root mean square along the last dimension
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        x_normalized = x / rms
        
        # Scale with learnable parameter (no shift parameter in RMSNorm)
        return self.gamma * x_normalized


class PowerNorm(nn.Module):
    """
    Power Normalization
    
    A variant that uses higher-order moments for normalization.
    """
    
    def __init__(self, normalized_shape, p=2, eps=1e-5):
        super(PowerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.p = p  # Power (typically 2, equivalent to RMSNorm)
        self.eps = eps
    
    def forward(self, x):
        # Calculate p-th moment
        power = torch.mean(torch.abs(x)**self.p, dim=-1, keepdim=True)
        
        # Normalize by p-th root of the p-th moment
        x_normalized = x / (power**(1/self.p) + self.eps)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta


class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization
    
    Adapts the normalization parameters based on additional input.
    Useful for tasks where normalization should be conditioned on
    some external factor (e.g., style transfer, language models).
    """
    
    def __init__(self, normalized_shape, condition_size, eps=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Projection from condition to scale and shift parameters
        self.condition_projection = nn.Linear(condition_size, normalized_shape * 2)
        self.eps = eps
        
    def forward(self, x, condition):
        """
        Forward pass with conditioning
        
        Args:
            x: Input tensor to normalize
            condition: Conditioning tensor
            
        Returns:
            Normalized and conditioned output
        """
        # Project condition to gamma and beta
        scale_shift = self.condition_projection(condition)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        
        # Apply layer normalization with conditional parameters
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply conditional scale and shift
        # Note: we add 1 to gamma to make the initial state close to identity mapping
        return (1 + gamma) * x_normalized + beta


# =============================================
# Practical Examples
# =============================================

def simple_nn_example():
    """
    Example of using Layer Normalization in a simple neural network
    """
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            # First linear layer
            x = self.fc1(x)
            
            # Apply layer normalization
            x = self.ln1(x)
            
            # Activation function
            x = torch.relu(x)
            
            # Output layer
            x = self.fc2(x)
            return x

    # Create a model
    model = SimpleNN(10, 50, 2)
    
    # Sample input
    x = torch.randn(32, 10)  # Batch size of 32, input size of 10
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return model


def transformer_encoder_example():
    """
    Example of using Layer Normalization in a Transformer Encoder Layer
    """
    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
            super(TransformerEncoderLayer, self).__init__()
            
            # Multi-head attention
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            
            # Feed-forward network
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Dropout
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            
        def forward(self, src, src_mask=None, src_key_padding_mask=None):
            """
            Forward pass for the encoder layer
            
            Args:
                src: Source sequence [sequence_length, batch_size, d_model]
                src_mask: Mask for the source sequence
                src_key_padding_mask: Mask for padding in the source
                
            Returns:
                Output sequence with same shape as src
            """
            # Pre-LayerNorm architecture (more stable)
            
            # Multi-head attention block
            src2 = self.norm1(src)  # Apply normalization first
            src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)  # Residual connection
            
            # Feed-forward block
            src2 = self.norm2(src)  # Apply normalization first
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
            src = src + self.dropout2(src2)  # Residual connection
            
            return src

    # Create a model
    encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
    
    # Sample input: [sequence_length, batch_size, d_model]
    src = torch.randn(10, 32, 512)
    
    # Forward pass
    output = encoder_layer(src)
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    
    return encoder_layer


def compare_normalization_techniques():
    """
    Compare different normalization techniques
    """
    # Create sample data with different scales
    batch_size = 10
    seq_len = 20
    feature_dim = 30
    
    # Create data with different scales for different features
    x = torch.zeros(batch_size, seq_len, feature_dim)
    for i in range(feature_dim):
        scale = 10 ** (i % 3)  # Some features have values around 1, some 10, some 100
        x[:, :, i] = torch.randn(batch_size, seq_len) * scale
    
    # Apply different normalization techniques
    ln = nn.LayerNorm(feature_dim)
    rms_norm = RMSNorm(feature_dim)
    power_norm = PowerNorm(feature_dim, p=4)
    
    # Normalize
    x_ln = ln(x)
    x_rms = rms_norm(x)
    x_power = power_norm(x)
    
    # Compute statistics
    print("Original data statistics:")
    print(f"Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
    print(f"Min: {x.min().item():.4f}, Max: {x.max().item():.4f}")
    
    print("\nLayer Norm statistics:")
    print(f"Mean: {x_ln.mean().item():.4f}, Std: {x_ln.std().item():.4f}")
    print(f"Min: {x_ln.min().item():.4f}, Max: {x_ln.max().item():.4f}")
    
    print("\nRMS Norm statistics:")
    print(f"Mean: {x_rms.mean().item():.4f}, Std: {x_rms.std().item():.4f}")
    print(f"Min: {x_rms.min().item():.4f}, Max: {x_rms.max().item():.4f}")
    
    print("\nPower Norm statistics:")
    print(f"Mean: {x_power.mean().item():.4f}, Std: {x_power.std().item():.4f}")
    print(f"Min: {x_power.min().item():.4f}, Max: {x_power.max().item():.4f}")
    
    return x, x_ln, x_rms, x_power


# =============================================
# Visualization Functions
# =============================================

def visualize_layer_norm_effect():
    """
    Visualize the effect of layer normalization on feature distributions
    """
    # Create sample data with different scales
    batch_size = 1000
    feature_dim = 50
    
    # Create data with different scales for different features
    x = torch.zeros(batch_size, feature_dim)
    for i in range(feature_dim):
        if i < feature_dim // 3:
            # First third: small values
            x[:, i] = torch.randn(batch_size) * 0.1
        elif i < 2 * feature_dim // 3:
            # Second third: medium values
            x[:, i] = torch.randn(batch_size) * 1.0
        else:
            # Last third: large values
            x[:, i] = torch.randn(batch_size) * 10.0
    
    # Apply layer normalization
    ln = nn.LayerNorm(feature_dim)
    x_ln = ln(x)
    
    # Plot histograms
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.hist(x[:, 0].numpy(), bins=30, alpha=0.5, label='Feature 1 (small scale)')
    plt.hist(x[:, feature_dim//2].numpy(), bins=30, alpha=0.5, label='Feature 25 (medium scale)')
    plt.hist(x[:, -1].numpy(), bins=30, alpha=0.5, label='Feature 50 (large scale)')
    plt.title('Original Feature Distributions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Normalized data
    plt.subplot(1, 3, 2)
    plt.hist(x_ln[:, 0].detach().numpy(), bins=30, alpha=0.5, label='Feature 1 (normalized)')
    plt.hist(x_ln[:, feature_dim//2].detach().numpy(), bins=30, alpha=0.5, label='Feature 25 (normalized)')
    plt.hist(x_ln[:, -1].detach().numpy(), bins=30, alpha=0.5, label='Feature 50 (normalized)')
    plt.title('Layer Normalized Feature Distributions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Boxplot comparison
    plt.subplot(1, 3, 3)
    data = [
        x[:, 0].numpy(), 
        x[:, feature_dim//2].numpy(), 
        x[:, -1].numpy(),
        x_ln[:, 0].detach().numpy(), 
        x_ln[:, feature_dim//2].detach().numpy(), 
        x_ln[:, -1].detach().numpy()
    ]
    labels = [
        'Orig F1', 
        'Orig F25', 
        'Orig F50',
        'Norm F1', 
        'Norm F25', 
        'Norm F50'
    ]
    plt.boxplot(data, labels=labels)
    plt.title('Feature Distribution Comparison')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('layer_norm_visualization.png')
    plt.close()
    
    print("Visualization saved as 'layer_norm_visualization.png'")


if __name__ == "__main__":
    # Run examples
    print("Simple Neural Network Example:")
    simple_nn_example()
    
    print("\nTransformer Encoder Example:")
    transformer_encoder_example()
    
    print("\nComparing Normalization Techniques:")
    compare_normalization_techniques()
    
    print("\nVisualizing Layer Normalization Effect:")
    visualize_layer_norm_effect()
