# Layer Normalization Exercises

This document contains practical exercises to help you understand and implement Layer Normalization from scratch. These exercises range from basic to advanced and are designed to reinforce your understanding of the concepts.

## Exercise 1: Basic Implementation

Implement a basic Layer Normalization function in NumPy without using any built-in normalization functions.

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Implement layer normalization from scratch.
    
    Args:
        x: Input array of shape (batch_size, features)
        gamma: Scale parameter of shape (features,)
        beta: Shift parameter of shape (features,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized array of same shape as x
    """
    # TODO: Implement layer normalization
    # 1. Calculate mean along the feature dimension
    # 2. Calculate variance along the feature dimension
    # 3. Normalize the input
    # 4. Scale and shift with gamma and beta
    
    # Your code here
    
    return normalized_x
```

## Exercise 2: Forward and Backward Pass

Implement both the forward and backward pass for Layer Normalization in NumPy.

```python
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.eps = eps
        
        # For storing intermediate values
        self.input = None
        self.normalized = None
        self.mean = None
        self.var = None
        
    def forward(self, x):
        """
        Forward pass of Layer Normalization
        
        Args:
            x: Input array of shape (batch_size, features)
            
        Returns:
            Normalized array of same shape as x
        """
        # TODO: Implement forward pass
        # Your code here
        
        return output
    
    def backward(self, dout):
        """
        Backward pass of Layer Normalization
        
        Args:
            dout: Gradient of loss with respect to output
            
        Returns:
            dx: Gradient of loss with respect to input
            dgamma: Gradient of loss with respect to gamma
            dbeta: Gradient of loss with respect to beta
        """
        # TODO: Implement backward pass
        # Your code here
        
        return dx, dgamma, dbeta
```

## Exercise 3: PyTorch Implementation

Implement Layer Normalization as a PyTorch module.

```python
import torch
import torch.nn as nn

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        
        # TODO: Initialize parameters
        # Your code here
        
    def forward(self, x):
        # TODO: Implement forward pass
        # Your code here
        
        return output
```

## Exercise 4: Comparing Normalization Techniques

Compare the performance of Layer Normalization with Batch Normalization on a simple classification task.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# TODO: Implement two models, one with BatchNorm and one with LayerNorm
# Compare their training curves and final performance

class ModelWithBatchNorm(nn.Module):
    # Your implementation here
    pass

class ModelWithLayerNorm(nn.Module):
    # Your implementation here
    pass

# TODO: Train both models and compare results
```

## Exercise 5: Implementing RMSNorm

Implement Root Mean Square Layer Normalization (RMSNorm) as described in the paper.

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(RMSNorm, self).__init__()
        
        # TODO: Initialize parameters
        # Your code here
        
    def forward(self, x):
        # TODO: Implement RMSNorm forward pass
        # Your code here
        
        return output
```

## Exercise 6: Conditional Layer Normalization

Implement Conditional Layer Normalization where the normalization parameters depend on an external input.

```python
import torch
import torch.nn as nn

class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, condition_size, eps=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        
        # TODO: Initialize parameters
        # Your code here
        
    def forward(self, x, condition):
        # TODO: Implement conditional layer normalization
        # Your code here
        
        return output
```

## Exercise 7: Layer Normalization in a Transformer

Implement a simplified Transformer encoder block with Layer Normalization.

```python
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # TODO: Implement a transformer encoder block with:
        # 1. Multi-head self-attention
        # 2. Layer normalization
        # 3. Feed-forward network
        # 4. Residual connections
        
        # Your code here
        
    def forward(self, src, src_mask=None):
        # TODO: Implement the forward pass
        # Your code here
        
        return output
```

## Exercise 8: Visualizing the Effect of Layer Normalization

Create a visualization that shows how Layer Normalization affects the distribution of activations in a neural network.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# TODO: Create a simple neural network with and without layer normalization
# Visualize the distribution of activations at different layers

def visualize_activations():
    # Your code here
    pass
```

## Exercise 9: Pre-LayerNorm vs Post-LayerNorm

Implement both Pre-LayerNorm and Post-LayerNorm variants of a Transformer encoder and compare their training stability.

```python
import torch
import torch.nn as nn

class PreLayerNormTransformer(nn.Module):
    # TODO: Implement Pre-LayerNorm Transformer
    pass

class PostLayerNormTransformer(nn.Module):
    # TODO: Implement Post-LayerNorm Transformer
    pass

# TODO: Compare training stability
```

## Exercise 10: Layer Normalization for Different Data Types

Adapt Layer Normalization to work with different types of data: images, text, and time series.

```python
import torch
import torch.nn as nn

class ImageLayerNorm(nn.Module):
    # TODO: Implement Layer Normalization for image data (2D)
    pass

class TextLayerNorm(nn.Module):
    # TODO: Implement Layer Normalization for text data (sequence)
    pass

class TimeSeriesLayerNorm(nn.Module):
    # TODO: Implement Layer Normalization for time series data
    pass
```

## Solutions

The solutions to these exercises can be found in the `layer_normalization_solutions.py` file.

## Further Challenges

1. Implement a variant of Layer Normalization that normalizes across multiple dimensions
2. Create a custom Layer Normalization that adapts its parameters based on the input distribution
3. Implement Group Normalization and compare it with Layer Normalization
4. Explore the effect of different initialization strategies for the gamma and beta parameters
5. Implement a version of Layer Normalization that works efficiently with sparse inputs
