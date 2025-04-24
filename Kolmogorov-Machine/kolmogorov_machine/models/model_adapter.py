"""
Model adapters for different frameworks.

This module provides adapters for working with neural network models
from different frameworks (PyTorch, TensorFlow, etc.) in a unified way.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

class ModelAdapter:
    """
    Base class for model adapters.
    
    This class defines the interface for model adapters that provide
    a unified way to work with neural network models from different frameworks.
    """
    
    def __init__(self):
        """Initialize the model adapter."""
        pass
    
    def get_layer_names(self) -> List[str]:
        """
        Get the names of all layers in the model.
        
        Returns:
            layer_names: List of layer names
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_layer_shape(self, layer_name: str) -> Tuple[int, ...]:
        """
        Get the shape of a layer's output.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            shape: Shape of the layer's output
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_layer_activations(self, 
                            layer_name: str, 
                            input_data: np.ndarray) -> np.ndarray:
        """
        Get activations for a specific layer given input data.
        
        Args:
            layer_name: Name of the layer
            input_data: Input data to feed into the model
            
        Returns:
            activations: Activations of the specified layer
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_weights(self, layer_name: str) -> Dict[str, np.ndarray]:
        """
        Get weights for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            weights: Dictionary of weight tensors for the layer
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_gradients(self, 
                    layer_name: str, 
                    input_data: np.ndarray, 
                    target_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get gradients for a specific layer given input and target data.
        
        Args:
            layer_name: Name of the layer
            input_data: Input data to feed into the model
            target_data: Target data for computing the loss
            
        Returns:
            gradients: Dictionary of gradient tensors for the layer
        """
        raise NotImplementedError("Subclasses must implement this method")


class PyTorchAdapter(ModelAdapter):
    """
    Adapter for PyTorch models.
    
    This class provides methods to work with PyTorch models in a unified way.
    """
    
    def __init__(self, model):
        """
        Initialize the PyTorch model adapter.
        
        Args:
            model: PyTorch model (nn.Module)
        """
        super().__init__()
        self.model = model
        self.hooks = {}
        self.activations = {}
    
    def get_layer_names(self) -> List[str]:
        """
        Get the names of all layers in the model.
        
        Returns:
            layer_names: List of layer names
        """
        layer_names = []
        for name, _ in self.model.named_modules():
            if name:  # Skip the root module
                layer_names.append(name)
        return layer_names
    
    def get_layer_shape(self, layer_name: str) -> Tuple[int, ...]:
        """
        Get the shape of a layer's output.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            shape: Shape of the layer's output
        """
        # This requires running a forward pass to get the shape
        # We'll return None for now and implement this properly later
        return None
    
    def _get_layer_by_name(self, layer_name: str):
        """
        Get a layer by name.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            layer: The layer module
        """
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def _register_hook(self, layer_name: str):
        """
        Register a forward hook for a layer.
        
        Args:
            layer_name: Name of the layer
        """
        layer = self._get_layer_by_name(layer_name)
        
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach().cpu().numpy()
        
        hook = layer.register_forward_hook(hook_fn)
        self.hooks[layer_name] = hook
    
    def get_layer_activations(self, 
                            layer_name: str, 
                            input_data: np.ndarray) -> np.ndarray:
        """
        Get activations for a specific layer given input data.
        
        Args:
            layer_name: Name of the layer
            input_data: Input data to feed into the model
            
        Returns:
            activations: Activations of the specified layer
        """
        import torch
        
        # Register hook if not already registered
        if layer_name not in self.hooks:
            self._register_hook(layer_name)
        
        # Convert input data to PyTorch tensor
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data)
        
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        input_data = input_data.to(device)
        
        # Forward pass
        with torch.no_grad():
            self.model(input_data)
        
        # Return activations
        return self.activations[layer_name]
    
    def get_weights(self, layer_name: str) -> Dict[str, np.ndarray]:
        """
        Get weights for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            weights: Dictionary of weight tensors for the layer
        """
        layer = self._get_layer_by_name(layer_name)
        weights = {}
        
        for name, param in layer.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        
        return weights
    
    def get_gradients(self, 
                    layer_name: str, 
                    input_data: np.ndarray, 
                    target_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get gradients for a specific layer given input and target data.
        
        Args:
            layer_name: Name of the layer
            input_data: Input data to feed into the model
            target_data: Target data for computing the loss
            
        Returns:
            gradients: Dictionary of gradient tensors for the layer
        """
        import torch
        import torch.nn as nn
        
        # Convert input and target data to PyTorch tensors
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data)
        if not isinstance(target_data, torch.Tensor):
            target_data = torch.tensor(target_data)
        
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        # Get the layer
        layer = self._get_layer_by_name(layer_name)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_data)
        
        # Compute loss
        criterion = nn.MSELoss()  # Default to MSE loss
        loss = criterion(output, target_data)
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = {}
        for name, param in layer.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy()
        
        return gradients


class TensorFlowAdapter(ModelAdapter):
    """
    Adapter for TensorFlow models.
    
    This class provides methods to work with TensorFlow models in a unified way.
    """
    
    def __init__(self, model):
        """
        Initialize the TensorFlow model adapter.
        
        Args:
            model: TensorFlow model (tf.keras.Model)
        """
        super().__init__()
        self.model = model
    
    def get_layer_names(self) -> List[str]:
        """
        Get the names of all layers in the model.
        
        Returns:
            layer_names: List of layer names
        """
        return [layer.name for layer in self.model.layers]
    
    def get_layer_shape(self, layer_name: str) -> Tuple[int, ...]:
        """
        Get the shape of a layer's output.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            shape: Shape of the layer's output
        """
        layer = self._get_layer_by_name(layer_name)
        return layer.output_shape
    
    def _get_layer_by_name(self, layer_name: str):
        """
        Get a layer by name.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            layer: The layer
        """
        for layer in self.model.layers:
            if layer.name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def get_layer_activations(self, 
                            layer_name: str, 
                            input_data: np.ndarray) -> np.ndarray:
        """
        Get activations for a specific layer given input data.
        
        Args:
            layer_name: Name of the layer
            input_data: Input data to feed into the model
            
        Returns:
            activations: Activations of the specified layer
        """
        import tensorflow as tf
        
        # Get the layer
        layer = self._get_layer_by_name(layer_name)
        
        # Create a new model that outputs the layer's activations
        activation_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
        
        # Get activations
        activations = activation_model.predict(input_data)
        
        return activations
    
    def get_weights(self, layer_name: str) -> Dict[str, np.ndarray]:
        """
        Get weights for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            weights: Dictionary of weight tensors for the layer
        """
        layer = self._get_layer_by_name(layer_name)
        weights = {}
        
        # Get weights
        layer_weights = layer.get_weights()
        
        # TensorFlow doesn't provide names for weights, so we'll use indices
        for i, w in enumerate(layer_weights):
            if i == 0:
                weights['kernel'] = w
            elif i == 1:
                weights['bias'] = w
            else:
                weights[f'weight_{i}'] = w
        
        return weights
    
    def get_gradients(self, 
                    layer_name: str, 
                    input_data: np.ndarray, 
                    target_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get gradients for a specific layer given input and target data.
        
        Args:
            layer_name: Name of the layer
            input_data: Input data to feed into the model
            target_data: Target data for computing the loss
            
        Returns:
            gradients: Dictionary of gradient tensors for the layer
        """
        import tensorflow as tf
        
        # Get the layer
        layer = self._get_layer_by_name(layer_name)
        
        # Convert input and target data to TensorFlow tensors
        if not isinstance(input_data, tf.Tensor):
            input_data = tf.convert_to_tensor(input_data)
        if not isinstance(target_data, tf.Tensor):
            target_data = tf.convert_to_tensor(target_data)
        
        # Define loss function
        loss_fn = tf.keras.losses.MeanSquaredError()  # Default to MSE loss
        
        # Get gradients
        with tf.GradientTape() as tape:
            tape.watch(layer.trainable_weights)
            output = self.model(input_data, training=True)
            loss = loss_fn(target_data, output)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        
        # Convert gradients to dictionary
        gradients_dict = {}
        for i, grad in enumerate(gradients):
            if i == 0:
                gradients_dict['kernel'] = grad.numpy()
            elif i == 1:
                gradients_dict['bias'] = grad.numpy()
            else:
                gradients_dict[f'weight_{i}'] = grad.numpy()
        
        return gradients_dict
