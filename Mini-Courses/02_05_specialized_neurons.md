# Module 2.5: Specialized Neurons in Neural Networks

This module explains what specialized neurons are, different types of specialized neurons, and how they contribute to neural network functionality.

## 2.5.1 What Are Specialized Neurons?

Specialized neurons are neurons in a neural network that have been designed or have learned to perform specific functions or detect particular patterns in the input data.

### Types of Specialization

Neurons can specialize in several ways:

1. **Architectural Specialization**: Neurons designed with specific structures or connections
2. **Functional Specialization**: Neurons that perform specific mathematical operations
3. **Feature Specialization**: Neurons that learn to detect specific features in the data
4. **Task Specialization**: Neurons dedicated to particular sub-tasks within the network

### How Specialization Emerges

Specialization can emerge through:

1. **Training**: Neurons naturally specialize as they learn from data
2. **Architecture Design**: Networks can be designed to encourage specialization
3. **Regularization**: Techniques like dropout can promote more robust specialization
4. **Transfer Learning**: Pre-trained neurons may already be specialized for certain features

## 2.5.2 Architectural Specialized Neurons

Some neurons have specialized architectures designed for specific purposes.

### LSTM Cells (Long Short-Term Memory)

LSTM cells are specialized neurons designed to remember information over long periods:

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Forget gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Cell state
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        
        # Combine input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Calculate gates
        i = torch.sigmoid(self.W_i(combined))  # Input gate
        f = torch.sigmoid(self.W_f(combined))  # Forget gate
        o = torch.sigmoid(self.W_o(combined))  # Output gate
        
        # Update cell state
        c_tilde = torch.tanh(self.W_c(combined))
        c = f * c_prev + i * c_tilde
        
        # Calculate hidden state
        h = o * torch.tanh(c)
        
        return h, c
```

**Key Components of LSTM Cells**:
- **Input Gate**: Controls what new information to store
- **Forget Gate**: Controls what information to discard
- **Output Gate**: Controls what information to output
- **Cell State**: Long-term memory storage

### GRU Cells (Gated Recurrent Unit)

GRU cells are a simplified version of LSTM cells:

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x, h_prev):
        # Combine input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Calculate gates
        z = torch.sigmoid(self.W_z(combined))  # Update gate
        r = torch.sigmoid(self.W_r(combined))  # Reset gate
        
        # Calculate candidate hidden state
        combined_r = torch.cat((x, r * h_prev), dim=1)
        h_tilde = torch.tanh(self.W_h(combined_r))
        
        # Calculate new hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
```

**Key Components of GRU Cells**:
- **Update Gate**: Controls how much of the previous state to keep
- **Reset Gate**: Controls how much of the previous state to use in computing the new state

### Attention Mechanisms

Attention mechanisms are specialized components that allow networks to focus on specific parts of the input:

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        
        # Create query, key, value projections
        Q = self.query(x)  # [batch_size, seq_len, hidden_size]
        K = self.key(x)    # [batch_size, seq_len, hidden_size]
        V = self.value(x)  # [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

**Key Components of Attention Mechanisms**:
- **Query**: What we're looking for
- **Key**: What we match against
- **Value**: What we retrieve
- **Attention Weights**: How much to focus on each part of the input

## 2.5.3 Functional Specialized Neurons

Some neurons are specialized to perform specific mathematical functions.

### Radial Basis Function (RBF) Neurons

RBF neurons compute the distance between the input and a center point:

```python
class RBFLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(RBFLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Centers for RBF neurons
        self.centres = nn.Parameter(torch.Tensor(output_size, input_size))
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigma, 1)
        
    def forward(self, x):
        # Calculate squared distance between inputs and centers
        size = (x.size(0), self.output_size, self.input_size)
        x = x.unsqueeze(1).expand(size)
        centres = self.centres.unsqueeze(0).expand(size)
        
        # Squared distances
        distances = ((x - centres) ** 2).sum(dim=2)
        
        # Apply RBF function
        return torch.exp(-distances / (2 * self.sigma.unsqueeze(0) ** 2))
```

**Key Properties of RBF Neurons**:
- Output is highest when input is close to the center
- Output decreases as input moves away from the center
- Useful for function approximation and pattern recognition

### Capsule Neurons

Capsule neurons output vectors instead of scalars and use dynamic routing:

```python
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, routing_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.routing_iterations = routing_iterations
        
        # Transformation matrices
        self.W = nn.Parameter(torch.randn(num_capsules, in_channels, out_channels))
        
    def squash(self, x):
        # Squashing function for capsule outputs
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x):
        # x shape: [batch_size, num_input_capsules, in_channels]
        
        # Expand input for each output capsule
        x = x.unsqueeze(1).repeat(1, self.num_capsules, 1, 1)
        
        # Transform input
        u_hat = torch.matmul(x, self.W)
        
        # Initialize routing logits
        b = torch.zeros(x.size(0), self.num_capsules, x.size(2), 1)
        
        # Dynamic routing
        for i in range(self.routing_iterations):
            c = torch.softmax(b, dim=1)
            s = (c * u_hat).sum(dim=2)
            v = self.squash(s)
            
            if i < self.routing_iterations - 1:
                b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1, keepdim=True)
        
        return v
```

**Key Properties of Capsule Neurons**:
- Output vectors represent entity properties
- Length of vector represents probability of entity existence
- Direction of vector represents entity properties
- Dynamic routing allows parts to be assigned to wholes

### Self-Organizing Map (SOM) Neurons

SOM neurons organize themselves to represent the input space:

```python
class SOM(nn.Module):
    def __init__(self, input_size, map_size):
        super(SOM, self).__init__()
        self.input_size = input_size
        self.map_size = map_size
        
        # Weight vectors for each neuron in the map
        self.weights = nn.Parameter(torch.randn(map_size[0], map_size[1], input_size))
        
    def forward(self, x):
        # Calculate distance between input and each neuron
        batch_size = x.size(0)
        input_expanded = x.unsqueeze(1).unsqueeze(1)
        weights_expanded = self.weights.unsqueeze(0)
        
        # Euclidean distance
        distances = torch.sqrt(((input_expanded - weights_expanded) ** 2).sum(dim=3))
        
        # Find best matching unit (BMU)
        _, bmu_indices = distances.min(dim=2, keepdim=True)
        bmu_indices = bmu_indices.min(dim=1, keepdim=True)[1]
        
        return distances, bmu_indices
```

**Key Properties of SOM Neurons**:
- Self-organize to represent the input distribution
- Preserve topological properties of the input space
- Useful for dimensionality reduction and visualization

## 2.5.4 Feature-Specialized Neurons

During training, neurons often specialize to detect specific features in the data.

### Convolutional Filters as Feature Detectors

In convolutional neural networks (CNNs), filters specialize to detect specific visual features:

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

# Visualize filters
def visualize_filters(model):
    # Get the weights from the first convolutional layer
    filters = model.conv1.weight.data.cpu().numpy()
    
    # Plot filters
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            ax.imshow(filters[i, 0], cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cnn_filters.png')
    plt.close()
```

**Examples of Feature Specialization in CNNs**:
- Edge detectors (horizontal, vertical, diagonal)
- Texture detectors
- Color detectors
- Shape detectors
- More complex patterns in deeper layers

### Visualizing Neuron Specialization

We can visualize what features neurons have specialized to detect:

```python
def visualize_neuron_activation(model, layer_idx, neuron_idx, input_size=(28, 28)):
    """
    Generate an input that maximally activates a specific neuron
    using gradient ascent.
    """
    # Create a random input image
    x = torch.randn(1, 1, *input_size, requires_grad=True)
    
    # Define optimizer
    optimizer = torch.optim.Adam([x], lr=0.1)
    
    # Extract the specified layer
    layers = list(model.children())
    target_layer = nn.Sequential(*layers[:layer_idx+1])
    
    # Optimization loop
    for i in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        activation = target_layer(x)
        
        # Get activation of the target neuron
        if len(activation.shape) == 4:  # Convolutional layer
            target_activation = activation[0, neuron_idx].mean()
        else:  # Fully connected layer
            target_activation = activation[0, neuron_idx]
        
        # Compute loss (negative activation to maximize it)
        loss = -target_activation
        
        # Backward pass
        loss.backward()
        
        # Update input
        optimizer.step()
    
    # Return the optimized input
    return x.detach().cpu().numpy()[0, 0]

# Visualize what activates specific neurons
def visualize_neuron_specialization(model, layer_name='conv1'):
    # Get the layer
    if layer_name == 'conv1':
        layer_idx = 0
        num_neurons = model.conv1.out_channels
    elif layer_name == 'conv2':
        layer_idx = 2
        num_neurons = model.conv2.out_channels
    
    # Generate and visualize optimal inputs
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_neurons:
            optimal_input = visualize_neuron_activation(model, layer_idx, i)
            ax.imshow(optimal_input, cmap='viridis')
            ax.set_title(f'Neuron {i}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{layer_name}_specialization.png')
    plt.close()
```

## 2.5.5 Task-Specialized Neurons

Some neurons specialize in performing specific sub-tasks within the network.

### Output Neurons in Multi-Task Learning

In multi-task learning, different output neurons specialize for different tasks:

```python
class MultiTaskNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_regression_outputs):
        super(MultiTaskNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Task-specific layers
        self.classification = nn.Linear(hidden_size, num_classes)
        self.regression = nn.Linear(hidden_size, num_regression_outputs)
        
    def forward(self, x):
        # Shared representation
        shared_features = self.shared(x)
        
        # Task-specific outputs
        classification_output = self.classification(shared_features)
        regression_output = self.regression(shared_features)
        
        return classification_output, regression_output
```

**Examples of Task Specialization**:
- Classification neurons
- Regression neurons
- Segmentation neurons
- Detection neurons

### Mixture of Experts

In a Mixture of Experts (MoE) architecture, different "expert" networks specialize in different parts of the input space:

```python
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts):
        super(MixtureOfExperts, self).__init__()
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # Calculate gating weights
        gates = self.gate(x)
        
        # Apply each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        # Combine expert outputs using gates
        expert_outputs = expert_outputs.permute(1, 0, 2)
        gates = gates.unsqueeze(-1)
        
        # Weighted sum of expert outputs
        output = torch.sum(expert_outputs * gates, dim=1)
        
        return output, gates.squeeze(-1)
```

**Key Properties of MoE**:
- Each expert specializes in a different region of the input space
- Gating network decides which experts to use for each input
- Allows for efficient handling of complex tasks

## 2.5.6 Detecting and Analyzing Specialized Neurons

We can detect and analyze specialized neurons in trained networks.

### Activation Maximization

Finding inputs that maximally activate specific neurons:

```python
def activation_maximization(model, layer_idx, neuron_idx, input_shape=(3, 224, 224), iterations=100):
    # Create a random input
    x = torch.randn(1, *input_shape, requires_grad=True)
    
    # Define optimizer
    optimizer = torch.optim.Adam([x], lr=0.1)
    
    # Get the target layer
    target_layer = None
    for i, layer in enumerate(model.children()):
        if i == layer_idx:
            target_layer = layer
            break
    
    # Optimization loop
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass to the target layer
        activation = x
        for j, layer in enumerate(model.children()):
            activation = layer(activation)
            if j == layer_idx:
                break
        
        # Target the specific neuron
        if len(activation.shape) == 4:  # Conv layer
            target_activation = activation[0, neuron_idx].sum()
        else:  # FC layer
            target_activation = activation[0, neuron_idx]
        
        # Maximize activation
        loss = -target_activation
        loss.backward()
        optimizer.step()
    
    return x.detach()
```

### Feature Visualization

Visualizing what features neurons have learned to detect:

```python
def feature_visualization(model, layer_name, neuron_idx, input_shape=(3, 224, 224)):
    # Get layer index
    layer_idx = None
    for i, (name, _) in enumerate(model.named_children()):
        if name == layer_name:
            layer_idx = i
            break
    
    if layer_idx is None:
        raise ValueError(f"Layer {layer_name} not found")
    
    # Generate input that maximizes neuron activation
    optimal_input = activation_maximization(model, layer_idx, neuron_idx, input_shape)
    
    # Convert to image
    img = optimal_input[0].permute(1, 2, 0).cpu().numpy()
    
    # Normalize for visualization
    img = (img - img.min()) / (img.max() - img.min())
    
    return img
```

### Network Dissection

Identifying what real-world concepts neurons represent:

```python
def network_dissection(model, dataset, layer_name, threshold=0.5):
    """
    Simplified network dissection to identify what concepts
    neurons in a layer represent.
    """
    # Get activations for all images
    activations = []
    concepts = []
    
    for images, labels in dataset:
        # Forward pass to the target layer
        act = get_layer_activation(model, images, layer_name)
        activations.append(act)
        concepts.append(labels)
    
    # Stack activations and concepts
    activations = torch.cat(activations, dim=0)
    concepts = torch.cat(concepts, dim=0)
    
    # For each neuron, find which concepts it activates for
    num_neurons = activations.shape[1]
    neuron_concepts = []
    
    for i in range(num_neurons):
        # Get activations for this neuron
        neuron_act = activations[:, i]
        
        # Binarize activations using threshold
        binary_act = (neuron_act > threshold).float()
        
        # Calculate IoU with each concept
        concept_iou = []
        for j in range(concepts.shape[1]):
            concept = concepts[:, j]
            intersection = (binary_act * concept).sum()
            union = binary_act.sum() + concept.sum() - intersection
            iou = intersection / (union + 1e-8)
            concept_iou.append(iou.item())
        
        # Find best matching concept
        best_concept = np.argmax(concept_iou)
        best_iou = concept_iou[best_concept]
        
        neuron_concepts.append((best_concept, best_iou))
    
    return neuron_concepts
```

## 2.5.7 Bimodal and Multimodal Distributions in Specialized Neurons

When neurons specialize, they can develop bimodal or multimodal activation distributions.

### What Are Bimodal/Multimodal Distributions?

Bimodal distributions have two peaks, while multimodal distributions have multiple peaks. These can indicate:

1. **Feature Specialization**: The neuron responds strongly to specific features
2. **Class Specialization**: The neuron distinguishes between different classes
3. **Context Specialization**: The neuron activates differently in different contexts

### Detecting Multimodal Distributions

We can detect multimodal distributions in neuron activations:

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def detect_multimodality(activations, neuron_idx):
    """
    Detect if a neuron's activation distribution is multimodal.
    
    Args:
        activations: Tensor of shape [num_samples, num_neurons]
        neuron_idx: Index of the neuron to analyze
        
    Returns:
        is_multimodal: Boolean indicating if the distribution is multimodal
        num_modes: Estimated number of modes
    """
    # Get activations for the specified neuron
    neuron_acts = activations[:, neuron_idx].cpu().numpy()
    
    # Fit a kernel density estimate
    kde = stats.gaussian_kde(neuron_acts)
    
    # Generate points to evaluate the KDE
    x = np.linspace(neuron_acts.min(), neuron_acts.max(), 1000)
    y = kde(x)
    
    # Find peaks (modes)
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            peaks.append((x[i], y[i]))
    
    # Filter peaks by prominence
    if len(peaks) > 1:
        # Calculate prominence of each peak
        prominences = []
        for i, (x_peak, y_peak) in enumerate(peaks):
            # Find valleys between peaks
            if i == 0:
                left_valley = y[0]
            else:
                left_idx = np.argmin(y[np.where(x < x_peak)[0]])
                left_valley = y[left_idx]
            
            if i == len(peaks) - 1:
                right_valley = y[-1]
            else:
                right_idx = np.argmin(y[np.where(x > x_peak)[0]])
                right_valley = y[right_idx]
            
            # Prominence is the minimum height above surrounding valleys
            prominence = y_peak - max(left_valley, right_valley)
            prominences.append(prominence)
        
        # Filter peaks by prominence
        significant_peaks = [peak for peak, prom in zip(peaks, prominences) if prom > 0.1 * max(prominences)]
        
        is_multimodal = len(significant_peaks) > 1
        num_modes = len(significant_peaks)
    else:
        is_multimodal = False
        num_modes = 1
    
    return is_multimodal, num_modes, x, y, peaks

# Visualize multimodal distributions
def visualize_neuron_distribution(activations, neuron_idx):
    is_multimodal, num_modes, x, y, peaks = detect_multimodality(activations, neuron_idx)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    
    # Mark peaks
    for x_peak, y_peak in peaks:
        plt.plot(x_peak, y_peak, 'ro')
    
    plt.title(f"Neuron {neuron_idx} - {'Multimodal' if is_multimodal else 'Unimodal'} ({num_modes} modes)")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.savefig(f"neuron_{neuron_idx}_distribution.png")
    plt.close()
```

### Visualizing Specialized Neurons with Multimodal Distributions

We can visualize what inputs cause different modes of activation:

```python
def visualize_multimodal_neuron(model, dataset, layer_name, neuron_idx):
    """
    Visualize inputs that cause different modes of activation for a neuron.
    """
    # Get activations for all images
    activations = []
    images_list = []
    
    for images, _ in dataset:
        # Forward pass to the target layer
        act = get_layer_activation(model, images, layer_name)
        activations.append(act)
        images_list.append(images)
    
    # Stack activations and images
    activations = torch.cat(activations, dim=0)
    images = torch.cat(images_list, dim=0)
    
    # Get activations for the specified neuron
    neuron_acts = activations[:, neuron_idx].cpu().numpy()
    
    # Detect modality
    is_multimodal, num_modes, x, y, peaks = detect_multimodality(activations, neuron_idx)
    
    if is_multimodal:
        # For each peak, find images that activate near that peak
        plt.figure(figsize=(15, 5 * num_modes))
        
        for i, (x_peak, _) in enumerate(peaks):
            # Find images with activations close to this peak
            distances = np.abs(neuron_acts - x_peak)
            closest_indices = np.argsort(distances)[:5]
            
            # Plot these images
            for j, idx in enumerate(closest_indices):
                plt.subplot(num_modes, 5, i*5 + j + 1)
                img = images[idx].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
                plt.title(f"Act: {neuron_acts[idx]:.2f}")
                plt.axis('off')
        
        plt.suptitle(f"Neuron {neuron_idx} - Multimodal ({num_modes} modes)")
        plt.tight_layout()
        plt.savefig(f"neuron_{neuron_idx}_multimodal.png")
        plt.close()
```

## Summary

In this module, we've explored specialized neurons in neural networks:

1. **What Specialized Neurons Are**: Neurons designed or trained to perform specific functions
2. **Architectural Specialized Neurons**: LSTM cells, GRU cells, attention mechanisms
3. **Functional Specialized Neurons**: RBF neurons, capsule neurons, SOM neurons
4. **Feature-Specialized Neurons**: Neurons that detect specific features in the data
5. **Task-Specialized Neurons**: Neurons dedicated to specific sub-tasks
6. **Detecting and Analyzing Specialization**: Methods to visualize and understand neuron specialization
7. **Bimodal/Multimodal Distributions**: How specialized neurons can develop multimodal activation patterns

Understanding specialized neurons helps us design better neural network architectures and interpret how networks process information.

## Practice Exercises

1. Implement a simple CNN and visualize what features different filters learn to detect.
2. Train a network on a classification task and analyze which neurons specialize for which classes.
3. Implement a Mixture of Experts model and visualize how different experts specialize for different inputs.
4. Detect and visualize neurons with multimodal activation distributions in a pre-trained network.
