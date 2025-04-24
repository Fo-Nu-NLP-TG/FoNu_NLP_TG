# Module 2.2: Forward and Backward Propagation

This module explains how information flows through a neural network during training, focusing on forward propagation (making predictions) and backward propagation (learning from errors).

## 2.2.1 Forward Propagation

Forward propagation is the process of passing input data through the network to generate predictions. It involves sequential computation through each layer of the network.

### Mathematical Representation

For a neural network with L layers:

1. Input layer (layer 0): $a^{[0]} = X$
2. For each layer l from 1 to L:
   - Compute weighted sum: $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$
   - Apply activation function: $a^{[l]} = g^{[l]}(z^{[l]})$
3. Output: $\hat{y} = a^{[L]}$

Where:
- $X$ is the input data
- $W^{[l]}$ is the weight matrix for layer l
- $b^{[l]}$ is the bias vector for layer l
- $g^{[l]}$ is the activation function for layer l
- $a^{[l]}$ is the activation output from layer l
- $\hat{y}$ is the predicted output

### Code Implementation

```python
import numpy as np

def forward_propagation(X, parameters, activation_fn):
    """
    Implement forward propagation for a neural network
    
    Args:
        X: Input data of shape (input_size, batch_size)
        parameters: Dictionary containing weights and biases
        activation_fn: Activation function to use
        
    Returns:
        A dictionary containing the activations and weighted sums for each layer
    """
    # Get number of layers
    L = len(parameters) // 2
    
    # Store activations and weighted sums
    cache = {}
    
    # Input layer
    A = X
    cache["A0"] = A
    
    # Hidden layers and output layer
    for l in range(1, L+1):
        # Get weights and biases for current layer
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        
        # Compute weighted sum
        Z = np.dot(W, A) + b
        cache[f"Z{l}"] = Z
        
        # Apply activation function
        A = activation_fn(Z)
        cache[f"A{l}"] = A
    
    return cache

# Example usage
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Create a simple network with 2 layers
# Input size: 3, Hidden layer: 4, Output: 2
parameters = {
    "W1": np.random.randn(4, 3) * 0.01,
    "b1": np.zeros((4, 1)),
    "W2": np.random.randn(2, 4) * 0.01,
    "b2": np.zeros((2, 1))
}

# Sample input
X = np.random.randn(3, 5)  # 5 samples, 3 features each

# Forward propagation
cache = forward_propagation(X, parameters, sigmoid)

# Print output
print(f"Input shape: {cache['A0'].shape}")
print(f"Hidden layer output shape: {cache['A1'].shape}")
print(f"Output shape: {cache['A2'].shape}")
```

### Visualization of Forward Propagation

```mermaid
graph LR
    X[Input X] --> Z1[Z¹ = W¹X + b¹]
    Z1 --> A1[A¹ = g¹(Z¹)]
    A1 --> Z2[Z² = W²A¹ + b²]
    Z2 --> A2[A² = g²(Z²)]
    A2 --> Y[Output ŷ = A²]
    
    style Z1 fill:#f9f,stroke:#333,stroke-width:2px
    style Z2 fill:#f9f,stroke:#333,stroke-width:2px
    style A1 fill:#bbf,stroke:#333,stroke-width:2px
    style A2 fill:#bbf,stroke:#333,stroke-width:2px
```

![Forward Propagation](https://mermaid.ink/img/pako:eNp1ksFuwjAMhl8l8glQCX2BHiaNl-DANGm0Qy5xG7OSVEmGxDTePU5bVAbbKbG_fP7t2D6BNJKggkZFrZZkLD2o0NqHos1aoy2J8KI2ZMfVQ0FbcqwUWVLujawNzid_9sqRsuSsGUi8kSVrpSuisx05q5XrWfUkNFnnWu1asuPqvhAtOah6Vi9FdN4U0ZIdV-6liB05q924eiyiJWeVH1aPRbyRM2ZcPxXRkrPGD-vHIlrSoPtx_VRER96rKB7Xz0W05D35af1UREvO2fFcvxTRkXd2PNfPRbTknB3P9UsRLTnnxnP9WkRH3rnxXL8X0ZL3djyv34voyHs7ntcfRbTkvR3P688iOvLBjef1VxEt-eDG8_q7iJZ8cON5_VNER8GN5_VvES0FN57X_0W0FNx4Xv8V0VFw43l9KaKl4MZ1_Q-Nh6Yw?type=png)

## 2.2.2 Loss Functions

Loss functions measure how well the network's predictions match the true values. They quantify the error that the network aims to minimize during training.

### Common Loss Functions

#### Mean Squared Error (MSE)
Used for regression problems:

$$L(y, \hat{y}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

```python
def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE loss
    """
    return np.mean((y_true - y_pred) ** 2)
```

#### Binary Cross-Entropy
Used for binary classification problems:

$$L(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Calculate binary cross-entropy loss
    
    Args:
        y_true: True values (0 or 1)
        y_pred: Predicted probabilities
        epsilon: Small constant for numerical stability
        
    Returns:
        Binary cross-entropy loss
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss
```

### Visualization of Loss Functions

```python
import numpy as np
import matplotlib.pyplot as plt

# Create true values
y_true = np.array([0, 1, 0, 1])

# Create range of predictions
y_pred_range = np.linspace(0.01, 0.99, 100)

# Calculate losses for different predictions
mse_losses = []
bce_losses = []

for pred_value in y_pred_range:
    # Create predictions
    y_pred = np.array([pred_value] * 4)
    
    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    mse_losses.append(mse)
    
    # Calculate BCE
    bce = binary_cross_entropy(y_true, y_pred)
    bce_losses.append(bce)

# Plot losses
plt.figure(figsize=(12, 6))
plt.plot(y_pred_range, mse_losses, label='Mean Squared Error', color='blue')
plt.plot(y_pred_range, bce_losses, label='Binary Cross-Entropy', color='red')
plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3)
plt.grid(alpha=0.3)
plt.title('Loss Functions Comparison')
plt.xlabel('Prediction Value')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_functions.png')
plt.close()
```

![Loss Functions](https://mermaid.ink/img/pako:eNp1ksFuwjAMhl8l8glQCX2BHiaNl-DANGm0Qy5xG7OSVEmGxDTePU5bVAbbKbG_fP7t2D6BNJKggkZFrZZkLD2o0NqHos1aoy2J8KI2ZMfVQ0FbcqwUWVLujawNzid_9sqRsuSsGUi8kSVrpSuisx05q5XrWfUkNFnnWu1asuPqvhAtOah6Vi9FdN4U0ZIdV-6liB05q924eiyiJWeVH1aPRbyRM2ZcPxXRkrPGD-vHIlrSoPtx_VRER96rKB7Xz0W05D35af1UREvO2fFcvxTRkXd2PNfPRbTknB3P9UsRLTnnxnP9WkRH3rnxXL8X0ZL3djyv34voyHs7ntcfRbTkvR3P688iOvLBjef1VxEt-eDG8_q7iJZ8cON5_VNER8GN5_VvES0FN57X_0W0FNx4Xv8V0VFw43l9KaKl4MZ1_Q-Nh6Yw?type=png)

## 2.2.3 Backward Propagation

Backward propagation (or backpropagation) is the process of computing gradients of the loss function with respect to the network's parameters. These gradients are then used to update the parameters in the direction that minimizes the loss.

### Mathematical Representation

For a neural network with L layers:

1. Output layer (layer L):
   - $dZ^{[L]} = A^{[L]} - Y$ (for MSE with linear activation)
   - $dW^{[L]} = \frac{1}{m} dZ^{[L]} (A^{[L-1]})^T$
   - $db^{[L]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[L]}$

2. For each layer l from L-1 down to 1:
   - $dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \odot g^{[l]'}(Z^{[l]})$
   - $dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$
   - $db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l]}$

Where:
- $dZ^{[l]}$ is the gradient of the loss with respect to the weighted sum in layer l
- $dW^{[l]}$ is the gradient of the loss with respect to the weights in layer l
- $db^{[l]}$ is the gradient of the loss with respect to the biases in layer l
- $g^{[l]'}$ is the derivative of the activation function in layer l
- $\odot$ represents element-wise multiplication

### Code Implementation

```python
def sigmoid_derivative(Z):
    """
    Derivative of sigmoid function
    """
    s = 1 / (1 + np.exp(-Z))
    return s * (1 - s)

def backward_propagation(X, Y, parameters, cache, activation_fn_derivative):
    """
    Implement backward propagation for a neural network
    
    Args:
        X: Input data of shape (input_size, batch_size)
        Y: True labels of shape (output_size, batch_size)
        parameters: Dictionary containing weights and biases
        cache: Dictionary containing activations and weighted sums from forward propagation
        activation_fn_derivative: Derivative of the activation function
        
    Returns:
        A dictionary containing the gradients
    """
    # Get number of layers
    L = len(parameters) // 2
    
    # Get batch size
    m = X.shape[1]
    
    # Initialize gradients
    grads = {}
    
    # Output layer
    dZ = cache[f"A{L}"] - Y
    grads[f"dW{L}"] = (1/m) * np.dot(dZ, cache[f"A{L-1}"].T)
    grads[f"db{L}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    # Hidden layers
    for l in range(L-1, 0, -1):
        dA = np.dot(parameters[f"W{l+1}"].T, dZ)
        dZ = dA * activation_fn_derivative(cache[f"Z{l}"])
        grads[f"dW{l}"] = (1/m) * np.dot(dZ, cache[f"A{l-1}"].T)
        grads[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    return grads

# Example usage
# Continuing from the forward propagation example
Y = np.random.randint(0, 2, (2, 5))  # Binary labels for 5 samples

# Backward propagation
grads = backward_propagation(X, Y, parameters, cache, sigmoid_derivative)

# Print gradients
print(f"dW1 shape: {grads['dW1'].shape}")
print(f"db1 shape: {grads['db1'].shape}")
print(f"dW2 shape: {grads['dW2'].shape}")
print(f"db2 shape: {grads['db2'].shape}")
```

### Visualization of Backward Propagation

```mermaid
graph RL
    L[Loss] --> dA2[dA²]
    dA2 --> dZ2[dZ² = dA² ⊙ g²'(Z²)]
    dZ2 --> dW2[dW² = dZ² · (A¹)ᵀ]
    dZ2 --> db2[db² = sum(dZ²)]
    dZ2 --> dA1[dA¹ = (W²)ᵀ · dZ²]
    dA1 --> dZ1[dZ¹ = dA¹ ⊙ g¹'(Z¹)]
    dZ1 --> dW1[dW¹ = dZ¹ · Xᵀ]
    dZ1 --> db1[db¹ = sum(dZ¹)]
    
    style dZ1 fill:#f9f,stroke:#333,stroke-width:2px
    style dZ2 fill:#f9f,stroke:#333,stroke-width:2px
    style dW1 fill:#bbf,stroke:#333,stroke-width:2px
    style dW2 fill:#bbf,stroke:#333,stroke-width:2px
    style db1 fill:#bbf,stroke:#333,stroke-width:2px
    style db2 fill:#bbf,stroke:#333,stroke-width:2px
```

![Backward Propagation](https://mermaid.ink/img/pako:eNp1ksFuwjAMhl8l8glQCX2BHiaNl-DANGm0Qy5xG7OSVEmGxDTePU5bVAbbKbG_fP7t2D6BNJKggkZFrZZkLD2o0NqHos1aoy2J8KI2ZMfVQ0FbcqwUWVLujawNzid_9sqRsuSsGUi8kSVrpSuisx05q5XrWfUkNFnnWu1asuPqvhAtOah6Vi9FdN4U0ZIdV-6liB05q924eiyiJWeVH1aPRbyRM2ZcPxXRkrPGD-vHIlrSoPtx_VRER96rKB7Xz0W05D35af1UREvO2fFcvxTRkXd2PNfPRbTknB3P9UsRLTnnxnP9WkRH3rnxXL8X0ZL3djyv34voyHs7ntcfRbTkvR3P688iOvLBjef1VxEt-eDG8_q7iJZ8cON5_VNER8GN5_VvES0FN57X_0W0FNx4Xv8V0VFw43l9KaKl4MZ1_Q-Nh6Yw?type=png)

## 2.2.4 Parameter Updates

After computing the gradients, the network parameters are updated using an optimization algorithm. The most basic is gradient descent:

$$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$

Where $\alpha$ is the learning rate, which controls the step size.

```python
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Args:
        parameters: Dictionary containing weights and biases
        grads: Dictionary containing gradients
        learning_rate: Learning rate
        
    Returns:
        Updated parameters
    """
    # Get number of layers
    L = len(parameters) // 2
    
    # Update parameters for each layer
    for l in range(1, L+1):
        parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"]
    
    return parameters

# Example usage
# Continuing from the backward propagation example
learning_rate = 0.01
updated_parameters = update_parameters(parameters, grads, learning_rate)

# Print updated parameters
print(f"Original W1[0,0]: {parameters['W1'][0,0]}")
print(f"Updated W1[0,0]: {updated_parameters['W1'][0,0]}")
```

## 2.2.5 Complete Training Loop

Putting it all together, here's a complete training loop for a neural network:

```python
def train_neural_network(X, Y, layer_dims, learning_rate, num_iterations):
    """
    Train a neural network
    
    Args:
        X: Input data of shape (input_size, batch_size)
        Y: True labels of shape (output_size, batch_size)
        layer_dims: List of integers representing the dimensions of each layer
        learning_rate: Learning rate for gradient descent
        num_iterations: Number of training iterations
        
    Returns:
        Trained parameters and loss history
    """
    # Initialize parameters
    parameters = initialize_parameters(layer_dims)
    
    # Loss history
    loss_history = []
    
    # Training loop
    for i in range(num_iterations):
        # Forward propagation
        cache = forward_propagation(X, parameters, sigmoid)
        
        # Compute loss
        loss = binary_cross_entropy(Y, cache[f"A{len(layer_dims)-1}"])
        
        # Backward propagation
        grads = backward_propagation(X, Y, parameters, cache, sigmoid_derivative)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Record loss
        if i % 100 == 0:
            loss_history.append(loss)
            print(f"Iteration {i}, Loss: {loss}")
    
    return parameters, loss_history

def initialize_parameters(layer_dims):
    """
    Initialize parameters for a neural network
    
    Args:
        layer_dims: List of integers representing the dimensions of each layer
        
    Returns:
        Dictionary containing initialized weights and biases
    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    
    return parameters

# Example usage
# Create a simple dataset
np.random.seed(42)
X = np.random.randn(2, 100)  # 100 samples, 2 features each
Y = np.array([[(X[0,i] > 0) & (X[1,i] > 0) for i in range(X.shape[1])]])  # AND operation

# Define network architecture
layer_dims = [2, 4, 1]  # 2 input features, 4 hidden neurons, 1 output neuron

# Train the network
trained_parameters, loss_history = train_neural_network(X, Y, layer_dims, 0.1, 1000)

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(range(0, 1000, 100), loss_history)
plt.grid(alpha=0.3)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
plt.close()
```

![Training Loss](https://mermaid.ink/img/pako:eNp1ksFuwjAMhl8l8glQCX2BHiaNl-DANGm0Qy5xG7OSVEmGxDTePU5bVAbbKbG_fP7t2D6BNJKggkZFrZZkLD2o0NqHos1aoy2J8KI2ZMfVQ0FbcqwUWVLujawNzid_9sqRsuSsGUi8kSVrpSuisx05q5XrWfUkNFnnWu1asuPqvhAtOah6Vi9FdN4U0ZIdV-6liB05q924eiyiJWeVH1aPRbyRM2ZcPxXRkrPGD-vHIlrSoPtx_VRER96rKB7Xz0W05D35af1UREvO2fFcvxTRkXd2PNfPRbTknB3P9UsRLTnnxnP9WkRH3rnxXL8X0ZL3djyv34voyHs7ntcfRbTkvR3P688iOvLBjef1VxEt-eDG8_q7iJZ8cON5_VNER8GN5_VvES0FN57X_0W0FNx4Xv8V0VFw43l9KaKl4MZ1_Q-Nh6Yw?type=png)

## 2.2.6 Where Layer Normalization Fits

Layer Normalization is applied during the forward pass, typically after computing the weighted sum but before applying the activation function:

```python
def forward_propagation_with_layer_norm(X, parameters, activation_fn, epsilon=1e-8):
    """
    Implement forward propagation with layer normalization
    
    Args:
        X: Input data of shape (input_size, batch_size)
        parameters: Dictionary containing weights, biases, gammas, and betas
        activation_fn: Activation function to use
        epsilon: Small constant for numerical stability
        
    Returns:
        A dictionary containing the activations and weighted sums for each layer
    """
    # Get number of layers
    L = len(parameters) // 4  # We now have W, b, gamma, beta for each layer
    
    # Store activations and weighted sums
    cache = {}
    
    # Input layer
    A = X
    cache["A0"] = A
    
    # Hidden layers and output layer
    for l in range(1, L+1):
        # Get parameters for current layer
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        gamma = parameters[f"gamma{l}"]
        beta = parameters[f"beta{l}"]
        
        # Compute weighted sum
        Z = np.dot(W, A) + b
        cache[f"Z{l}"] = Z
        
        # Apply layer normalization
        # Calculate mean and variance along the feature dimension
        mean = np.mean(Z, axis=0, keepdims=True)
        var = np.var(Z, axis=0, keepdims=True)
        
        # Normalize
        Z_norm = (Z - mean) / np.sqrt(var + epsilon)
        cache[f"Z_norm{l}"] = Z_norm
        
        # Scale and shift
        Z_tilde = gamma * Z_norm + beta
        cache[f"Z_tilde{l}"] = Z_tilde
        
        # Apply activation function
        A = activation_fn(Z_tilde)
        cache[f"A{l}"] = A
    
    return cache
```

During backpropagation, we need to compute gradients through the layer normalization step:

```python
def backward_propagation_with_layer_norm(X, Y, parameters, cache, activation_fn_derivative, epsilon=1e-8):
    """
    Implement backward propagation with layer normalization
    
    Args:
        X: Input data of shape (input_size, batch_size)
        Y: True labels of shape (output_size, batch_size)
        parameters: Dictionary containing weights, biases, gammas, and betas
        cache: Dictionary containing activations and weighted sums from forward propagation
        activation_fn_derivative: Derivative of the activation function
        epsilon: Small constant for numerical stability
        
    Returns:
        A dictionary containing the gradients
    """
    # Implementation details omitted for brevity
    # This would involve computing gradients through the layer normalization step
    
    return grads
```

### Visualization of Layer Normalization in Forward and Backward Propagation

```mermaid
graph TD
    subgraph "Forward Propagation"
        X[Input X] --> Z1[Z = WX + b]
        Z1 --> ZN[Z_norm = (Z - μ) / √(σ² + ε)]
        ZN --> ZT[Z_tilde = γ · Z_norm + β]
        ZT --> A[A = g(Z_tilde)]
    end
    
    subgraph "Backward Propagation"
        dA[dA] --> dZT[dZ_tilde = dA ⊙ g'(Z_tilde)]
        dZT --> dGamma[dγ = sum(dZ_tilde ⊙ Z_norm)]
        dZT --> dBeta[dβ = sum(dZ_tilde)]
        dZT --> dZN[dZ_norm = dZ_tilde ⊙ γ]
        dZN --> dZ[dZ = complex gradient through normalization]
        dZ --> dW[dW = dZ · Xᵀ]
        dZ --> db[db = sum(dZ)]
    end
    
    style ZN fill:#f9f,stroke:#333,stroke-width:2px
    style ZT fill:#bbf,stroke:#333,stroke-width:2px
    style dZN fill:#f9f,stroke:#333,stroke-width:2px
    style dZ fill:#bbf,stroke:#333,stroke-width:2px
```

![Layer Normalization in Forward and Backward Propagation](https://mermaid.ink/img/pako:eNp1ksFuwjAMhl8l8glQCX2BHiaNl-DANGm0Qy5xG7OSVEmGxDTePU5bVAbbKbG_fP7t2D6BNJKggkZFrZZkLD2o0NqHos1aoy2J8KI2ZMfVQ0FbcqwUWVLujawNzid_9sqRsuSsGUi8kSVrpSuisx05q5XrWfUkNFnnWu1asuPqvhAtOah6Vi9FdN4U0ZIdV-6liB05q924eiyiJWeVH1aPRbyRM2ZcPxXRkrPGD-vHIlrSoPtx_VRER96rKB7Xz0W05D35af1UREvO2fFcvxTRkXd2PNfPRbTknB3P9UsRLTnnxnP9WkRH3rnxXL8X0ZL3djyv34voyHs7ntcfRbTkvR3P688iOvLBjef1VxEt-eDG8_q7iJZ8cON5_VNER8GN5_VvES0FN57X_0W0FNx4Xv8V0VFw43l9KaKl4MZ1_Q-Nh6Yw?type=png)

## Summary

In this module, we've covered the essential processes of forward and backward propagation in neural networks:

1. **Forward Propagation**: How inputs are transformed through the network to produce predictions
2. **Loss Functions**: How to measure the error between predictions and true values
3. **Backward Propagation**: How gradients are computed to update the network's parameters
4. **Parameter Updates**: How parameters are adjusted to minimize the loss
5. **Layer Normalization Integration**: How layer normalization fits into the forward and backward passes

In the next module, we'll explore the challenges in training deep neural networks and how normalization techniques help address these challenges.

## Practice Exercises

1. Implement a neural network with layer normalization from scratch and train it on a simple dataset.
2. Visualize the distribution of activations before and after layer normalization during training.
3. Compare the training dynamics (loss curves) of networks with and without layer normalization.
4. Implement backward propagation through the layer normalization step.
5. Experiment with different initialization strategies for the gamma and beta parameters in layer normalization.
