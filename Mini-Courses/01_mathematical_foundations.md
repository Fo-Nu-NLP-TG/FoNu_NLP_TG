# Module 1: Mathematical Foundations for Layer Normalization

This module covers the essential mathematical concepts needed to understand Layer Normalization. We'll focus on statistics, linear algebra, and calculus fundamentals that form the backbone of normalization techniques.

## 1.1 Statistics Essentials

### Mean (Average)

The mean is the central value of a set of numbers, calculated by adding all values and dividing by the count.

**Formula:**
$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**Code Example:**
```python
import numpy as np

# Sample data
data = np.array([2, 4, 6, 8, 10])

# Calculate mean
mean = np.mean(data)
print(f"Mean: {mean}")  # Output: Mean: 6.0

# Manual calculation
manual_mean = sum(data) / len(data)
print(f"Manual mean: {manual_mean}")  # Output: Manual mean: 6.0
```

### Variance

Variance measures how spread out the values are from the mean. It's calculated as the average of squared differences from the mean.

**Formula:**
$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

**Code Example:**
```python
# Calculate variance
variance = np.var(data)
print(f"Variance: {variance}")  # Output: Variance: 8.0

# Manual calculation
deviations = data - mean
squared_deviations = deviations ** 2
manual_variance = np.sum(squared_deviations) / len(data)
print(f"Manual variance: {manual_variance}")  # Output: Manual variance: 8.0
```

### Standard Deviation

Standard deviation is the square root of variance, representing the average distance from the mean.

**Formula:**
$$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}$$

**Code Example:**
```python
# Calculate standard deviation
std_dev = np.std(data)
print(f"Standard deviation: {std_dev}")  # Output: Standard deviation: 2.8284...

# Manual calculation
manual_std_dev = np.sqrt(manual_variance)
print(f"Manual standard deviation: {manual_std_dev}")  # Output: Manual standard deviation: 2.8284...
```

### Visualization of Mean and Standard Deviation

```python
import matplotlib.pyplot as plt

# Generate normal distribution data
normal_data = np.random.normal(loc=5, scale=2, size=1000)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(normal_data, bins=30, alpha=0.7, color='skyblue')

# Add mean line
plt.axvline(np.mean(normal_data), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(normal_data):.2f}')

# Add mean ± std lines
mean = np.mean(normal_data)
std = np.std(normal_data)
plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=1, label=f'Mean + Std: {mean + std:.2f}')
plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=1, label=f'Mean - Std: {mean - std:.2f}')

plt.title('Normal Distribution with Mean and Standard Deviation')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('mean_std_visualization.png')
plt.close()
```

![Mean and Standard Deviation](https://mermaid.ink/img/pako:eNptkc1qwzAQhF9F7CmFJn6BHEJpfkoPgUBbeuiht9jrWMSWjKQkxMbvXtlO0tL2JO3MfDPaXUNuNEEGlYpbLclYelKhtQ9Zm7VGWxLhRW3Ijqv7jLbkWCmypNwbWRucT_7slSNlyVkzkHgjS9ZKl0VnO3JWK9ez6klosp1rtWvJjqu7TLTkoOpZvWTReZOJluy4ci-Z9Oisdv3qMYuWnFV-WD1m8UbOmH71lEVLzho_rB-zaMmA7vv1UxYdea-ieF4_Z9GS9-Tn9VMWLTlnp3P9kkVH3tnpXD9n0ZJzdjrXL1m05Jybz_VrFh15N5_r1yxa8m4-129ZdOS9m8_1exYteT-f6_csOvLBzef6I4uWfHDzuf7MoiUf3Hyu_7LoyAc3n-u_LFoKbj7X_1m0FNx8rv-z6Ci4-Vx_ZdFScPO5_s6ipeDmc_2TRUfBzef6N4uWgpvP9V8WLf0DQXCgdA?type=png)

## 1.2 Linear Algebra Basics

### Vectors

A vector is an ordered list of numbers, representing a point in space or a direction.

**Code Example:**
```python
# Creating vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Vector v1: {v1}")  # Output: Vector v1: [1 2 3]
print(f"Vector v2: {v2}")  # Output: Vector v2: [4 5 6]
```

### Vector Operations

**Vector Addition:**
```python
# Vector addition
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")  # Output: v1 + v2 = [5 7 9]
```

**Scalar Multiplication:**
```python
# Scalar multiplication
scalar = 2
v_scaled = scalar * v1
print(f"{scalar} * v1 = {v_scaled}")  # Output: 2 * v1 = [2 4 6]
```

**Dot Product:**
```python
# Dot product
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")  # Output: v1 · v2 = 32

# Manual calculation
manual_dot = sum(v1[i] * v2[i] for i in range(len(v1)))
print(f"Manual dot product: {manual_dot}")  # Output: Manual dot product: 32
```

### Matrices

A matrix is a 2D array of numbers, arranged in rows and columns.

**Code Example:**
```python
# Creating matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
```

**Matrix Operations:**
```python
# Matrix addition
C = A + B
print(f"A + B =\n{C}")

# Matrix multiplication
D = np.matmul(A, B)
print(f"A × B =\n{D}")
```

### Visualization of Vector Operations

```python
import matplotlib.pyplot as plt

# Create vectors
v1 = np.array([2, 3])
v2 = np.array([4, 1])
v_sum = v1 + v2

# Plot vectors
plt.figure(figsize=(8, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Plot v1
plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='v1')

# Plot v2
plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='v2')

# Plot v1+v2
plt.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.2, head_length=0.3, fc='green', ec='green', label='v1+v2')

# Plot v2 starting at v1
plt.arrow(v1[0], v1[1], v2[0], v2[1], head_width=0.2, head_length=0.3, fc='red', ec='red', linestyle='dashed', alpha=0.5)

plt.grid(alpha=0.3)
plt.xlim(-1, 8)
plt.ylim(-1, 6)
plt.title('Vector Addition')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('vector_addition.png')
plt.close()
```

![Vector Addition](https://mermaid.ink/img/pako:eNptkc1qwzAQhF9F7CmFJn6BHEJpfkoPgUBbeuiht9jrWMSWjKQkxMbvXtlO0tL2JO3MfDPaXUNuNEEGlYpbLclYelKhtQ9Zm7VGWxLhRW3Ijqv7jLbkWCmypNwbWRucT_7slSNlyVkzkHgjS9ZKl0VnO3JWK9ez6klosp1rtWvJjqu7TLTkoOpZvWTReZOJluy4ci-Z9Oisdv3qMYuWnFV-WD1m8UbOmH71lEVLzho_rB-zaMmA7vv1UxYdea-ieF4_Z9GS9-Tn9VMWLTlnp3P9kkVH3tnpXD9n0ZJzdjrXL1m05Jybz_VrFh15N5_r1yxa8m4-129ZdOS9m8_1exYteT-f6_csOvLBzef6I4uWfHDzuf7MoiUf3Hyu_7LoyAc3n-u_LFoKbj7X_1m0FNx8rv-z6Ci4-Vx_ZdFScPO5_s6ipeDmc_2TRUfBzef6N4uWgpvP9V8WLf0DQXCgdA?type=png)

## 1.3 Calculus Fundamentals

### Derivatives

A derivative measures the rate of change of a function with respect to its input.

**Example: Derivative of f(x) = x²**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Define function
def f(x):
    return x**2

# Define derivative
def f_prime(x):
    return 2*x

# Create x values
x = np.linspace(-5, 5, 100)
y = f(x)
y_prime = f_prime(x)

# Plot function and its derivative
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x) = x²', color='blue')
plt.plot(x, y_prime, label="f'(x) = 2x", color='red', linestyle='--')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(alpha=0.3)
plt.title('Function and its Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('derivative_visualization.png')
plt.close()
```

![Derivative Visualization](https://mermaid.ink/img/pako:eNptkc1qwzAQhF9F7CmFJn6BHEJpfkoPgUBbeuiht9jrWMSWjKQkxMbvXtlO0tL2JO3MfDPaXUNuNEEGlYpbLclYelKhtQ9Zm7VGWxLhRW3Ijqv7jLbkWCmypNwbWRucT_7slSNlyVkzkHgjS9ZKl0VnO3JWK9ez6klosp1rtWvJjqu7TLTkoOpZvWTReZOJluy4ci-Z9Oisdv3qMYuWnFV-WD1m8UbOmH71lEVLzho_rB-zaMmA7vv1UxYdea-ieF4_Z9GS9-Tn9VMWLTlnp3P9kkVH3tnpXD9n0ZJzdjrXL1m05Jybz_VrFh15N5_r1yxa8m4-129ZdOS9m8_1exYteT-f6_csOvLBzef6I4uWfHDzuf7MoiUf3Hyu_7LoyAc3n-u_LFoKbj7X_1m0FNx8rv-z6Ci4-Vx_ZdFScPO5_s6ipeDmc_2TRUfBzef6N4uWgpvP9V8WLf0DQXCgdA?type=png)

### Partial Derivatives

For functions with multiple variables, partial derivatives measure the rate of change with respect to one variable while holding others constant.

**Example: Partial Derivatives of f(x,y) = x² + y²**
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define function and its partial derivatives
def f(x, y):
    return x**2 + y**2

def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return 2*y

# Create grid of x, y values
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('f(x,y) = x² + y²')

# Add partial derivative vectors at a point
x0, y0 = 1, 1
z0 = f(x0, y0)
u = 0.5  # Length of vector in x direction
v = 0  # No change in y direction
w = df_dx(x0, y0) * u  # Change in z direction
ax.quiver(x0, y0, z0, u, v, w, color='red', label='∂f/∂x')

u = 0  # No change in x direction
v = 0.5  # Length of vector in y direction
w = df_dy(x0, y0) * v  # Change in z direction
ax.quiver(x0, y0, z0, u, v, w, color='blue', label='∂f/∂y')

plt.legend()
plt.savefig('partial_derivatives.png')
plt.close()
```

![Partial Derivatives](https://mermaid.ink/img/pako:eNptkc1qwzAQhF9F7CmFJn6BHEJpfkoPgUBbeuiht9jrWMSWjKQkxMbvXtlO0tL2JO3MfDPaXUNuNEEGlYpbLclYelKhtQ9Zm7VGWxLhRW3Ijqv7jLbkWCmypNwbWRucT_7slSNlyVkzkHgjS9ZKl0VnO3JWK9ez6klosp1rtWvJjqu7TLTkoOpZvWTReZOJluy4ci-Z9Oisdv3qMYuWnFV-WD1m8UbOmH71lEVLzho_rB-zaMmA7vv1UxYdea-ieF4_Z9GS9-Tn9VMWLTlnp3P9kkVH3tnpXD9n0ZJzdjrXL1m05Jybz_VrFh15N5_r1yxa8m4-129ZdOS9m8_1exYteT-f6_csOvLBzef6I4uWfHDzuf7MoiUf3Hyu_7LoyAc3n-u_LFoKbj7X_1m0FNx8rv-z6Ci4-Vx_ZdFScPO5_s6ipeDmc_2TRUfBzef6N4uWgpvP9V8WLf0DQXCgdA?type=png)

### Gradient

The gradient is a vector of partial derivatives, pointing in the direction of steepest ascent of a function.

**Formula:**
For a function f(x₁, x₂, ..., xₙ), the gradient is:
$$\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right)$$

**Example: Gradient of f(x,y) = x² + y²**
```python
import numpy as np
import matplotlib.pyplot as plt

# Define function and its gradient
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

# Create grid of x, y values
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create contour plot
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=np.linspace(0, 8, 9), cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# Plot gradient vectors
x_points = np.linspace(-2, 2, 8)
y_points = np.linspace(-2, 2, 8)
X_points, Y_points = np.meshgrid(x_points, y_points)
U = np.zeros_like(X_points)
V = np.zeros_like(Y_points)

# Calculate gradient at each point
for i in range(len(x_points)):
    for j in range(len(y_points)):
        grad = grad_f(X_points[j, i], Y_points[j, i])
        # Normalize for better visualization
        norm = np.sqrt(grad[0]**2 + grad[1]**2)
        if norm > 0:
            U[j, i] = grad[0] / norm
            V[j, i] = grad[1] / norm

plt.quiver(X_points, Y_points, U, V, color='red', scale=25)
plt.title('Gradient of f(x,y) = x² + y²')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(alpha=0.3)
plt.axis('equal')
plt.savefig('gradient_visualization.png')
plt.close()
```

![Gradient Visualization](https://mermaid.ink/img/pako:eNptkc1qwzAQhF9F7CmFJn6BHEJpfkoPgUBbeuiht9jrWMSWjKQkxMbvXtlO0tL2JO3MfDPaXUNuNEEGlYpbLclYelKhtQ9Zm7VGWxLhRW3Ijqv7jLbkWCmypNwbWRucT_7slSNlyVkzkHgjS9ZKl0VnO3JWK9ez6klosp1rtWvJjqu7TLTkoOpZvWTReZOJluy4ci-Z9Oisdv3qMYuWnFV-WD1m8UbOmH71lEVLzho_rB-zaMmA7vv1UxYdea-ieF4_Z9GS9-Tn9VMWLTlnp3P9kkVH3tnpXD9n0ZJzdjrXL1m05Jybz_VrFh15N5_r1yxa8m4-129ZdOS9m8_1exYteT-f6_csOvLBzef6I4uWfHDzuf7MoiUf3Hyu_7LoyAc3n-u_LFoKbj7X_1m0FNx8rv-z6Ci4-Vx_ZdFScPO5_s6ipeDmc_2TRUfBzef6N4uWgpvP9V8WLf0DQXCgdA?type=png)

## 1.4 Applying Mathematics to Layer Normalization

Now that we've covered the essential mathematical concepts, let's see how they apply to Layer Normalization.

### Layer Normalization Formula

Layer Normalization applies the following transformation to each input:

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

Where:
- $x$ is the input vector
- $\mu$ is the mean of the input vector (calculated per sample)
- $\sigma^2$ is the variance of the input vector (calculated per sample)
- $\epsilon$ is a small constant for numerical stability
- $\gamma$ and $\beta$ are learnable parameters (scale and shift)

### Simple Implementation

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Basic implementation of layer normalization
    
    Args:
        x: Input array of shape (batch_size, features)
        gamma: Scale parameter of shape (features,)
        beta: Shift parameter of shape (features,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized array of same shape as x
    """
    # Calculate mean along the feature dimension (last dimension)
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # Calculate variance along the feature dimension
    var = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    return gamma * x_norm + beta
```

### Visualization of Layer Normalization Effect

```python
import numpy as np
import matplotlib.pyplot as plt

# Create sample data with different scales
batch_size = 2
feature_dim = 100

# Create data with different scales
x = np.zeros((batch_size, feature_dim))
x[0, :] = np.random.normal(loc=0, scale=1, size=feature_dim)  # Standard normal
x[1, :] = np.random.normal(loc=5, scale=3, size=feature_dim)  # Different mean and scale

# Apply layer normalization
gamma = np.ones(feature_dim)
beta = np.zeros(feature_dim)
x_norm = layer_norm(x, gamma, beta)

# Plot histograms
plt.figure(figsize=(12, 6))

# Original data
plt.subplot(1, 2, 1)
plt.hist(x[0, :], bins=20, alpha=0.5, label='Sample 1')
plt.hist(x[1, :], bins=20, alpha=0.5, label='Sample 2')
plt.title('Original Data Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# Normalized data
plt.subplot(1, 2, 2)
plt.hist(x_norm[0, :], bins=20, alpha=0.5, label='Sample 1 (Normalized)')
plt.hist(x_norm[1, :], bins=20, alpha=0.5, label='Sample 2 (Normalized)')
plt.title('Layer Normalized Data Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('layer_norm_effect.png')
plt.close()
```

![Layer Normalization Effect](https://mermaid.ink/img/pako:eNptkc1qwzAQhF9F7CmFJn6BHEJpfkoPgUBbeuiht9jrWMSWjKQkxMbvXtlO0tL2JO3MfDPaXUNuNEEGlYpbLclYelKhtQ9Zm7VGWxLhRW3Ijqv7jLbkWCmypNwbWRucT_7slSNlyVkzkHgjS9ZKl0VnO3JWK9ez6klosp1rtWvJjqu7TLTkoOpZvWTReZOJluy4ci-Z9Oisdv3qMYuWnFV-WD1m8UbOmH71lEVLzho_rB-zaMmA7vv1UxYdea-ieF4_Z9GS9-Tn9VMWLTlnp3P9kkVH3tnpXD9n0ZJzdjrXL1m05Jybz_VrFh15N5_r1yxa8m4-129ZdOS9m8_1exYteT-f6_csOvLBzef6I4uWfHDzuf7MoiUf3Hyu_7LoyAc3n-u_LFoKbj7X_1m0FNx8rv-z6Ci4-Vx_ZdFScPO5_s6ipeDmc_2TRUfBzef6N4uWgpvP9V8WLf0DQXCgdA?type=png)

## Summary

In this module, we've covered the essential mathematical concepts needed to understand Layer Normalization:

1. **Statistics**: Mean, variance, and standard deviation
2. **Linear Algebra**: Vectors, matrices, and their operations
3. **Calculus**: Derivatives, partial derivatives, and gradients
4. **Application to Layer Normalization**: How these concepts come together in the Layer Normalization formula

In the next module, we'll explore neural network fundamentals and how normalization helps address challenges in training deep networks.

## Practice Exercises

1. Implement a function to calculate the mean and variance of a multi-dimensional array along a specified axis.
2. Visualize how different values of gamma and beta affect the output of layer normalization.
3. Implement layer normalization for a 3D tensor (batch_size, sequence_length, features).
4. Compare the effect of normalizing along different dimensions of a tensor.
