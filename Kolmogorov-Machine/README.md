# Kolmogorov-Machine

A general-purpose neural network distributions analyzer and visualizer that can natively analyze any neural network across frameworks like TensorFlow and PyTorch.

## Overview

Kolmogorov-Machine provides tools to analyze and visualize distributions in neural networks, including:

- Detecting multimodal distributions
- Identifying long-tailed distributions
- Finding specialized neurons
- Visualizing weight and activation distributions
- Comparing distributions across different layers and models
- Supporting both PyTorch and TensorFlow models

## Installation

```bash
# Basic installation
pip install kolmogorov-machine

# With PyTorch support
pip install kolmogorov-machine[pytorch]

# With TensorFlow support
pip install kolmogorov-machine[tensorflow]

# With all dependencies
pip install kolmogorov-machine[all]
```

## Quick Start

```python
import numpy as np
import torch
import torch.nn as nn
from kolmogorov_machine.analyzers import DistributionAnalyzer
from kolmogorov_machine.visualizers import DistributionVisualizer
from kolmogorov_machine.models import PyTorchAdapter

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Create a model adapter
adapter = PyTorchAdapter(model)

# Generate some random input data
input_data = torch.randn(100, 10)

# Get activations for a layer
activations = adapter.get_layer_activations('0', input_data)

# Analyze the distribution
analyzer = DistributionAnalyzer()
is_multimodal, num_modes, peaks = analyzer.detect_multimodality(activations[:, 0])
is_long_tailed, tail_index, tail_weight = analyzer.detect_long_tailed(activations[:, 0])
specialized_neurons = analyzer.detect_specialized_neurons(activations)

# Visualize the distribution
visualizer = DistributionVisualizer()
visualizer.plot_histogram(activations[:, 0], title='Neuron 0 Activations')
visualizer.plot_multimodal_detection(activations[:, 0])
visualizer.plot_long_tailed_detection(activations[:, 0])
```

## Features

### Distribution Analysis

- **Multimodality Detection**: Identify neurons with multimodal activation distributions
- **Long-Tailed Distribution Detection**: Find distributions with heavy tails
- **Specialized Neuron Detection**: Identify neurons that have specialized for specific features
- **Statistical Analysis**: Calculate various statistics for distributions

### Visualization

- **Histograms and KDEs**: Visualize distributions with histograms and kernel density estimates
- **Multimodal Visualization**: Visualize detected modes in distributions
- **Long-Tailed Visualization**: Visualize the tail region of distributions
- **Distribution Comparison**: Compare distributions across different layers, models, or training stages

### Framework Support

- **PyTorch Adapter**: Work with PyTorch models
- **TensorFlow Adapter**: Work with TensorFlow models
- **Unified Interface**: Use the same code regardless of the underlying framework

## Documentation

For detailed documentation, see the [docs](docs/) directory or visit our [documentation website](https://kolmogorov-machine.readthedocs.io/).

## Examples

Check out the [examples](examples/) directory for more examples of how to use Kolmogorov-Machine.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
