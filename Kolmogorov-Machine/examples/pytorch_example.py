"""
Example of using Kolmogorov-Machine with a PyTorch model.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kolmogorov_machine.analyzers.distribution_analyzer import DistributionAnalyzer
from kolmogorov_machine.visualizers.distribution_visualizer import DistributionVisualizer
from kolmogorov_machine.models.model_adapter import PyTorchAdapter

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    # Create output directory if it doesn't exist
    os.makedirs('examples/outputs', exist_ok=True)
    
    # Create a model
    model = SimpleModel()
    
    # Create a model adapter
    adapter = PyTorchAdapter(model)
    
    # Print layer names
    print("Layer names:", adapter.get_layer_names())
    
    # Generate some random input data
    input_data = torch.randn(1000, 10)
    
    # Get activations for the first layer
    activations = adapter.get_layer_activations('fc1', input_data)
    
    # Create analyzer and visualizer
    analyzer = DistributionAnalyzer()
    visualizer = DistributionVisualizer()
    
    # Analyze and visualize distributions for each neuron in the first layer
    for i in range(min(5, activations.shape[1])):  # Limit to 5 neurons for brevity
        neuron_activations = activations[:, i]
        
        # Analyze distribution
        is_multimodal, num_modes, peaks = analyzer.detect_multimodality(neuron_activations)
        is_long_tailed, tail_index, tail_weight = analyzer.detect_long_tailed(neuron_activations)
        stats = analyzer.calculate_distribution_statistics(neuron_activations)
        
        # Print analysis results
        print(f"\nNeuron {i} Analysis:")
        print(f"  Multimodal: {is_multimodal} ({num_modes} modes)")
        print(f"  Long-tailed: {is_long_tailed} (tail index: {tail_index:.4f}, tail weight: {tail_weight:.4f})")
        print(f"  Statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}, skewness={stats['skewness']:.4f}")
        
        # Visualize distribution
        fig = visualizer.plot_histogram(
            neuron_activations, 
            title=f'Neuron {i} Activations',
            save_path=f'examples/outputs/neuron_{i}_histogram.png'
        )
        plt.close(fig)
        
        fig = visualizer.plot_multimodal_detection(
            neuron_activations,
            title=f'Neuron {i} Multimodality',
            save_path=f'examples/outputs/neuron_{i}_multimodal.png'
        )
        plt.close(fig)
        
        fig = visualizer.plot_long_tailed_detection(
            neuron_activations,
            title=f'Neuron {i} Long-Tailed Analysis',
            save_path=f'examples/outputs/neuron_{i}_long_tailed.png'
        )
        plt.close(fig)
    
    # Detect specialized neurons
    specialized_indices = analyzer.detect_specialized_neurons(activations)
    print(f"\nSpecialized neurons: {specialized_indices}")
    
    # Compare distributions of different neurons
    neuron_indices = list(range(min(5, activations.shape[1])))
    data_list = [activations[:, i] for i in neuron_indices]
    labels = [f'Neuron {i}' for i in neuron_indices]
    
    fig = visualizer.plot_distribution_comparison(
        data_list,
        labels,
        title='Neuron Activation Comparison',
        save_path='examples/outputs/neuron_comparison.png'
    )
    plt.close(fig)
    
    print("\nAnalysis complete. Visualizations saved to examples/outputs/")

if __name__ == "__main__":
    main()
