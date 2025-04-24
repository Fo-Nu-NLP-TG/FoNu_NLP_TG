"""
Distribution visualizer for neural networks.

This module provides tools to visualize distributions in neural networks,
including histograms, kernel density estimates, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional, Union

class DistributionVisualizer:
    """
    Visualizer for neural network distributions.
    
    This class provides methods to visualize distributions in neural networks,
    including histograms, kernel density estimates, and specialized visualizations.
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the distribution visualizer.
        
        Args:
            style: Matplotlib style to use for visualizations
        """
        self.style = style
        if style != 'default':
            plt.style.use(style)
    
    def plot_histogram(self, 
                      data: np.ndarray, 
                      title: str = 'Distribution Histogram', 
                      xlabel: str = 'Value', 
                      ylabel: str = 'Frequency',
                      bins: int = 50,
                      figsize: Tuple[int, int] = (10, 6),
                      color: str = 'blue',
                      alpha: float = 0.7,
                      show_stats: bool = True,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a histogram of the data.
        
        Args:
            data: 1D array of values to visualize
            title: Title for the plot
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            bins: Number of bins for the histogram
            figsize: Figure size (width, height) in inches
            color: Color for the histogram bars
            alpha: Transparency for the histogram bars
            show_stats: Whether to show statistics on the plot
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(data, bins=bins, alpha=alpha, color=color)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Add statistics if requested
        if show_stats:
            stats_text = (
                f"Mean: {np.mean(data):.4f}\n"
                f"Median: {np.median(data):.4f}\n"
                f"Std Dev: {np.std(data):.4f}\n"
                f"Min: {np.min(data):.4f}\n"
                f"Max: {np.max(data):.4f}"
            )
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_kde(self, 
                data: np.ndarray, 
                title: str = 'Kernel Density Estimate', 
                xlabel: str = 'Value', 
                ylabel: str = 'Density',
                figsize: Tuple[int, int] = (10, 6),
                color: str = 'blue',
                fill: bool = True,
                show_stats: bool = True,
                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a kernel density estimate of the data.
        
        Args:
            data: 1D array of values to visualize
            title: Title for the plot
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            figsize: Figure size (width, height) in inches
            color: Color for the KDE line
            fill: Whether to fill the area under the KDE curve
            show_stats: Whether to show statistics on the plot
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            fig: Matplotlib figure object
        """
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate KDE
        kde = stats.gaussian_kde(data)
        x = np.linspace(np.min(data), np.max(data), 1000)
        y = kde(x)
        
        # Plot KDE
        ax.plot(x, y, color=color, linewidth=2)
        if fill:
            ax.fill_between(x, y, alpha=0.3, color=color)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Add statistics if requested
        if show_stats:
            stats_text = (
                f"Mean: {np.mean(data):.4f}\n"
                f"Median: {np.median(data):.4f}\n"
                f"Std Dev: {np.std(data):.4f}\n"
                f"Skewness: {stats.skew(data):.4f}\n"
                f"Kurtosis: {stats.kurtosis(data):.4f}"
            )
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multimodal_detection(self, 
                                data: np.ndarray, 
                                title: str = 'Multimodal Distribution Detection', 
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a distribution with detected modes.
        
        Args:
            data: 1D array of values to visualize
            title: Title for the plot
            figsize: Figure size (width, height) in inches
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            fig: Matplotlib figure object
        """
        from scipy import stats
        from kolmogorov_machine.analyzers.distribution_analyzer import DistributionAnalyzer
        
        analyzer = DistributionAnalyzer()
        is_multimodal, num_modes, peaks = analyzer.detect_multimodality(data)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate KDE
        kde = stats.gaussian_kde(data)
        x = np.linspace(np.min(data), np.max(data), 1000)
        y = kde(x)
        
        # Plot KDE
        ax.plot(x, y, color='blue', linewidth=2)
        ax.fill_between(x, y, alpha=0.3, color='blue')
        
        # Mark peaks
        for x_peak, y_peak in peaks:
            ax.plot(x_peak, y_peak, 'ro', markersize=8)
            ax.vlines(x_peak, 0, y_peak, colors='r', linestyles='dashed', alpha=0.5)
        
        # Add title and labels
        modality_text = "Multimodal" if is_multimodal else "Unimodal"
        ax.set_title(f"{title} - {modality_text} ({num_modes} modes)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_long_tailed_detection(self, 
                                 data: np.ndarray, 
                                 title: str = 'Long-Tailed Distribution Detection', 
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a distribution with long-tail detection.
        
        Args:
            data: 1D array of values to visualize
            title: Title for the plot
            figsize: Figure size (width, height) in inches
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            fig: Matplotlib figure object
        """
        from kolmogorov_machine.analyzers.distribution_analyzer import DistributionAnalyzer
        
        analyzer = DistributionAnalyzer()
        is_long_tailed, tail_index, tail_weight = analyzer.detect_long_tailed(data)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        counts, bins, patches = ax.hist(data, bins=50, alpha=0.7, color='blue')
        
        # Mark the tail region
        threshold_value = np.percentile(data, 95)
        for i, patch in enumerate(patches):
            if bins[i] > threshold_value:
                patch.set_facecolor('red')
        
        # Add vertical line at the tail threshold
        ax.axvline(threshold_value, color='red', linestyle='dashed', 
                  label=f'Tail Threshold (95th percentile)')
        
        # Add title and labels
        tail_text = "Long-Tailed" if is_long_tailed else "Not Long-Tailed"
        ax.set_title(f"{title} - {tail_text}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        
        # Add statistics
        stats_text = (
            f"Tail Index: {tail_index:.4f}\n"
            f"Tail Weight: {tail_weight:.4f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid and legend
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distribution_comparison(self, 
                                   data_list: List[np.ndarray], 
                                   labels: List[str],
                                   title: str = 'Distribution Comparison', 
                                   xlabel: str = 'Value', 
                                   ylabel: str = 'Density',
                                   figsize: Tuple[int, int] = (10, 6),
                                   colors: Optional[List[str]] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple distributions for comparison.
        
        Args:
            data_list: List of 1D arrays to compare
            labels: List of labels for each distribution
            title: Title for the plot
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            figsize: Figure size (width, height) in inches
            colors: List of colors for each distribution (if None, uses default colors)
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            fig: Matplotlib figure object
        """
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use default colors if none provided
        if colors is None:
            colors = plt.cm.tab10.colors
        
        # Plot KDE for each distribution
        for i, (data, label) in enumerate(zip(data_list, labels)):
            color = colors[i % len(colors)]
            
            # Calculate KDE
            kde = stats.gaussian_kde(data)
            x = np.linspace(np.min(data), np.max(data), 1000)
            y = kde(x)
            
            # Plot KDE
            ax.plot(x, y, color=color, linewidth=2, label=label)
            ax.fill_between(x, y, alpha=0.1, color=color)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add grid and legend
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
