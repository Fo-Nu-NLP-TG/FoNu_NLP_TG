"""
Distribution analyzer for neural networks.

This module provides tools to analyze distributions in neural networks,
including detecting multimodality, long-tailed distributions, and specialized neurons.
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Any, Optional, Union

class DistributionAnalyzer:
    """
    Analyzer for neural network distributions.
    
    This class provides methods to analyze distributions in neural networks,
    including detecting multimodality, long-tailed distributions, and specialized neurons.
    """
    
    def __init__(self):
        """Initialize the distribution analyzer."""
        pass
    
    def detect_multimodality(self, 
                            data: np.ndarray, 
                            significance_threshold: float = 0.1) -> Tuple[bool, int, List[Tuple[float, float]]]:
        """
        Detect if a distribution is multimodal.
        
        Args:
            data: 1D array of values to analyze
            significance_threshold: Threshold for peak prominence (relative to max prominence)
            
        Returns:
            is_multimodal: Boolean indicating if the distribution is multimodal
            num_modes: Estimated number of modes
            peaks: List of (position, height) tuples for each peak
        """
        # Fit a kernel density estimate
        kde = stats.gaussian_kde(data)
        
        # Generate points to evaluate the KDE
        x = np.linspace(np.min(data), np.max(data), 1000)
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
            significant_peaks = [peak for peak, prom in zip(peaks, prominences) 
                                if prom > significance_threshold * max(prominences)]
            
            is_multimodal = len(significant_peaks) > 1
            num_modes = len(significant_peaks)
            return is_multimodal, num_modes, significant_peaks
        else:
            return False, 1, peaks
    
    def detect_long_tailed(self, 
                          data: np.ndarray, 
                          tail_threshold: float = 0.95) -> Tuple[bool, float, float]:
        """
        Detect if a distribution is long-tailed.
        
        Args:
            data: 1D array of values to analyze
            tail_threshold: Percentile threshold for defining the tail
            
        Returns:
            is_long_tailed: Boolean indicating if the distribution is long-tailed
            tail_index: Estimated tail index (lower values indicate heavier tails)
            tail_weight: Proportion of total mass in the tail
        """
        # Sort data
        sorted_data = np.sort(data)
        
        # Calculate tail threshold value
        threshold_value = np.percentile(data, tail_threshold * 100)
        
        # Extract tail data
        tail_data = sorted_data[sorted_data > threshold_value]
        
        # Calculate tail weight
        tail_weight = len(tail_data) / len(data)
        
        # Estimate tail index using Hill estimator
        if len(tail_data) > 10:  # Need enough tail points for reliable estimation
            log_tail = np.log(tail_data)
            tail_mean = np.mean(log_tail)
            tail_index = 1 / np.mean(log_tail - log_tail[0])
        else:
            tail_index = np.nan
        
        # Check if distribution is long-tailed
        # A distribution is considered long-tailed if the tail contains a significant
        # proportion of the total mass and the tail index is relatively low
        is_long_tailed = (tail_weight > 0.05) and (tail_index < 3 or np.isnan(tail_index))
        
        return is_long_tailed, tail_index, tail_weight
    
    def detect_specialized_neurons(self, 
                                 activations: np.ndarray, 
                                 threshold: float = 0.7) -> List[int]:
        """
        Detect specialized neurons based on activation patterns.
        
        Args:
            activations: 2D array of shape [num_samples, num_neurons] containing neuron activations
            threshold: Threshold for specialization detection
            
        Returns:
            specialized_indices: Indices of neurons that appear to be specialized
        """
        num_neurons = activations.shape[1]
        specialized_indices = []
        
        for i in range(num_neurons):
            neuron_acts = activations[:, i]
            
            # Check for multimodality
            is_multimodal, _, _ = self.detect_multimodality(neuron_acts)
            
            # Check for high sparsity (many zeros or near-zeros)
            sparsity = np.mean(np.abs(neuron_acts) < 1e-6)
            
            # Check for high activation variance
            normalized_variance = np.var(neuron_acts) / (np.mean(np.abs(neuron_acts)) + 1e-8) ** 2
            
            # A neuron is considered specialized if it has multimodal distribution,
            # high sparsity, or high normalized variance
            if is_multimodal or sparsity > threshold or normalized_variance > threshold:
                specialized_indices.append(i)
        
        return specialized_indices
    
    def calculate_distribution_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate various statistics for a distribution.
        
        Args:
            data: 1D array of values to analyze
            
        Returns:
            stats_dict: Dictionary containing various distribution statistics
        """
        stats_dict = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        }
        
        return stats_dict
