"""
Tests for the distribution analyzer.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kolmogorov_machine.analyzers.distribution_analyzer import DistributionAnalyzer

class TestDistributionAnalyzer(unittest.TestCase):
    """Tests for the DistributionAnalyzer class."""
    
    def setUp(self):
        """Set up the test case."""
        self.analyzer = DistributionAnalyzer()
    
    def test_detect_multimodality_unimodal(self):
        """Test multimodality detection on a unimodal distribution."""
        # Generate a unimodal normal distribution
        np.random.seed(42)
        data = np.random.normal(loc=0, scale=1, size=1000)
        
        # Detect multimodality
        is_multimodal, num_modes, peaks = self.analyzer.detect_multimodality(data)
        
        # Check results
        self.assertFalse(is_multimodal)
        self.assertEqual(num_modes, 1)
        self.assertEqual(len(peaks), 1)
    
    def test_detect_multimodality_bimodal(self):
        """Test multimodality detection on a bimodal distribution."""
        # Generate a bimodal distribution
        np.random.seed(42)
        data1 = np.random.normal(loc=-3, scale=1, size=500)
        data2 = np.random.normal(loc=3, scale=1, size=500)
        data = np.concatenate([data1, data2])
        
        # Detect multimodality
        is_multimodal, num_modes, peaks = self.analyzer.detect_multimodality(data)
        
        # Check results
        self.assertTrue(is_multimodal)
        self.assertEqual(num_modes, 2)
        self.assertEqual(len(peaks), 2)
    
    def test_detect_long_tailed_normal(self):
        """Test long-tailed detection on a normal distribution."""
        # Generate a normal distribution
        np.random.seed(42)
        data = np.random.normal(loc=0, scale=1, size=1000)
        
        # Detect long-tailed
        is_long_tailed, tail_index, tail_weight = self.analyzer.detect_long_tailed(data)
        
        # Check results
        self.assertFalse(is_long_tailed)
        self.assertGreater(tail_weight, 0)
    
    def test_detect_long_tailed_exponential(self):
        """Test long-tailed detection on an exponential distribution."""
        # Generate an exponential distribution
        np.random.seed(42)
        data = np.random.exponential(scale=1, size=1000)
        
        # Detect long-tailed
        is_long_tailed, tail_index, tail_weight = self.analyzer.detect_long_tailed(data)
        
        # Check results
        self.assertTrue(is_long_tailed)
        self.assertGreater(tail_weight, 0)
    
    def test_detect_specialized_neurons(self):
        """Test specialized neuron detection."""
        # Generate random activations
        np.random.seed(42)
        activations = np.random.normal(loc=0, scale=1, size=(1000, 10))
        
        # Make one neuron specialized (multimodal)
        activations[:500, 0] = np.random.normal(loc=-3, scale=1, size=500)
        activations[500:, 0] = np.random.normal(loc=3, scale=1, size=500)
        
        # Make another neuron specialized (sparse)
        activations[:900, 1] = 0
        activations[900:, 1] = np.random.normal(loc=5, scale=1, size=100)
        
        # Detect specialized neurons
        specialized_indices = self.analyzer.detect_specialized_neurons(activations)
        
        # Check results
        self.assertIn(0, specialized_indices)
        self.assertIn(1, specialized_indices)
    
    def test_calculate_distribution_statistics(self):
        """Test calculation of distribution statistics."""
        # Generate a normal distribution
        np.random.seed(42)
        data = np.random.normal(loc=0, scale=1, size=1000)
        
        # Calculate statistics
        stats = self.analyzer.calculate_distribution_statistics(data)
        
        # Check results
        self.assertAlmostEqual(stats['mean'], 0, delta=0.1)
        self.assertAlmostEqual(stats['std'], 1, delta=0.1)
        self.assertIn('skewness', stats)
        self.assertIn('kurtosis', stats)
        self.assertIn('iqr', stats)

if __name__ == '__main__':
    unittest.main()
