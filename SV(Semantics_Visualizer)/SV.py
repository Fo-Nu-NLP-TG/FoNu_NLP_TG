"""
Semantics Visualizer (SV) - Enhanced Version

This module provides tools for visualizing word embeddings and semantic relationships
in transformer models during training and inference.

Features:
- Visualize pre-trained word embeddings (Word2Vec, GloVe, etc.)
- Extract and visualize embeddings from transformer models
- Track embedding changes during training
- Compare embeddings across different model versions
- Support for both PyTorch and TensorFlow models
- Interactive visualizations with dimensionality reduction (PCA, t-SNE, UMAP)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging

# Dimensionality reduction tools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Optional: for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SV')

class SemanticsVisualizer:
    """
    Main class for visualizing and analyzing semantic relationships in embeddings.
    """
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize the Semantics Visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for embeddings during training
        self.embedding_history = {}
        self.epoch_history = []
        
        # Default dimensionality reduction method
        self.dim_reduction_method = 'pca'
        
        logger.info(f"Initialized SemanticsVisualizer. Outputs will be saved to {output_dir}")
    
    def load_pretrained_embeddings(self, path: str, binary: bool = True):
        """
        Load pre-trained word embeddings from file.
        
        Args:
            path: Path to the embeddings file
            binary: Whether the file is in binary format
            
        Returns:
            The loaded embeddings model
        """
        try:
            from gensim.models import KeyedVectors
            logger.info(f"Loading pre-trained embeddings from {path}")
            self.pretrained_model = KeyedVectors.load_word2vec_format(path, binary=binary)
            logger.info(f"Loaded embeddings with vocabulary size: {len(self.pretrained_model.key_to_index)}")
            return self.pretrained_model
        except ImportError:
            logger.error("Gensim is required to load pre-trained embeddings. Install with: pip install gensim")
            raise
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def visualize_pretrained(self, words: List[str], method: str = 'pca', 
                            n_components: int = 2, title: str = None,
                            save: bool = True, show: bool = True):
        """
        Visualize pre-trained word embeddings.
        
        Args:
            words: List of words to visualize
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
            n_components: Number of dimensions to reduce to
            title: Plot title
            save: Whether to save the visualization
            show: Whether to display the visualization
            
        Returns:
            The figure and reduced vectors
        """
        if not hasattr(self, 'pretrained_model'):
            logger.error("No pre-trained embeddings loaded. Call load_pretrained_embeddings first.")
            return None, None
        
        # Filter words that are in the vocabulary
        valid_words = [w for w in words if w in self.pretrained_model.key_to_index]
        if len(valid_words) < len(words):
            logger.warning(f"Some words are not in the vocabulary: {set(words) - set(valid_words)}")
        
        if not valid_words:
            logger.error("None of the provided words are in the vocabulary.")
            return None, None
        
        # Get vectors for valid words
        vectors = [self.pretrained_model[word] for word in valid_words]
        
        # Reduce dimensions
        reduced_vectors = self._reduce_dimensions(vectors, method, n_components)
        
        # Create visualization
        fig = plt.figure(figsize=(10, 8))
        if n_components == 2:
            for i, word in enumerate(valid_words):
                plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
                plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
            
            plt.title(title or f'Semantic Relationships ({method.upper()})')
            plt.xlabel(f'Component 1')
            plt.ylabel(f'Component 2')
            
        elif n_components == 3 and PLOTLY_AVAILABLE:
            # Create 3D visualization with Plotly
            fig = go.Figure(data=[go.Scatter3d(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2],
                mode='markers+text',
                text=valid_words,
                marker=dict(size=10, opacity=0.8)
            )])
            fig.update_layout(title=title or f'Semantic Relationships in 3D ({method.upper()})')
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/pretrained_{method}_{timestamp}"
            
            if PLOTLY_AVAILABLE and n_components == 3:
                fig.write_html(f"{filename}.html")
            else:
                plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            
            logger.info(f"Saved visualization to {filename}")
        
        if show:
            if PLOTLY_AVAILABLE and n_components == 3:
                fig.show()
            else:
                plt.show()
        
        return fig, reduced_vectors
    
    def extract_embeddings_from_model(self, model, tokenizer, words: List[str], 
                                     layer: str = 'src_embed', device: str = 'cpu'):
        """
        Extract embeddings for specific words from a transformer model.
        
        Args:
            model: The transformer model
            tokenizer: Tokenizer for the model
            words: List of words to extract embeddings for
            layer: Which layer to extract embeddings from ('src_embed' or 'tgt_embed')
            device: Device to run the model on
            
        Returns:
            Dictionary mapping words to their embeddings
        """
        model.eval()
        model = model.to(device)
        
        embeddings = {}
        
        for word in words:
            try:
                # Tokenize the word
                token_ids = tokenizer.encode(word, out_type=int)
                token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
                
                # Get embeddings
                with torch.no_grad():
                    if layer == 'src_embed':
                        # Get source embeddings (without positional encoding)
                        if hasattr(model, 'src_embed') and isinstance(model.src_embed, torch.nn.Sequential):
                            # Access just the embedding layer (first part of Sequential)
                            embed_layer = model.src_embed[0]
                            embedding = embed_layer(token_tensor)
                        else:
                            logger.warning(f"Model structure doesn't match expected. Trying direct embedding.")
                            embedding = model.src_embed(token_tensor)
                    elif layer == 'tgt_embed':
                        # Get target embeddings (without positional encoding)
                        if hasattr(model, 'tgt_embed') and isinstance(model.tgt_embed, torch.nn.Sequential):
                            # Access just the embedding layer (first part of Sequential)
                            embed_layer = model.tgt_embed[0]
                            embedding = embed_layer(token_tensor)
                        else:
                            logger.warning(f"Model structure doesn't match expected. Trying direct embedding.")
                            embedding = model.tgt_embed(token_tensor)
                    else:
                        logger.error(f"Unknown layer: {layer}")
                        continue
                
                # Average embeddings if the word was split into multiple tokens
                mean_embedding = embedding.mean(dim=1).cpu().numpy()
                embeddings[word] = mean_embedding[0]  # Remove batch dimension
                
            except Exception as e:
                logger.error(f"Error extracting embedding for word '{word}': {e}")
        
        return embeddings
    
    def visualize_model_embeddings(self, embeddings: Dict[str, np.ndarray], 
                                  method: str = 'pca', n_components: int = 2,
                                  title: str = None, save: bool = True, 
                                  show: bool = True, filename_prefix: str = 'model'):
        """
        Visualize embeddings extracted from a model.
        
        Args:
            embeddings: Dictionary mapping words to their embeddings
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
            n_components: Number of dimensions to reduce to
            title: Plot title
            save: Whether to save the visualization
            show: Whether to display the visualization
            filename_prefix: Prefix for the saved file
            
        Returns:
            The figure and reduced vectors
        """
        if not embeddings:
            logger.error("No embeddings provided.")
            return None, None
        
        words = list(embeddings.keys())
        vectors = list(embeddings.values())
        
        # Reduce dimensions
        reduced_vectors = self._reduce_dimensions(vectors, method, n_components)
        
        # Create visualization
        fig = plt.figure(figsize=(10, 8))
        if n_components == 2:
            for i, word in enumerate(words):
                plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
                plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
            
            plt.title(title or f'Model Embeddings ({method.upper()})')
            plt.xlabel(f'Component 1')
            plt.ylabel(f'Component 2')
            
        elif n_components == 3 and PLOTLY_AVAILABLE:
            # Create 3D visualization with Plotly
            fig = go.Figure(data=[go.Scatter3d(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2],
                mode='markers+text',
                text=words,
                marker=dict(size=10, opacity=0.8)
            )])
            fig.update_layout(title=title or f'Model Embeddings in 3D ({method.upper()})')
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/{filename_prefix}_{method}_{timestamp}"
            
            if PLOTLY_AVAILABLE and n_components == 3:
                fig.write_html(f"{filename}.html")
            else:
                plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            
            logger.info(f"Saved visualization to {filename}")
        
        if show:
            if PLOTLY_AVAILABLE and n_components == 3:
                fig.show()
            else:
                plt.show()
        
        return fig, reduced_vectors
    
    def register_training_hook(self, model, tokenizer, words: List[str], 
                              layer: str = 'src_embed', device: str = 'cpu'):
        """
        Register a hook to track embeddings during training.
        
        Args:
            model: The transformer model
            tokenizer: Tokenizer for the model
            words: List of words to track
            layer: Which layer to extract embeddings from
            device: Device to run the model on
            
        Returns:
            Function to call after each epoch
        """
        def after_epoch_hook(epoch: int):
            """Hook to call after each epoch to track embeddings."""
            embeddings = self.extract_embeddings_from_model(
                model, tokenizer, words, layer, device
            )
            
            # Store embeddings for this epoch
            self.embedding_history[epoch] = embeddings
            self.epoch_history.append(epoch)
            
            logger.info(f"Tracked embeddings for epoch {epoch}")
            
            # Save the current state
            self._save_embedding_history()
            
            # Optionally create a visualization for this epoch
            self.visualize_model_embeddings(
                embeddings, 
                method=self.dim_reduction_method,
                title=f'Embeddings after Epoch {epoch}',
                filename_prefix=f'epoch_{epoch}',
                show=False  # Don't show during training
            )
        
        return after_epoch_hook
    
    def create_embedding_animation(self, output_file: str = None, method: str = 'pca'):
        """
        Create an animation showing how embeddings change during training.
        
        Args:
            output_file: Path to save the animation (GIF)
            method: Dimensionality reduction method
            
        Returns:
            The animation object
        """
        if not self.embedding_history:
            logger.error("No embedding history available. Train the model with a registered hook first.")
            return None
        
        # Get all words from the first epoch
        first_epoch = min(self.embedding_history.keys())
        words = list(self.embedding_history[first_epoch].keys())
        
        # Collect all vectors across epochs
        all_vectors = []
        for epoch in sorted(self.embedding_history.keys()):
            epoch_vectors = [self.embedding_history[epoch][word] for word in words]
            all_vectors.extend(epoch_vectors)
        
        # Fit dimensionality reduction on all vectors
        reducer = self._get_reducer(method, n_components=2)
        all_reduced = reducer.fit_transform(all_vectors)
        
        # Split back by epoch
        epochs = sorted(self.embedding_history.keys())
        vectors_by_epoch = {}
        idx = 0
        for epoch in epochs:
            vectors_by_epoch[epoch] = all_reduced[idx:idx+len(words)]
            idx += len(words)
        
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            epoch = epochs[frame]
            reduced_vectors = vectors_by_epoch[epoch]
            
            for i, word in enumerate(words):
                ax.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
                ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
            
            ax.set_title(f'Embeddings Evolution - Epoch {epoch}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            
            # Set consistent axis limits
            all_x = [v[0] for e in epochs for v in vectors_by_epoch[e]]
            all_y = [v[1] for e in epochs for v in vectors_by_epoch[e]]
            margin = 0.1
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_margin = (x_max - x_min) * margin
            y_margin = (y_max - y_min) * margin
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        animation = FuncAnimation(fig, update, frames=len(epochs), interval=500)
        
        if output_file:
            animation.save(output_file, writer='pillow', fps=1)
            logger.info(f"Saved animation to {output_file}")
        
        plt.close(fig)
        return animation
    
    def compare_embeddings(self, embeddings1: Dict[str, np.ndarray], 
                          embeddings2: Dict[str, np.ndarray],
                          labels: Tuple[str, str] = ('Model 1', 'Model 2'),
                          method: str = 'pca', title: str = None,
                          save: bool = True, show: bool = True):
        """
        Compare embeddings from two different sources.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: Labels for the two embedding sets
            method: Dimensionality reduction method
            title: Plot title
            save: Whether to save the visualization
            show: Whether to display the visualization
            
        Returns:
            The figure
        """
        # Find common words
        common_words = set(embeddings1.keys()) & set(embeddings2.keys())
        if not common_words:
            logger.error("No common words between the two embedding sets.")
            return None
        
        common_words = sorted(list(common_words))
        logger.info(f"Comparing {len(common_words)} common words")
        
        # Get vectors for common words
        vectors1 = [embeddings1[word] for word in common_words]
        vectors2 = [embeddings2[word] for word in common_words]
        
        # Combine vectors for dimensionality reduction
        all_vectors = vectors1 + vectors2
        
        # Reduce dimensions
        reduced_vectors = self._reduce_dimensions(all_vectors, method, n_components=2)
        
        # Split back into two sets
        reduced1 = reduced_vectors[:len(common_words)]
        reduced2 = reduced_vectors[len(common_words):]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot first set
        for i, word in enumerate(common_words):
            ax.scatter(reduced1[i, 0], reduced1[i, 1], color='blue', alpha=0.7)
            ax.annotate(f"{word} ({labels[0]})", (reduced1[i, 0], reduced1[i, 1]), 
                       color='blue', alpha=0.7)
        
        # Plot second set
        for i, word in enumerate(common_words):
            ax.scatter(reduced2[i, 0], reduced2[i, 1], color='red', alpha=0.7)
            ax.annotate(f"{word} ({labels[1]})", (reduced2[i, 0], reduced2[i, 1]), 
                       color='red', alpha=0.7)
        
        # Draw lines connecting the same words
        for i in range(len(common_words)):
            ax.plot([reduced1[i, 0], reduced2[i, 0]], [reduced1[i, 1], reduced2[i, 1]], 
                   'k-', alpha=0.3)
        
        ax.set_title(title or f'Embedding Comparison ({method.upper()})')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label=labels[0]),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label=labels[1])
        ]
        ax.legend(handles=legend_elements)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/comparison_{method}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison to {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def _reduce_dimensions(self, vectors: List[np.ndarray], method: str = 'pca', 
                          n_components: int = 2) -> np.ndarray:
        """
        Reduce the dimensionality of vectors.
        
        Args:
            vectors: List of vectors to reduce
            method: Dimensionality reduction method
            n_components: Number of dimensions to reduce to
            
        Returns:
            Reduced vectors
        """
        reducer = self._get_reducer(method, n_components)
        return reducer.fit_transform(vectors)
    
    def _get_reducer(self, method: str, n_components: int):
        """Get the appropriate dimensionality reducer."""
        if method.lower() == 'pca':
            return PCA(n_components=n_components)
        elif method.lower() == 'tsne':
            return TSNE(n_components=n_components, perplexity=min(30, max(5, len(vectors) - 1)))
        elif method.lower() == 'umap':
            if not UMAP_AVAILABLE:
                logger.warning("UMAP not available. Falling back to PCA. Install with: pip install umap-learn")
                return PCA(n_components=n_components)
            return umap.UMAP(n_components=n_components)
        else:
            logger.warning(f"Unknown method: {method}. Falling back to PCA.")
            return PCA(n_components=n_components)
    
    def _save_embedding_history(self):
        """Save the current embedding history to disk."""
        history_file = os.path.join(self.output_dir, 'embedding_history.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump({
                'embedding_history': self.embedding_history,
                'epoch_history': self.epoch_history
            }, f)
        logger.info(f"Saved embedding history to {history_file}")
    
    def load_embedding_history(self, file_path: str):
        """
        Load embedding history from a file.
        
        Args:
            file_path: Path to the history file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.embedding_history = data['embedding_history']
                self.epoch_history = data['epoch_history']
            logger.info(f"Loaded embedding history from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding history: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create visualizer
    sv = SemanticsVisualizer()
    
    # Try to load pre-trained embeddings if available
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'GoogleNews-vectors-negative300.bin')
        if os.path.exists(model_path):
            sv.load_pretrained_embeddings(model_path)
            
            # Visualize some words
            words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'boy', 'girl']
            sv.visualize_pretrained(words, method='pca')
        else:
            print(f"Pre-trained embeddings not found at {model_path}")
            print("Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing")
    except Exception as e:
        print(f"Error loading pre-trained embeddings: {e}")
        print("You can still use the visualizer with your transformer model.")
