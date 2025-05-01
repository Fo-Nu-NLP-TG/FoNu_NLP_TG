"""
Semantics Visualizer with BERTViz Integration

This module extends the Semantics Visualizer (SV) with BERTViz capabilities,
allowing for both embedding visualization and attention visualization.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Import SV
from SV import SemanticsVisualizer

# Import BERTViz integration
from bertviz_integration import (
    load_model_and_tokenizers,
    prepare_inputs_for_bertviz,
    collect_attention_weights,
    visualize_head_view,
    visualize_model_view
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sv_bertviz')

class EnhancedSemanticsVisualizer(SemanticsVisualizer):
    """
    Enhanced Semantics Visualizer with BERTViz integration.
    """
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize the Enhanced Semantics Visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        super().__init__(output_dir=output_dir)
        self.bertviz_dir = os.path.join(output_dir, 'bertviz')
        os.makedirs(self.bertviz_dir, exist_ok=True)
        
        logger.info(f"Initialized EnhancedSemanticsVisualizer. BERTViz outputs will be saved to {self.bertviz_dir}")
    
    def visualize_attention(self, model, src_text, tgt_text, tokenizers, device='cpu'):
        """
        Visualize attention for a given text pair using BERTViz.
        
        Args:
            model: The transformer model
            src_text: Source text (e.g., Ewe)
            tgt_text: Target text (e.g., English)
            tokenizers: Tuple of (source_tokenizer, target_tokenizer)
            device: Device to run the model on
            
        Returns:
            Dictionary with paths to visualization files
        """
        src_tokenizer, tgt_tokenizer = tokenizers
        
        # Prepare inputs
        inputs = prepare_inputs_for_bertviz(src_text, tgt_text, src_tokenizer, tgt_tokenizer, device)
        
        # Collect attention weights
        attention_weights = collect_attention_weights(model, inputs)
        
        # Visualize attention
        src_tokens = inputs['src_token_strings']
        
        # Create head view visualization
        head_view_path = os.path.join(self.bertviz_dir, 'head_view.html')
        visualize_head_view(attention_weights, src_tokens, head_view_path)
        
        # Create model view visualization
        model_view_path = os.path.join(self.bertviz_dir, 'model_view.html')
        visualize_model_view(attention_weights, src_tokens, model_view_path)
        
        logger.info(f"Created attention visualizations in {self.bertviz_dir}")
        
        return {
            'head_view_path': head_view_path,
            'model_view_path': model_view_path
        }
    
    def analyze_model(self, model_path, words, src_text=None, tgt_text=None, device='cpu'):
        """
        Comprehensive analysis of a model, including embeddings and attention.
        
        Args:
            model_path: Path to the model checkpoint
            words: List of words to visualize embeddings for
            src_text: Source text for attention visualization (optional)
            tgt_text: Target text for attention visualization (optional)
            device: Device to run the model on
            
        Returns:
            Dictionary with paths to visualization files
        """
        # Load model and tokenizers
        model, tokenizers = load_model_and_tokenizers(model_path, device)
        if model is None or tokenizers is None:
            logger.error("Failed to load model or tokenizers")
            return None
        
        src_tokenizer, tgt_tokenizer = tokenizers
        
        # Visualize embeddings
        logger.info("Extracting and visualizing embeddings...")
        src_embeddings = self.extract_embeddings_from_model(
            model, src_tokenizer, words, layer='src_embed', device=device
        )
        
        tgt_embeddings = self.extract_embeddings_from_model(
            model, tgt_tokenizer, words, layer='tgt_embed', device=device
        )
        
        embedding_viz = {}
        
        if src_embeddings:
            src_pca_fig, _ = self.visualize_model_embeddings(
                src_embeddings,
                method='pca',
                title='Source Embeddings (PCA)',
                filename_prefix='source_embeddings'
            )
            embedding_viz['src_pca'] = src_pca_fig
            
            src_tsne_fig, _ = self.visualize_model_embeddings(
                src_embeddings,
                method='tsne',
                title='Source Embeddings (t-SNE)',
                filename_prefix='source_embeddings'
            )
            embedding_viz['src_tsne'] = src_tsne_fig
        
        if tgt_embeddings:
            tgt_pca_fig, _ = self.visualize_model_embeddings(
                tgt_embeddings,
                method='pca',
                title='Target Embeddings (PCA)',
                filename_prefix='target_embeddings'
            )
            embedding_viz['tgt_pca'] = tgt_pca_fig
            
            tgt_tsne_fig, _ = self.visualize_model_embeddings(
                tgt_embeddings,
                method='tsne',
                title='Target Embeddings (t-SNE)',
                filename_prefix='target_embeddings'
            )
            embedding_viz['tgt_tsne'] = tgt_tsne_fig
        
        if src_embeddings and tgt_embeddings:
            compare_fig = self.compare_embeddings(
                src_embeddings,
                tgt_embeddings,
                labels=('Source', 'Target'),
                method='pca',
                title='Source vs Target Embeddings'
            )
            embedding_viz['compare'] = compare_fig
        
        # Visualize attention if text is provided
        attention_viz = None
        if src_text:
            logger.info("Visualizing attention patterns...")
            attention_viz = self.visualize_attention(
                model, src_text, tgt_text, tokenizers, device
            )
        
        return {
            'embedding_viz': embedding_viz,
            'attention_viz': attention_viz
        }

def main():
    parser = argparse.ArgumentParser(description='Enhanced Semantics Visualizer with BERTViz')
    parser.add_argument('--model-path', type=str, default='models/transformer_ewe_english_final.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--words', type=str, nargs='+', 
                        default=['hello', 'world', 'language', 'model', 'transformer', 'attention'],
                        help='Words to visualize embeddings for')
    parser.add_argument('--src-text', type=str, default=None,
                        help='Source text for attention visualization (optional)')
    parser.add_argument('--tgt-text', type=str, default=None,
                        help='Target text for attention visualization (optional)')
    parser.add_argument('--output-dir', type=str, default='enhanced_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the model on')
    parser.add_argument('--open-browser', action='store_true',
                        help='Open visualizations in browser')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EnhancedSemanticsVisualizer(output_dir=args.output_dir)
    
    # Analyze model
    results = visualizer.analyze_model(
        args.model_path,
        args.words,
        args.src_text,
        args.tgt_text,
        args.device
    )
    
    if results and args.open_browser and results['attention_viz']:
        # Open attention visualizations in browser
        import webbrowser
        for path in results['attention_viz'].values():
            webbrowser.open(f'file://{os.path.abspath(path)}')
        
        print("\nAttention visualizations opened in your browser.")
    
    print(f"\nAll visualization files saved to {args.output_dir}/")
    print("You can open the HTML files in your browser to explore the visualizations.")

if __name__ == '__main__':
    main()
