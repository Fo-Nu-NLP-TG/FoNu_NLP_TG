"""
Transformer Semantics Visualizer Integration

This script demonstrates how to integrate the Semantics Visualizer with
the transformer model during training to visualize how embeddings evolve.
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Import the SemanticsVisualizer
from SV import SemanticsVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transformer_sv_integration')

def load_model_and_tokenizer(model_path, device='cpu'):
    """
    Load a trained transformer model and its tokenizer.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model, tokenizer, src_vocab_size, tgt_vocab_size
    """
    try:
        # Import necessary modules
        sys.path.append(str(Path(__file__).parent.parent))
        
        # Try to import from Transformew2 first
        try:
            from Transformew2.model import make_model, TransformEw2
            logger.info("Using Transformew2 model")
            is_transformew2 = True
        except ImportError:
            # Fall back to Transformew1
            from Transformew1.train_transformer import make_model
            logger.info("Using Transformew1 model")
            is_transformew2 = False
        
        # Load tokenizer
        try:
            import sentencepiece as spm
            src_tokenizer = spm.SentencePieceProcessor()
            tgt_tokenizer = spm.SentencePieceProcessor()
            
            # Find tokenizer models
            model_dir = Path(model_path).parent
            src_tokenizer_path = list(model_dir.glob("*.src.model"))
            tgt_tokenizer_path = list(model_dir.glob("*.tgt.model"))
            
            if src_tokenizer_path and tgt_tokenizer_path:
                src_tokenizer.load(str(src_tokenizer_path[0]))
                tgt_tokenizer.load(str(tgt_tokenizer_path[0]))
                logger.info(f"Loaded tokenizers from {model_dir}")
            else:
                logger.warning("Tokenizer models not found. Looking in data directory...")
                data_dir = Path(__file__).parent.parent / "data" / "processed"
                src_tokenizer_path = list(data_dir.glob("*.src.model"))
                tgt_tokenizer_path = list(data_dir.glob("*.tgt.model"))
                
                if src_tokenizer_path and tgt_tokenizer_path:
                    src_tokenizer.load(str(src_tokenizer_path[0]))
                    tgt_tokenizer.load(str(tgt_tokenizer_path[0]))
                    logger.info(f"Loaded tokenizers from {data_dir}")
                else:
                    logger.error("Tokenizer models not found")
                    return None, None, None, None
        except ImportError:
            logger.error("SentencePiece not installed. Install with: pip install sentencepiece")
            return None, None, None, None
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get vocabulary sizes
        if 'src_vocab_size' in checkpoint and 'tgt_vocab_size' in checkpoint:
            src_vocab_size = checkpoint['src_vocab_size']
            tgt_vocab_size = checkpoint['tgt_vocab_size']
        elif 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, 'src_vocab_size') and hasattr(args, 'tgt_vocab_size'):
                src_vocab_size = args.src_vocab_size
                tgt_vocab_size = args.tgt_vocab_size
            else:
                # Try to get from tokenizers
                src_vocab_size = src_tokenizer.get_piece_size()
                tgt_vocab_size = tgt_tokenizer.get_piece_size()
        else:
            # Try to get from tokenizers
            src_vocab_size = src_tokenizer.get_piece_size()
            tgt_vocab_size = tgt_tokenizer.get_piece_size()
        
        logger.info(f"Source vocabulary size: {src_vocab_size}")
        logger.info(f"Target vocabulary size: {tgt_vocab_size}")
        
        # Create model
        if is_transformew2:
            model = make_model(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                N=6,  # Default number of layers
                d_model=512,  # Default model dimension
                d_ff=2048,  # Default feed-forward dimension
                h=8,  # Default number of attention heads
                dropout=0.1  # Default dropout
            )
        else:
            model = make_model(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size
            )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            logger.error("Model state not found in checkpoint")
            return None, None, None, None
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        
        return model, (src_tokenizer, tgt_tokenizer), src_vocab_size, tgt_vocab_size
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def visualize_model_embeddings(model_path, words, output_dir='visualizations', device='cpu'):
    """
    Visualize embeddings from a trained transformer model.
    
    Args:
        model_path: Path to the model checkpoint
        words: List of words to visualize
        output_dir: Directory to save visualizations
        device: Device to run the model on
    """
    # Load model and tokenizer
    model, tokenizers, src_vocab_size, tgt_vocab_size = load_model_and_tokenizer(model_path, device)
    
    if model is None or tokenizers is None:
        logger.error("Failed to load model or tokenizer")
        return
    
    src_tokenizer, tgt_tokenizer = tokenizers
    
    # Create visualizer
    sv = SemanticsVisualizer(output_dir=output_dir)
    
    # Extract and visualize source embeddings
    src_embeddings = sv.extract_embeddings_from_model(
        model, src_tokenizer, words, layer='src_embed', device=device
    )
    
    if src_embeddings:
        logger.info(f"Extracted source embeddings for {len(src_embeddings)} words")
        sv.visualize_model_embeddings(
            src_embeddings,
            method='pca',
            title='Source Embeddings (PCA)',
            filename_prefix='source_embeddings'
        )
        
        # Try t-SNE as well
        sv.visualize_model_embeddings(
            src_embeddings,
            method='tsne',
            title='Source Embeddings (t-SNE)',
            filename_prefix='source_embeddings'
        )
    else:
        logger.warning("No source embeddings extracted")
    
    # Extract and visualize target embeddings
    tgt_embeddings = sv.extract_embeddings_from_model(
        model, tgt_tokenizer, words, layer='tgt_embed', device=device
    )
    
    if tgt_embeddings:
        logger.info(f"Extracted target embeddings for {len(tgt_embeddings)} words")
        sv.visualize_model_embeddings(
            tgt_embeddings,
            method='pca',
            title='Target Embeddings (PCA)',
            filename_prefix='target_embeddings'
        )
        
        # Try t-SNE as well
        sv.visualize_model_embeddings(
            tgt_embeddings,
            method='tsne',
            title='Target Embeddings (t-SNE)',
            filename_prefix='target_embeddings'
        )
    else:
        logger.warning("No target embeddings extracted")
    
    # Compare source and target embeddings if both are available
    if src_embeddings and tgt_embeddings:
        sv.compare_embeddings(
            src_embeddings,
            tgt_embeddings,
            labels=('Source', 'Target'),
            method='pca',
            title='Source vs Target Embeddings'
        )

def register_training_hooks(model, tokenizers, words, output_dir='visualizations', device='cpu'):
    """
    Register hooks to track embeddings during training.
    
    Args:
        model: The transformer model
        tokenizers: Tuple of (source_tokenizer, target_tokenizer)
        words: List of words to track
        output_dir: Directory to save visualizations
        device: Device to run the model on
        
    Returns:
        Dictionary of hooks to call during training
    """
    src_tokenizer, tgt_tokenizer = tokenizers
    
    # Create visualizer
    sv = SemanticsVisualizer(output_dir=output_dir)
    
    # Register hooks for source and target embeddings
    src_hook = sv.register_training_hook(
        model, src_tokenizer, words, layer='src_embed', device=device
    )
    
    tgt_hook = sv.register_training_hook(
        model, tgt_tokenizer, words, layer='tgt_embed', device=device
    )
    
    # Function to call after training to create animations
    def create_animations():
        src_animation_path = os.path.join(output_dir, 'source_embeddings_evolution.gif')
        sv.create_embedding_animation(output_file=src_animation_path, method='pca')
        
        tgt_animation_path = os.path.join(output_dir, 'target_embeddings_evolution.gif')
        sv.create_embedding_animation(output_file=tgt_animation_path, method='pca')
        
        logger.info(f"Created embedding animations at {output_dir}")
    
    return {
        'src_hook': src_hook,
        'tgt_hook': tgt_hook,
        'create_animations': create_animations
    }

def integrate_with_training_loop(training_function, model, tokenizers, words, 
                               output_dir='visualizations', device='cpu'):
    """
    Integrate the Semantics Visualizer with a training loop.
    
    Args:
        training_function: Function that trains the model
        model: The transformer model
        tokenizers: Tuple of (source_tokenizer, target_tokenizer)
        words: List of words to track
        output_dir: Directory to save visualizations
        device: Device to run the model on
    """
    # Register hooks
    hooks = register_training_hooks(model, tokenizers, words, output_dir, device)
    
    # Define a callback to call after each epoch
    def after_epoch_callback(epoch):
        hooks['src_hook'](epoch)
        hooks['tgt_hook'](epoch)
    
    # Train the model with the callback
    training_function(model, after_epoch_callback=after_epoch_callback)
    
    # Create animations after training
    hooks['create_animations']()

def main():
    parser = argparse.ArgumentParser(description='Visualize transformer embeddings')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--words', type=str, nargs='+', default=['hello', 'world', 'language', 'model', 'transformer', 'attention', 'neural', 'network'],
                        help='Words to visualize')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize embeddings
    visualize_model_embeddings(args.model_path, args.words, args.output_dir, args.device)

if __name__ == '__main__':
    main()
