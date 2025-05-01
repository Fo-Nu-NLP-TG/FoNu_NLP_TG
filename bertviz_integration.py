"""
BERTViz Integration for FoNu_NLP_TG

This module provides functions to visualize attention in the transformer model
using the BERTViz library.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Import BERTViz
from bertviz import head_view, model_view, neuron_view

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bertviz_integration')

def load_model_and_tokenizers(model_path, device='cpu'):
    """
    Load a trained transformer model and its tokenizers.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model, (src_tokenizer, tgt_tokenizer)
    """
    try:
        # Add parent directory to path to import transformer modules
        sys.path.append(str(Path(__file__).parent))
        
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
                data_dir = Path(__file__).parent / "data" / "processed"
                src_tokenizer_path = list(data_dir.glob("*.src.model"))
                tgt_tokenizer_path = list(data_dir.glob("*.tgt.model"))
                
                if src_tokenizer_path and tgt_tokenizer_path:
                    src_tokenizer.load(str(src_tokenizer_path[0]))
                    tgt_tokenizer.load(str(tgt_tokenizer_path[0]))
                    logger.info(f"Loaded tokenizers from {data_dir}")
                else:
                    logger.error("Tokenizer models not found")
                    return None, None
        except ImportError:
            logger.error("SentencePiece not installed. Install with: pip install sentencepiece")
            return None, None
        
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
            return None, None
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        
        return model, (src_tokenizer, tgt_tokenizer)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def prepare_inputs_for_bertviz(src_text, tgt_text, src_tokenizer, tgt_tokenizer, device='cpu'):
    """
    Prepare inputs for BERTViz visualization.
    
    Args:
        src_text: Source text (e.g., Ewe)
        tgt_text: Target text (e.g., English)
        src_tokenizer: Source tokenizer
        tgt_tokenizer: Target tokenizer
        device: Device to run the model on
        
    Returns:
        Tokenized inputs ready for the model
    """
    # Tokenize source and target text
    src_tokens = src_tokenizer.encode(src_text, out_type=int)
    tgt_tokens = tgt_tokenizer.encode(tgt_text, out_type=int)
    
    # Convert to tensors
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
    
    # Create masks
    from Transformew2.model import create_masks
    src_mask, tgt_mask = create_masks(src_tensor, tgt_tensor)
    
    # Get token strings for visualization
    src_token_strings = [src_tokenizer.id_to_piece(token_id) for token_id in src_tokens]
    tgt_token_strings = [tgt_tokenizer.id_to_piece(token_id) for token_id in tgt_tokens]
    
    return {
        'src_tensor': src_tensor,
        'tgt_tensor': tgt_tensor,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_token_strings': src_token_strings,
        'tgt_token_strings': tgt_token_strings
    }

def collect_attention_weights(model, inputs):
    """
    Collect attention weights from the model.
    
    Args:
        model: The transformer model
        inputs: Prepared inputs from prepare_inputs_for_bertviz
        
    Returns:
        Dictionary of attention weights
    """
    # Get inputs
    src_tensor = inputs['src_tensor']
    tgt_tensor = inputs['tgt_tensor']
    src_mask = inputs['src_mask']
    tgt_mask = inputs['tgt_mask']
    
    # Forward pass with attention collection
    with torch.no_grad():
        # Encode source
        src_embedded = model.src_embed(src_tensor)
        encoder_output = model.encoder(src_embedded, src_mask)
        
        # Decode target
        tgt_embedded = model.tgt_embed(tgt_tensor)
        decoder_output = model.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # Collect attention weights
        attention_weights = {
            'encoder_self_attention': [],
            'decoder_self_attention': [],
            'encoder_decoder_attention': []
        }
        
        # Collect encoder self-attention
        for i, layer in enumerate(model.encoder.layers):
            attention_weights['encoder_self_attention'].append(
                layer.self_attn.attn.cpu().numpy()
            )
        
        # Collect decoder self-attention and encoder-decoder attention
        for i, layer in enumerate(model.decoder.layers):
            attention_weights['decoder_self_attention'].append(
                layer.self_attn.attn.cpu().numpy()
            )
            attention_weights['encoder_decoder_attention'].append(
                layer.src_attn.attn.cpu().numpy()
            )
    
    return attention_weights

def visualize_head_view(attention_weights, tokens, output_path=None):
    """
    Visualize attention using BERTViz head view.
    
    Args:
        attention_weights: Attention weights from collect_attention_weights
        tokens: Token strings
        output_path: Path to save the visualization HTML
    """
    # Format attention for BERTViz
    # BERTViz expects attention of shape [batch_size, num_heads, seq_len, seq_len]
    encoder_attention = attention_weights['encoder_self_attention']
    
    # Create HTML visualization
    html = head_view(
        attention=encoder_attention,
        tokens=tokens
    )
    
    # Save or display
    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)
        logger.info(f"Saved head view visualization to {output_path}")
    
    return html

def visualize_model_view(attention_weights, tokens, output_path=None):
    """
    Visualize attention using BERTViz model view.
    
    Args:
        attention_weights: Attention weights from collect_attention_weights
        tokens: Token strings
        output_path: Path to save the visualization HTML
    """
    # Format attention for BERTViz
    encoder_attention = attention_weights['encoder_self_attention']
    
    # Create HTML visualization
    html = model_view(
        attention=encoder_attention,
        tokens=tokens
    )
    
    # Save or display
    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)
        logger.info(f"Saved model view visualization to {output_path}")
    
    return html

def visualize_attention_for_text(model_path, src_text, tgt_text=None, output_dir='bertviz_visualizations', device='cpu'):
    """
    Visualize attention for a given text pair.
    
    Args:
        model_path: Path to the model checkpoint
        src_text: Source text (e.g., Ewe)
        tgt_text: Target text (e.g., English). If None, only source attention is visualized.
        output_dir: Directory to save visualizations
        device: Device to run the model on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizers
    model, tokenizers = load_model_and_tokenizers(model_path, device)
    if model is None or tokenizers is None:
        logger.error("Failed to load model or tokenizers")
        return
    
    src_tokenizer, tgt_tokenizer = tokenizers
    
    # If target text is not provided, use an empty string
    if tgt_text is None:
        tgt_text = ""
    
    # Prepare inputs
    inputs = prepare_inputs_for_bertviz(src_text, tgt_text, src_tokenizer, tgt_tokenizer, device)
    
    # Collect attention weights
    attention_weights = collect_attention_weights(model, inputs)
    
    # Visualize attention
    src_tokens = inputs['src_token_strings']
    
    # Create head view visualization
    head_view_path = os.path.join(output_dir, 'head_view.html')
    visualize_head_view(attention_weights, src_tokens, head_view_path)
    
    # Create model view visualization
    model_view_path = os.path.join(output_dir, 'model_view.html')
    visualize_model_view(attention_weights, src_tokens, model_view_path)
    
    logger.info(f"Created visualizations in {output_dir}")
    
    return {
        'head_view_path': head_view_path,
        'model_view_path': model_view_path
    }

def main():
    parser = argparse.ArgumentParser(description='Visualize transformer attention with BERTViz')
    parser.add_argument('--model-path', type=str, default='models/transformer_ewe_english_final.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--src-text', type=str, required=True,
                        help='Source text to visualize attention for')
    parser.add_argument('--tgt-text', type=str, default=None,
                        help='Target text (optional)')
    parser.add_argument('--output-dir', type=str, default='bertviz_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the model on')
    
    args = parser.parse_args()
    
    # Visualize attention
    visualize_attention_for_text(
        args.model_path,
        args.src_text,
        args.tgt_text,
        args.output_dir,
        args.device
    )

if __name__ == '__main__':
    main()
