"""
Train Transformer with Semantics Visualizer Integration

This script extends the transformer training process to integrate with the
Semantics Visualizer, allowing visualization of embeddings during training.
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Import the SemanticsVisualizer integration
from transformer_sv_integration import register_training_hooks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_with_sv')

def train_transformer_with_sv(args):
    """
    Train a transformer model with Semantics Visualizer integration.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Add parent directory to path to import transformer modules
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
        
        # Import training utilities
        if is_transformew2:
            try:
                from Transformew2.train import train_model, load_tokenizers, prepare_data
            except ImportError:
                logger.error("Could not import training utilities from Transformew2")
                return
        else:
            try:
                from Transformew1.train_transformer import train_epoch, load_tokenizers, prepare_data
            except ImportError:
                logger.error("Could not import training utilities from Transformew1")
                return
        
        # Load tokenizers
        src_tokenizer, tgt_tokenizer = load_tokenizers(args.src_lang, args.tgt_lang, args.data_dir)
        
        # Get vocabulary sizes
        src_vocab_size = src_tokenizer.get_piece_size()
        tgt_vocab_size = tgt_tokenizer.get_piece_size()
        
        logger.info(f"Source vocabulary size: {src_vocab_size}")
        logger.info(f"Target vocabulary size: {tgt_vocab_size}")
        
        # Create model
        model = make_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            N=args.layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            h=args.heads,
            dropout=args.dropout
        )
        
        # Move model to device
        device = torch.device(args.device)
        model = model.to(device)
        
        # Prepare data
        train_dataloader, valid_dataloader = prepare_data(
            src_tokenizer, tgt_tokenizer, 
            args.data_dir, args.batch_size
        )
        
        # Define optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        
        # Register Semantics Visualizer hooks
        hooks = register_training_hooks(
            model, 
            (src_tokenizer, tgt_tokenizer), 
            args.track_words, 
            args.sv_output_dir, 
            device
        )
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            
            # Train for one epoch
            if is_transformew2:
                train_loss = train_model(
                    model, train_dataloader, optimizer, device, 
                    tgt_vocab_size, epoch
                )
            else:
                train_loss = train_epoch(
                    model, train_dataloader, optimizer, device, 
                    tgt_vocab_size
                )
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                args.save_dir, 
                f"transformer_{args.src_lang}_{args.tgt_lang}_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'src_vocab_size': src_vocab_size,
                'tgt_vocab_size': tgt_vocab_size,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Call Semantics Visualizer hooks
            hooks['src_hook'](epoch + 1)
            hooks['tgt_hook'](epoch + 1)
        
        # Create animations after training
        hooks['create_animations']()
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Train transformer with Semantics Visualizer')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory with processed data')
    parser.add_argument('--src-lang', type=str, default='ewe', help='Source language')
    parser.add_argument('--tgt-lang', type=str, default='english', help='Target language')
    
    # Model parameters
    parser.add_argument('--layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d-ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to train on')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    # Semantics Visualizer parameters
    parser.add_argument('--sv-output-dir', type=str, default='visualizations', 
                        help='Directory to save visualizations')
    parser.add_argument('--track-words', type=str, nargs='+', 
                        default=['hello', 'world', 'language', 'model', 'transformer', 
                                'attention', 'neural', 'network', 'translation'],
                        help='Words to track during training')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sv_output_dir, exist_ok=True)
    
    # Train the model
    train_transformer_with_sv(args)

if __name__ == '__main__':
    main()
