#!/usr/bin/env python3
"""
Launcher script for Colab kernels.
This script simply imports and runs the main function from the transformer CLI script.
"""

import os
import sys
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a transformer model for translation")
    parser.add_argument("--data_dir", default="./data/processed", help="Directory with processed data")
    parser.add_argument("--src_lang", default="ewe", help="Source language")
    parser.add_argument("--tgt_lang", default="english", help="Target language")
    parser.add_argument("--tokenizer_type", choices=["sentencepiece", "huggingface"], default="sentencepiece", 
                        help="Type of tokenizer to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--save_dir", default="./models", help="Directory to save models")
    
    # Filter out Jupyter/Colab specific arguments
    filtered_args = []
    for arg in sys.argv[1:]:
        if not arg.startswith('-f') and not 'kernel' in arg and not 'json' in arg:
            filtered_args.append(arg)
    
    # Parse the filtered arguments
    args = parser.parse_args(filtered_args)
    
    # Import and run the main function from the CLI script
    try:
        from Attention_Is_All_You_Need.train_transformer_cli import main as train_main
        train_main()
    except ImportError as e:
        print(f"Error importing transformer training module: {e}")
        print("Make sure you're running this script from the project root directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
