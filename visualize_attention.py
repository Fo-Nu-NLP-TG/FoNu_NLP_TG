#!/usr/bin/env python3
"""
Visualize Attention with BERTViz

This script provides a simple interface to visualize attention in the transformer model
using BERTViz.
"""

import os
import argparse
import webbrowser
from pathlib import Path
from bertviz_integration import visualize_attention_for_text

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
    parser.add_argument('--open-browser', action='store_true',
                        help='Open visualizations in browser')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Visualizing attention for: {args.src_text}")
    if args.tgt_text:
        print(f"Target text: {args.tgt_text}")
    
    # Visualize attention
    viz_paths = visualize_attention_for_text(
        args.model_path,
        args.src_text,
        args.tgt_text,
        args.output_dir,
        args.device
    )
    
    if viz_paths and args.open_browser:
        # Open visualizations in browser
        for path in viz_paths.values():
            webbrowser.open(f'file://{os.path.abspath(path)}')
        
        print("\nVisualizations opened in your browser.")
    
    print(f"\nVisualization files saved to {args.output_dir}/")
    print("You can open these HTML files in your browser to explore the attention patterns.")

if __name__ == '__main__':
    main()
