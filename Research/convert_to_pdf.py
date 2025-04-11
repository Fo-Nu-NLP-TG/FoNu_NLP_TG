#!/usr/bin/env python3
"""
Script to convert Markdown research reports to PDF format.
Requires pandoc and a LaTeX distribution to be installed.

Usage:
    python convert_to_pdf.py [input_file] [output_file]

Example:
    python convert_to_pdf.py ewe_english_transformer_research_updated.md ewe_english_transformer_research.pdf
"""

import os
import sys
import subprocess
import argparse

def convert_markdown_to_pdf(input_file, output_file=None):
    """
    Convert a Markdown file to PDF using pandoc.
    
    Args:
        input_file: Path to the input Markdown file
        output_file: Path to the output PDF file (optional)
    
    Returns:
        Path to the generated PDF file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # If output file is not specified, use the input filename with .pdf extension
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".pdf"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build the pandoc command
    cmd = [
        "pandoc",
        input_file,
        "-o", output_file,
        "--pdf-engine=xelatex",
        "--variable", "geometry:margin=1in",
        "--variable", "fontsize=11pt",
        "--toc",  # Table of contents
        "--highlight-style=tango"  # Code highlighting style
    ]
    
    try:
        # Run the pandoc command
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {input_file} to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}")
        return None
    except FileNotFoundError:
        print("Error: pandoc not found. Please install pandoc and a LaTeX distribution.")
        print("Installation instructions: https://pandoc.org/installing.html")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown research reports to PDF")
    parser.add_argument("input_file", help="Path to the input Markdown file")
    parser.add_argument("output_file", nargs="?", help="Path to the output PDF file (optional)")
    
    args = parser.parse_args()
    
    convert_markdown_to_pdf(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
