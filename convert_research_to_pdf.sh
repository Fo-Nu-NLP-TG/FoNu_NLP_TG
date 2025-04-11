#!/bin/bash
# Script to convert the research report to PDF

# Exit on error
set -e

# Display commands
set -x

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    source myenv/bin/activate
fi

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Pandoc is not installed. Please install it first."
    echo "Installation instructions: https://pandoc.org/installing.html"
    exit 1
fi

# Convert the research report to PDF
pandoc Research/ewe_english_transformer_research_updated.md \
    -o Research/ewe_english_transformer_research.pdf \
    --pdf-engine=xelatex \
    --variable geometry:margin=1in \
    --variable fontsize=11pt \
    --toc \
    --highlight-style=tango

echo "Research report converted to PDF successfully!"
