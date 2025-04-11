#!/bin/bash
# Script to build and deploy documentation to GitHub Pages

# Exit on error
set -e

# Display commands
set -x

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    source myenv/bin/activate
fi

# Install required packages if not already installed
pip install mkdocs mkdocs-material mkdocstrings pymdown-extensions mkdocs-git-revision-date-localized-plugin

# Build the documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy --force

echo "Documentation deployed successfully!"
