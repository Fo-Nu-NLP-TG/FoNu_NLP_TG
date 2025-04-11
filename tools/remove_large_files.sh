#!/bin/bash
# Script to remove large files from Git tracking

# Make sure we're in the project root
cd "$(git rev-parse --show-toplevel)" || exit 1

echo "Removing large files from Git tracking..."

# Remove the specific large files from Git tracking
git rm --cached data/processed/clean_ewe_french.csv
git rm --cached data/processed/ewe_french.csv

# Remove any other large CSV files in the data/processed directory
git rm --cached data/processed/*.csv

echo "Creating a new commit with the updated .gitignore and .gitattributes..."
git add .gitignore .gitattributes
git commit -m "Update .gitignore and add .gitattributes for Git LFS"

echo "Setting up Git LFS..."
# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is not installed. Please install it first:"
    echo "  - On Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  - On macOS: brew install git-lfs"
    echo "  - On Windows: Download from https://git-lfs.github.com/"
    exit 1
fi

# Initialize Git LFS
git lfs install

echo "Tracking large files with Git LFS..."
git lfs track "*.csv"
git lfs track "data/processed/*.csv"
git lfs track "data/processed/clean_*.csv"
git lfs track "data/processed/*_train.csv"
git lfs track "data/processed/*_val.csv"
git lfs track "data/processed/*_test.csv"

echo "Adding the large files back with Git LFS..."
# Only add the files if they exist
if [ -f "data/processed/clean_ewe_french.csv" ]; then
    git add data/processed/clean_ewe_french.csv
fi

if [ -f "data/processed/ewe_french.csv" ]; then
    git add data/processed/ewe_french.csv
fi

# Add any other CSV files in the data/processed directory
find data/processed -name "*.csv" -exec git add {} \;

echo "Creating a new commit with the Git LFS tracked files..."
git add .gitattributes
git commit -m "Add large files with Git LFS"

echo "Done! You can now push to GitHub with:"
echo "  git push origin main:main"
echo ""
echo "Note: If you still have issues, you may need to force push:"
echo "  git push -f origin main:main"
echo ""
echo "Or you may need to completely remove the large files from your repository history."
echo "For that, consider using the BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/"
