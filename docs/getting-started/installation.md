# Installation Guide

This guide will help you set up your environment to work with the FoNu NLP TG project.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Git
- pip (Python package installer)

## Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Lemniscate-world/FoNu_NLP_TG.git
cd FoNu_NLP_TG
```

## Step 2: Create a Virtual Environment

We recommend using a virtual environment to manage dependencies:

```bash
# Create a virtual environment named 'myenv'
python -m venv myenv

# Activate the virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

## Step 3: Install Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies for the project, including:

- PyTorch for deep learning
- SentencePiece for tokenization
- Transformers for model implementations
- Evaluation metrics
- Visualization tools
- Documentation tools

## Step 4: Download Pre-trained Models (Optional)

If you want to use pre-trained models, you can download them using the provided script:

```bash
python tools/download_models.py
```

This will download the pre-trained Ewe-English transformer model to the `models/` directory.

## Step 5: Verify Installation

To verify that everything is installed correctly, run the following command:

```bash
python -c "import torch; import sentencepiece; import transformers; print('Installation successful!')"
```

If you see "Installation successful!" without any errors, you're ready to go!

## Troubleshooting

If you encounter any issues during installation, please check the following:

### Common Issues

1. **ImportError: No module named 'torch'**
   - Make sure you've activated your virtual environment
   - Try reinstalling PyTorch: `pip install torch torchtext`

2. **SentencePiece installation fails**
   - On Linux, you might need additional dependencies: `apt-get install cmake build-essential pkg-config libgoogle-perftools-dev`
   - On Windows, try installing from a pre-built wheel: `pip install sentencepiece --no-build-isolation`

3. **CUDA-related errors**
   - If you're using a GPU, make sure you have the correct CUDA version installed
   - Try installing the CPU-only version of PyTorch if you don't have a compatible GPU

For more detailed troubleshooting, see our [Import Troubleshooting](../documentation/import-troubleshooting.md) guide.

## Next Steps

Now that you have installed the FoNu NLP TG project, you can:

- Follow the [Quick Start Guide](quick-start.md) to run your first translation
- Explore the [Project Structure](../documentation/project-structure.md) to understand the codebase
- Learn about the [Transformer Architecture](../model/transformer-architecture.md) used in the project
