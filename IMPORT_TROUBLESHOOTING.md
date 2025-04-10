# Troubleshooting Import Issues

This document provides solutions for common import issues in the FoNu_NLP_TG project.

## Common Import Issues

### 1. ModuleNotFoundError: No module named 'model_utils'

This error occurs when Python cannot find the `model_utils.py` file. This is a common issue when running Jupyter notebooks or scripts from different directories.

### 2. Jupyter Notebook Import Errors

Jupyter notebooks often have issues with imports because they run in a different context than regular Python scripts.

### 3. Colab Kernel Launcher Errors

When running in Google Colab, you might see errors related to the `colab_kernel_launcher.py` script.

## Solutions

### 1. Use the Fixed Scripts

We've created fixed versions of the scripts that handle imports correctly:

- `train_transformer_cli.py`: Command-line script for training the transformer model
- `train_transformer_fixed.ipynb`: Fixed Jupyter notebook for training the transformer model
- `train_transformer.py`: Launcher script that runs the CLI script
- `colab_kernel_launcher.py`: Launcher script for Google Colab

### 2. Fix Existing Notebooks

You can use the `fix_notebook_imports.py` script to fix import issues in existing notebooks:

```bash
python fix_notebook_imports.py path/to/your/notebook.ipynb
```

This will update the import statements in the notebook to use the correct paths.

### 3. Run Scripts from the Project Root

Always run scripts from the project root directory:

```bash
# From the project root directory
python train_transformer.py
```

### 4. Add Paths Manually

If you're writing your own scripts, add the necessary paths manually:

```python
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add the Attention_Is_All_You_Need directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Attention_Is_All_You_Need"))
```

### 5. Use Absolute Imports

Use absolute imports instead of relative imports:

```python
# Instead of
from model_utils import Generator

# Use
from Attention_Is_All_You_Need.model_utils import Generator
```

## Google Colab Specific Issues

When running in Google Colab, you might encounter additional issues:

1. Clone the repository first:

```python
!git clone https://github.com/Lemniscate-world/FoNu_NLP_TG.git
%cd FoNu_NLP_TG
```

2. Use the `colab_kernel_launcher.py` script:

```python
!python colab_kernel_launcher.py
```

3. If you're still having issues, try the dynamic import approach:

```python
import glob
import importlib.util
import sys

# Find the model_utils.py file
model_utils_files = glob.glob("**/model_utils.py", recursive=True)
if model_utils_files:
    # Import it dynamically
    spec = importlib.util.spec_from_file_location("model_utils", model_utils_files[0])
    model_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_utils)
    
    # Get the required classes and functions
    Generator = model_utils.Generator
    Encoder = model_utils.Encoder
    # ... and so on
```

## Still Having Issues?

If you're still having issues, please open an issue on the GitHub repository with details about the error you're encountering.
