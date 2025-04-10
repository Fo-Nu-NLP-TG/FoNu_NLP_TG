#!/usr/bin/env python3
"""
Script to fix import issues in Jupyter notebooks.
This script updates the import statements in the notebook to use the correct paths.
"""

import os
import sys
import json
import argparse

def fix_notebook_imports(notebook_path, output_path=None):
    """
    Fix import issues in a Jupyter notebook.
    
    Args:
        notebook_path (str): Path to the notebook to fix
        output_path (str, optional): Path to save the fixed notebook. If None, overwrites the original.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if output_path is None:
        output_path = notebook_path
    
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find cells with import statements
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = cell['source']
                
                # Check if this is an import cell
                import_cell = any('import' in line for line in source)
                
                if import_cell:
                    # Replace the import cell with our robust import code
                    cell['source'] = [
                        "import os\n",
                        "import sys\n",
                        "import copy\n",
                        "import torch\n",
                        "import torch.nn as nn\n",
                        "import torch.optim as optim\n",
                        "from torch.utils.data import DataLoader\n",
                        "import argparse\n",
                        "\n",
                        "# Add this to debug and fix path issues\n",
                        "print(\"Current directory:\", os.getcwd())\n",
                        "\n",
                        "# First, determine if we're running in Colab or locally\n",
                        "IN_COLAB = 'google.colab' in sys.modules\n",
                        "print(f\"Running in Google Colab: {IN_COLAB}\")\n",
                        "\n",
                        "# Add the parent directory to the path to handle imports correctly\n",
                        "current_dir = os.path.dirname(os.path.abspath(\"__file__\"))\n",
                        "parent_dir = os.path.dirname(current_dir)\n",
                        "print(f\"Adding {parent_dir} to path\")\n",
                        "sys.path.append(parent_dir)\n",
                        "\n",
                        "# If we're in the Attention_Is_All_You_Need directory, add the current directory too\n",
                        "if os.path.basename(current_dir) == \"Attention_Is_All_You_Need\":\n",
                        "    sys.path.append(current_dir)\n",
                        "    print(f\"Also adding {current_dir} to path\")\n",
                        "\n",
                        "# Try to find the model_utils.py file\n",
                        "import glob\n",
                        "model_utils_files = glob.glob(\"**/model_utils.py\", recursive=True)\n",
                        "print(\"Found model_utils.py at:\", model_utils_files)\n",
                        "\n",
                        "# If we found model_utils.py, add its directory to the path\n",
                        "if model_utils_files:\n",
                        "    model_dir = os.path.dirname(model_utils_files[0])\n",
                        "    print(f\"Adding {model_dir} to path\")\n",
                        "    sys.path.append(model_dir)\n",
                        "    \n",
                        "    # Add the Attention_Is_All_You_Need directory explicitly\n",
                        "    attention_dir = os.path.join(os.getcwd(), \"Attention_Is_All_You_Need\")\n",
                        "    if os.path.exists(attention_dir):\n",
                        "        sys.path.append(attention_dir)\n",
                        "        print(f\"Adding {attention_dir} to path\")\n",
                        "\n",
                        "# Try different import strategies\n",
                        "try:\n",
                        "    # Try direct import first (if we're in the same directory)\n",
                        "    from model_utils import Generator, Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadedAttention\n",
                        "    from model_utils import PositionwiseFeedForward, PositionalEncoding, Embeddings, subsequent_mask\n",
                        "    from encode_decode import EncodeDecode\n",
                        "    print(\"Direct import worked!\")\n",
                        "except ImportError:\n",
                        "    try:\n",
                        "        # Try with full module path\n",
                        "        from Attention_Is_All_You_Need.model_utils import Generator, Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadedAttention\n",
                        "        from Attention_Is_All_You_Need.model_utils import PositionwiseFeedForward, PositionalEncoding, Embeddings, subsequent_mask\n",
                        "        from Attention_Is_All_You_Need.encode_decode import EncodeDecode\n",
                        "        print(\"Import with Attention_Is_All_You_Need prefix worked!\")\n",
                        "    except ImportError as e:\n",
                        "        print(f\"Import error: {e}\")\n",
                        "        \n",
                        "        # Last resort: try to dynamically import from the found file\n",
                        "        if model_utils_files:\n",
                        "            import importlib.util\n",
                        "            spec = importlib.util.spec_from_file_location(\"model_utils\", model_utils_files[0])\n",
                        "            model_utils = importlib.util.module_from_spec(spec)\n",
                        "            spec.loader.exec_module(model_utils)\n",
                        "            \n",
                        "            # Get the required classes and functions\n",
                        "            Generator = model_utils.Generator\n",
                        "            Encoder = model_utils.Encoder\n",
                        "            Decoder = model_utils.Decoder\n",
                        "            EncoderLayer = model_utils.EncoderLayer\n",
                        "            DecoderLayer = model_utils.DecoderLayer\n",
                        "            MultiHeadedAttention = model_utils.MultiHeadedAttention\n",
                        "            PositionwiseFeedForward = model_utils.PositionwiseFeedForward\n",
                        "            PositionalEncoding = model_utils.PositionalEncoding\n",
                        "            Embeddings = model_utils.Embeddings\n",
                        "            subsequent_mask = model_utils.subsequent_mask\n",
                        "            \n",
                        "            # Now try to import encode_decode\n",
                        "            encode_decode_files = glob.glob(\"**/encode_decode.py\", recursive=True)\n",
                        "            if encode_decode_files:\n",
                        "                spec = importlib.util.spec_from_file_location(\"encode_decode\", encode_decode_files[0])\n",
                        "                encode_decode = importlib.util.module_from_spec(spec)\n",
                        "                spec.loader.exec_module(encode_decode)\n",
                        "                EncodeDecode = encode_decode.EncodeDecode\n",
                        "                print(\"Dynamic import worked!\")\n",
                        "            else:\n",
                        "                print(\"Could not find encode_decode.py\")\n",
                        "        else:\n",
                        "            print(\"All import attempts failed.\")\n"
                    ]
                    # Only fix one cell (the first import cell)
                    break
        
        # Write the fixed notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Successfully fixed notebook: {output_path}")
        return True
    
    except Exception as e:
        print(f"Error fixing notebook: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix import issues in Jupyter notebooks")
    parser.add_argument("notebook_path", help="Path to the notebook to fix")
    parser.add_argument("--output", "-o", help="Path to save the fixed notebook. If not provided, overwrites the original.")
    args = parser.parse_args()
    
    fix_notebook_imports(args.notebook_path, args.output)


if __name__ == "__main__":
    main()
