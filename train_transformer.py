#!/usr/bin/env python3
"""
Launcher script for training a transformer model.
This script simply imports and runs the main function from the CLI script.
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function from the CLI script
try:
    from Attention_Is_All_You_Need.train_transformer_cli import main
    main()
except ImportError as e:
    print(f"Error importing transformer training module: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)
