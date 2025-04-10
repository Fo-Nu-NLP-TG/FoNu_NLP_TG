#!/usr/bin/env python3
"""
Script to create train/val/test splits from the combined dataset.
"""

import pandas as pd
import os
from dataset_splitter import create_data_splits

def main():
    data_dir = "./data/processed"
    
    # Check if combined dataset exists
    combined_file = os.path.join(data_dir, "ewe_english.csv")
    if not os.path.exists(combined_file):
        print(f"Error: Combined dataset not found at {combined_file}")
        return
    
    # Load the dataset
    df = pd.read_csv(combined_file)
    print(f"Loaded dataset with {len(df)} rows")
    
    # Create splits using your function
    train_df, val_df, test_df = create_data_splits(df)
    
    # Save splits
    train_df.to_csv(os.path.join(data_dir, "ewe_english_train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "ewe_english_val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "ewe_english_test.csv"), index=False)
    
    print("Created and saved train/val/test splits")

if __name__ == "__main__":
    main()