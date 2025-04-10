def create_data_splits(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """Create train/validation/test splits with stratification if possible
    
    Args:
        df: DataFrame with parallel text
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    from sklearn.model_selection import train_test_split
    
    # First split: training vs (validation + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=val_size + test_size,
        random_state=random_state
    )
    
    # Second split: validation vs test
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state
    )
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df