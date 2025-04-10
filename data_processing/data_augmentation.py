def augment_translation_data(df, src_col, tgt_col, techniques=None):
    """Augment translation data with various techniques
    
    Args:
        df: DataFrame with parallel text
        src_col: Source language column name
        tgt_col: Target language column name
        techniques: List of techniques to apply
    
    Returns:
        Augmented DataFrame
    """
    if techniques is None:
        techniques = ['word_dropout', 'word_swap']
    
    augmented_data = [df]
    
    for technique in techniques:
        if technique == 'word_dropout':
            # Randomly drop words with 10% probability
            aug_df = df.copy()
            aug_df[src_col] = aug_df[src_col].apply(
                lambda x: ' '.join([w for w in x.split() if np.random.random() > 0.1])
            )
            augmented_data.append(aug_df)
            
        elif technique == 'word_swap':
            # Randomly swap adjacent words with 20% probability
            aug_df = df.copy()
            aug_df[src_col] = aug_df[src_col].apply(word_swap)
            
            augmented_data.append(aug_df)
    
    return pd.concat(augmented_data).reset_index(drop=True)

def word_swap(text):
    words = text.split()
    for i in range(len(words) - 1):
        if np.random.random() < 0.2:
            words[i], words[i+1] = words[i+1], words[i]
    return ' '.join(words)