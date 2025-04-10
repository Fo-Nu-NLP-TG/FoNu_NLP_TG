import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sentencepiece as spm
import os

class TranslationDataset(Dataset):
    """Dataset for transformer-based translation"""
    
    def __init__(self, 
                 data_path, 
                 src_lang_col, 
                 tgt_lang_col,
                 src_tokenizer,
                 tgt_tokenizer,
                 max_len=128):
        """
        Args:
            data_path: Path to CSV file with parallel text
            src_lang_col: Column name for source language
            tgt_lang_col: Column name for target language
            src_tokenizer: Tokenizer for source language
            tgt_tokenizer: Tokenizer for target language
            max_len: Maximum sequence length
        """
        self.df = pd.read_csv(data_path)
        self.src_lang_col = src_lang_col
        self.tgt_lang_col = tgt_lang_col
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
        # Check if columns exist
        if src_lang_col not in self.df.columns:
            raise ValueError(f"Source language column '{src_lang_col}' not found in dataset")
        if tgt_lang_col not in self.df.columns:
            raise ValueError(f"Target language column '{tgt_lang_col}' not found in dataset")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        src_text = str(self.df.iloc[idx][self.src_lang_col])
        tgt_text = str(self.df.iloc[idx][self.tgt_lang_col])
        
        # Tokenize source text
        if isinstance(self.src_tokenizer, spm.SentencePieceProcessor):
            # SentencePiece tokenizer
            src_ids = self.src_tokenizer.encode(src_text, out_type=int)
            # Add BOS/EOS if not already added by the tokenizer
            if src_ids[0] != 2:  # BOS id
                src_ids = [2] + src_ids
            if src_ids[-1] != 3:  # EOS id
                src_ids = src_ids + [3]
        else:
            # Hugging Face tokenizer
            src_encoding = self.src_tokenizer.encode(src_text)
            src_ids = src_encoding.ids
            
        # Tokenize target text
        if isinstance(self.tgt_tokenizer, spm.SentencePieceProcessor):
            # SentencePiece tokenizer
            tgt_ids = self.tgt_tokenizer.encode(tgt_text, out_type=int)
            # Add BOS/EOS if not already added by the tokenizer
            if tgt_ids[0] != 2:  # BOS id
                tgt_ids = [2] + tgt_ids
            if tgt_ids[-1] != 3:  # EOS id
                tgt_ids = tgt_ids + [3]
        else:
            # Hugging Face tokenizer
            tgt_encoding = self.tgt_tokenizer.encode(tgt_text)
            tgt_ids = tgt_encoding.ids
            
        # Truncate sequences if they exceed max_len
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        
        # Convert to tensors
        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
        
        return {
            "source": src_tensor,
            "target": tgt_tensor,
            "source_text": src_text,
            "target_text": tgt_text
        }
