import os
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
import torch

def load_sentencepiece_tokenizer(model_path):
    """Load a trained SentencePiece tokenizer
    
    Args:
        model_path: Path to the .model file
        
    Returns:
        A SentencePieceProcessor object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SentencePiece model not found at {model_path}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    print(f"Loaded SentencePiece tokenizer with vocabulary size {sp.get_piece_size()}")
    
    return sp

def load_huggingface_tokenizer(tokenizer_path):
    """Load a trained Hugging Face tokenizer
    
    Args:
        tokenizer_path: Path to the tokenizer directory or .json file
        
    Returns:
        A PreTrainedTokenizerFast object
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Hugging Face tokenizer not found at {tokenizer_path}")
    
    if os.path.isdir(tokenizer_path):
        # Load from directory
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        # Load from json file
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
    # Set special tokens if they're not already set
    special_tokens = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
        "mask_token": "<MASK>"
    }
    tokenizer.add_special_tokens(special_tokens)
    
    print(f"Loaded Hugging Face tokenizer with vocabulary size {len(tokenizer)}")
    
    return tokenizer

# Creates a translation dataset using tokenized source (Ewe) and target (English) languages)
def create_translation_dataset(src_tokenizer, tgt_tokenizer, data_path, src_lang_col, tgt_lang_col, max_len=128):
    """Create a TranslationDataset using the loaded tokenizers
    
    Args:
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        data_path: Path to CSV file with parallel text
        src_lang_col: Column name for source language
        tgt_lang_col: Column name for target language
        max_len: Maximum sequence length
        
    Returns:
        A TranslationDataset object
    """
    from transformer_dataset import TranslationDataset
    
    dataset = TranslationDataset(
        data_path=data_path,
        src_lang_col=src_lang_col,
        tgt_lang_col=tgt_lang_col,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len
    )
    
    print(f"Created dataset with {len(dataset)} translation pairs")
    
    return dataset

# Example usage
if __name__ == "__main__":
    # Paths to your trained tokenizers
    data_dir = "./data/processed"
    
    # Load SentencePiece tokenizers
    src_sp_path = os.path.join(data_dir, "ewe_sp.model")
    tgt_sp_path = os.path.join(data_dir, "english_sp.model")
    
    try:
        src_tokenizer = load_sentencepiece_tokenizer(src_sp_path)
        tgt_tokenizer = load_sentencepiece_tokenizer(tgt_sp_path)
        
        # Create dataset with SentencePiece tokenizers
        dataset = create_translation_dataset(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            data_path=os.path.join(data_dir, "ewe_english_train.csv"),
            src_lang_col="Ewe",
            tgt_lang_col="English"
        )
        
        # Create DataLoader
        from torch.utils.data import DataLoader
        batch_size = 32
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda batch: {
                "source": torch.nn.utils.rnn.pad_sequence([item["source"] for item in batch], batch_first=True, padding_value=0),
                "target": torch.nn.utils.rnn.pad_sequence([item["target"] for item in batch], batch_first=True, padding_value=0),
                "source_text": [item["source_text"] for item in batch],
                "target_text": [item["target_text"] for item in batch]
            }
        )
        
        print(f"Created DataLoader with batch size {batch_size}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Trying Hugging Face tokenizers instead...")
        
        # Load Hugging Face tokenizers
        tokenizers_dir = os.path.join(data_dir, "tokenizers")
        src_hf_path = os.path.join(tokenizers_dir, "ewe_hf_tokenizer")
        tgt_hf_path = os.path.join(tokenizers_dir, "english_hf_tokenizer")
        
        try:
            src_tokenizer = load_huggingface_tokenizer(src_hf_path)
            tgt_tokenizer = load_huggingface_tokenizer(tgt_hf_path)
            
            # Create dataset with Hugging Face tokenizers
            dataset = create_translation_dataset(
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                data_path=os.path.join(data_dir, "ewe_english_train.csv"),
                src_lang_col="Ewe",
                tgt_lang_col="English"
            )
            
            # Create DataLoader
            from torch.utils.data import DataLoader
            batch_size = 32
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
            
            print(f"Created DataLoader with batch size {batch_size}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train tokenizers first using tokenizer_trainer.py")