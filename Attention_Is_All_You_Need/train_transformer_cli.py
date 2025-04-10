#!/usr/bin/env python3
"""
Command-line script for training a transformer model for translation.
This script is designed to be run from the command line and handles imports correctly.
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# Add the parent directory to the path to handle imports correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import from the Attention_Is_All_You_Need directory
try:
    from model_utils import Generator, Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadedAttention
    from model_utils import PositionwiseFeedForward, PositionalEncoding, Embeddings, subsequent_mask
    from encode_decode import EncodeDecode
except ImportError:
    try:
        from Attention_Is_All_You_Need.model_utils import Generator, Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadedAttention
        from Attention_Is_All_You_Need.model_utils import PositionwiseFeedForward, PositionalEncoding, Embeddings, subsequent_mask
        from Attention_Is_All_You_Need.encode_decode import EncodeDecode
    except ImportError as e:
        print(f"Error importing transformer modules: {e}")
        print("Make sure you're running this script from the project root directory")
        sys.exit(1)

# Import data processing modules
try:
    from data_processing.load_tokenizers import load_sentencepiece_tokenizer, load_huggingface_tokenizer, create_translation_dataset
except ImportError as e:
    print(f"Error importing data processing modules: {e}")
    print("Make sure the data_processing directory is in your path")
    sys.exit(1)


def make_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Construct a full transformer model"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncodeDecode(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        Generator(d_model, tgt_vocab_size)
    )
    
    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        src = batch["source"].to(device)
        tgt = batch["target"].to(device)
        
        # Create masks
        src_mask = (src != 0).unsqueeze(-2)
        tgt_mask = (tgt != 0).unsqueeze(-2)
        
        # Create subsequent mask for target sequence
        tgt_len = tgt.size(1) - 1  # Adjust for the shifted target
        subsequent_mask_tensor = subsequent_mask(tgt_len).to(device)
        
        # Apply both padding mask and subsequent mask
        tgt_mask = tgt_mask[:, :, :-1] & subsequent_mask_tensor
        
        # Forward pass - shift target by 1 for teacher forcing
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        # Calculate loss
        loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                         tgt[:, 1:].contiguous().view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train a transformer model for translation")
    parser.add_argument("--data_dir", default="./data/processed", help="Directory with processed data")
    parser.add_argument("--src_lang", default="ewe", help="Source language")
    parser.add_argument("--tgt_lang", default="english", help="Target language")
    parser.add_argument("--tokenizer_type", choices=["sentencepiece", "huggingface"], default="sentencepiece", 
                        help="Type of tokenizer to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--save_dir", default="./models", help="Directory to save models")
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizers
    if args.tokenizer_type == "sentencepiece":
        src_tokenizer_path = os.path.join(args.data_dir, f"{args.src_lang}_sp.model")
        tgt_tokenizer_path = os.path.join(args.data_dir, f"{args.tgt_lang}_sp.model")
        
        src_tokenizer = load_sentencepiece_tokenizer(src_tokenizer_path)
        tgt_tokenizer = load_sentencepiece_tokenizer(tgt_tokenizer_path)
    else:
        tokenizers_dir = os.path.join(args.data_dir, "tokenizers")
        src_tokenizer_path = os.path.join(tokenizers_dir, f"{args.src_lang}_hf_tokenizer")
        tgt_tokenizer_path = os.path.join(tokenizers_dir, f"{args.tgt_lang}_hf_tokenizer")
        
        src_tokenizer = load_huggingface_tokenizer(src_tokenizer_path)
        tgt_tokenizer = load_huggingface_tokenizer(tgt_tokenizer_path)
    
    # Create dataset
    train_data_path = os.path.join(args.data_dir, f"{args.src_lang}_{args.tgt_lang}_train.csv")
    val_data_path = os.path.join(args.data_dir, f"{args.src_lang}_{args.tgt_lang}_val.csv")
    
    train_dataset = create_translation_dataset(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        data_path=train_data_path,
        src_lang_col=args.src_lang.capitalize(),
        tgt_lang_col=args.tgt_lang.capitalize(),
        max_len=args.max_len
    )
    
    val_dataset = create_translation_dataset(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        data_path=val_data_path,
        src_lang_col=args.src_lang.capitalize(),
        tgt_lang_col=args.tgt_lang.capitalize(),
        max_len=args.max_len
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: {
            "source": torch.nn.utils.rnn.pad_sequence([item["source"] for item in batch], batch_first=True, padding_value=0),
            "target": torch.nn.utils.rnn.pad_sequence([item["target"] for item in batch], batch_first=True, padding_value=0),
            "source_text": [item["source_text"] for item in batch],
            "target_text": [item["target_text"] for item in batch]
        }
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        collate_fn=lambda batch: {
            "source": torch.nn.utils.rnn.pad_sequence([item["source"] for item in batch], batch_first=True, padding_value=0),
            "target": torch.nn.utils.rnn.pad_sequence([item["target"] for item in batch], batch_first=True, padding_value=0),
            "source_text": [item["source_text"] for item in batch],
            "target_text": [item["target_text"] for item in batch]
        }
    )
    
    # Get vocabulary sizes
    if args.tokenizer_type == "sentencepiece":
        src_vocab_size = src_tokenizer.get_piece_size()
        tgt_vocab_size = tgt_tokenizer.get_piece_size()
    else:
        src_vocab_size = len(src_tokenizer)
        tgt_vocab_size = len(tgt_tokenizer)
    
    # Create model
    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        N=args.layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.heads,
        dropout=args.dropout
    )
    model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    
    # Train the model
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"transformer_{args.src_lang}_{args.tgt_lang}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'args': vars(args)
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"transformer_{args.src_lang}_{args.tgt_lang}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'args': vars(args)
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    main()
