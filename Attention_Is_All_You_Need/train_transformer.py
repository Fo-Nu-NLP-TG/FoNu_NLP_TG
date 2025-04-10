import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import copy

# Add parent directory to path to import from data_processing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.load_tokenizers import load_sentencepiece_tokenizer, load_huggingface_tokenizer, create_translation_dataset
from model_utils import Generator, Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings, subsequent_mask
class EncodeDecode(nn.Module):
    """Standard Encoder-Decoder architecture"""
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncodeDecode, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences"""
        memory = self.encode(src, src_mask)
        decoder_output = self.decode(memory, src_mask, tgt, tgt_mask)
        # Apply the generator to get logits over vocabulary
        return self.generator(decoder_output)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """Define standard linear + softmax generation step"""
    
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """Project features to vocabulary size"""
        # Debug output
        if hasattr(self, 'proj'):
            # Print shape information for debugging
            # print(f"Generator input shape: {x.shape}, output features: {self.proj.out_features}")
            pass
        return self.proj(x)

def make_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Construct a full transformer model"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    # Print vocabulary sizes for debugging
    print(f"Creating model with src_vocab_size={src_vocab_size}, tgt_vocab_size={tgt_vocab_size}")
    
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

def train_epoch(model, dataloader, optimizer, criterion, device, tgt_vocab_size):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
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
        
        # Debug output dimensions
        if batch_idx == 0:
            print(f"Output shape: {output.shape}, Output size(-1): {output.size(-1)}")
            print(f"Target shape: {tgt[:, 1:].shape}")
            print(f"Target flat shape: {tgt[:, 1:].contiguous().view(-1).shape}")
        
        # Add debug info to find problematic indices
        target_flat = tgt[:, 1:].contiguous().view(-1)
        max_target_idx = target_flat.max().item()
        
        # Print more detailed debugging information
        if batch_idx == 0 or max_target_idx >= tgt_vocab_size:
            print(f"Batch {batch_idx}: Max target index: {max_target_idx}, Vocab size: {tgt_vocab_size}")
            
            if max_target_idx >= tgt_vocab_size:
                print(f"WARNING: Target index {max_target_idx} exceeds vocabulary size {tgt_vocab_size}")
                # Find all problematic indices
                problematic_indices = (target_flat >= tgt_vocab_size).nonzero().squeeze().tolist()
                if not isinstance(problematic_indices, list):
                    problematic_indices = [problematic_indices]
                
                # Print some of the problematic examples
                for idx in problematic_indices[:3]:  # Print first 3 problematic indices
                    batch_idx = idx // (tgt.size(1) - 1)
                    seq_idx = idx % (tgt.size(1) - 1)
                    if batch_idx < len(batch["target_text"]):
                        print(f"Problematic text: {batch['target_text'][batch_idx]}")
                        print(f"Problematic index position: {seq_idx}")
                        print(f"Token ID: {target_flat[idx].item()}")
        
        # Ensure target indices are within bounds by clamping
        target_flat = torch.clamp(target_flat, 0, tgt_vocab_size - 1)
        
        # Check output dimensions match expected
        if output.size(-1) != tgt_vocab_size:
            print(f"ERROR: Output dimension {output.size(-1)} doesn't match target vocab size {tgt_vocab_size}")
        
        # Calculate loss
        loss = criterion(output.contiguous().view(-1, output.size(-1)), target_flat)
        
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
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Create model with explicit vocabulary sizes
    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        N=args.layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.heads,
        dropout=args.dropout
    )
    
    # Check Generator output dimension
    if hasattr(model, 'generator') and hasattr(model.generator, 'proj'):
        print(f"Generator output dimension: {model.generator.proj.out_features}")
        if model.generator.proj.out_features != tgt_vocab_size:
            print(f"WARNING: Generator output dimension doesn't match target vocabulary size!")
            # Try to fix it
            model.generator.proj = nn.Linear(model.generator.proj.in_features, tgt_vocab_size)
            print(f"Fixed Generator output dimension to: {model.generator.proj.out_features}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index (0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, tgt_vocab_size)
        print(f"Train Loss: {train_loss:.4f}")
        
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
    import copy
    main()
