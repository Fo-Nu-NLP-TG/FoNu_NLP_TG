"""
Simple translation script for the Ewe-English transformer model.
"""

import torch
import argparse
import sentencepiece as spm
import sys
import os

# Add the project root to the path to import custom modules
sys.path.append('.')

# Import subsequent_mask from model_utils
from Attention_Is_All_You_Need.model_utils import subsequent_mask

def generate_square_subsequent_mask(size):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    # Create a mask that prevents attending to future positions
    # This is for the decoder's self-attention
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

    # Convert to binary mask where 1 means keep and 0 means mask
    # This is the format expected by the attention mechanism
    binary_mask = (mask == 0).float()

    return binary_mask

def translate(model, src_text, src_tokenizer, tgt_tokenizer, device, max_len=150):
    """Translate a single source text to target language."""
    model.eval()

    # Tokenize the source text
    src_tokens = src_tokenizer.encode(src_text, out_type=int)
    src_tokens = torch.tensor([src_tokens], dtype=torch.long).to(device)

    # Create source mask - this is for padding
    src_mask = (src_tokens != 0).unsqueeze(1).to(device)

    # Apply embedding and positional encoding
    with torch.no_grad():
        # First apply the embedding layer
        src_embedded = model.src_embed(src_tokens)
        # Then pass the embedded tokens to the encoder
        memory = model.encoder(src_embedded, src_mask)

    # Initialize the output with the start token
    ys = torch.ones(1, 1).fill_(1).type(torch.long).to(device)  # Start token index

    # Generate the translation
    for i in range(max_len):
        # Create target mask (causal mask for decoder)
        # This creates a mask that prevents attending to future positions
        tgt_mask = subsequent_mask(ys.size(1)).to(device)

        with torch.no_grad():
            # Apply target embedding
            tgt_embedded = model.tgt_embed(ys)

            # Pass through decoder
            # The src_mask should be passed to help the cross-attention mechanism
            out = model.decoder(
                tgt_embedded,  # embedded target tokens
                memory,        # encoder output
                src_mask,      # source padding mask for cross-attention
                tgt_mask       # causal mask for self-attention
            )

            # Get prediction for next token
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

        # Add predicted token to output sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=1)

        # Stop if end token is predicted
        if next_word == 2:  # End token index
            break

    # Convert tokens to text
    ys = ys.cpu().numpy().tolist()[0][1:]  # Remove start token
    if 2 in ys:  # Remove end token if present
        ys = ys[:ys.index(2)]

    translation = tgt_tokenizer.decode(ys)
    return translation

def main():
    parser = argparse.ArgumentParser(description='Ewe-English Translation with Transformer')
    parser.add_argument('--model_path', type=str, default='./models/transformer_ewe_english_final.pt',
                        help='Path to the trained model')
    parser.add_argument('--src_tokenizer', type=str, default='./data/processed/ewe_sp.model',
                        help='Path to the source tokenizer model')
    parser.add_argument('--tgt_tokenizer', type=str, default='./data/processed/english_sp.model',
                        help='Path to the target tokenizer model')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to translate (if not provided, will use example sentences)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizers
    src_tokenizer = spm.SentencePieceProcessor()
    tgt_tokenizer = spm.SentencePieceProcessor()
    src_tokenizer.load(args.src_tokenizer)
    tgt_tokenizer.load(args.tgt_tokenizer)
    print(f"Loaded source tokenizer with vocabulary size {src_tokenizer.get_piece_size()}")
    print(f"Loaded target tokenizer with vocabulary size {tgt_tokenizer.get_piece_size()}")

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    print(f"Loaded checkpoint from {args.model_path}")

    # Import the model architecture
    from Attention_Is_All_You_Need.encode_decode import EncodeDecode
    from Attention_Is_All_You_Need.model_utils import make_model

    # Create model with the same parameters as the saved model
    src_vocab_size = checkpoint['src_vocab_size']
    tgt_vocab_size = checkpoint['tgt_vocab_size']

    # Check if args are saved in the checkpoint
    if 'args' in checkpoint:
        args_dict = checkpoint['args']
        d_model = args_dict.get('d_model', 512)
        d_ff = args_dict.get('d_ff', 2048)
        n_heads = args_dict.get('n_heads', 8)
        n_layers = args_dict.get('n_layers', 6)
        dropout = args_dict.get('dropout', 0.1)
    else:
        # Default values
        d_model = 512
        d_ff = 2048
        n_heads = 8
        n_layers = 6
        dropout = 0.1

    # Create the model
    model = make_model(
        src_vocab_size,
        tgt_vocab_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        dropout
    )

    # Load the saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded successfully with src_vocab_size={src_vocab_size}, tgt_vocab_size={tgt_vocab_size}")

    # Example sentences if no text is provided
    example_sentences = [
        "Ŋdi nyuie",
        "Akpe ɖe wo ŋu",
        "Mele tefe ka?",
        "Nye ŋkɔe nye John",
        "Aleke nèfɔ ŋdi sia?"
    ]

    if args.interactive:
        print("\nInteractive Mode - Enter Ewe text to translate (type 'exit' to quit):")
        while True:
            user_input = input("\nEwe > ")
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                continue

            translation = translate(model, user_input, src_tokenizer, tgt_tokenizer, device)
            print(f"English: {translation}")
    elif args.text:
        # Translate the provided text
        translation = translate(model, args.text, src_tokenizer, tgt_tokenizer, device)
        print(f"\nSource (Ewe): {args.text}")
        print(f"Translation (English): {translation}")
    else:
        # Translate example sentences
        print("\nTranslating example sentences:")
        print("-" * 50)

        for src_text in example_sentences:
            translation = translate(model, src_text, src_tokenizer, tgt_tokenizer, device)
            print(f"Source (Ewe): {src_text}")
            print(f"Translation (English): {translation}")
            print("-" * 50)

if __name__ == "__main__":
    main()
