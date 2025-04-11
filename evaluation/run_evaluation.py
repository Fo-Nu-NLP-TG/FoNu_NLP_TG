import torch
import argparse
from sacrebleu import corpus_bleu
from tqdm import tqdm
import sentencepiece as spm
import sys
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Add the project root to the path to import custom modules
sys.path.append('.')

# Import necessary modules
from Attention_Is_All_You_Need.model_utils import subsequent_mask, clones, Generator, LayerNorm, EncoderLayer, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, Embeddings, PositionalEncoding
# We'll define our own make_model function to avoid circular imports

# Import EncodeDecode directly to avoid circular import issues
import torch.nn as nn
import copy

class EncodeDecode(nn.Module):
    """EncodeDecode is a base class for encoder-decoder architectures in sequence-to-sequence models."""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncodeDecode, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Perform the forward pass of the encoder-decoder model.

        Args:
            src (torch.Tensor): Source sequence.
            tgt (torch.Tensor): Target sequence.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Output of the model.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """Encode the source sequence.

        Args:
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor): Source mask.

        Returns:
            torch.Tensor: Encoded representation of the source sequence.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """Decode the target sequence.

        Args:
            memory (torch.Tensor): Encoded representation of the source sequence.
            src_mask (torch.Tensor): Source mask.
            tgt (torch.Tensor): Target sequence.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Decoded representation of the target sequence.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# Define Encoder class
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Define Decoder class
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# Define make_model function
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncodeDecode(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # Initialize parameters with Xavier uniform
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class SimpleTranslationDataset(Dataset):
    """Simple dataset for translation evaluation"""

    def __init__(self, data_path, src_col='source', tgt_col='target'):
        """
        Args:
            data_path: Path to CSV file with parallel text
            src_col: Column name for source language
            tgt_col: Column name for target language
        """
        self.df = pd.read_csv(data_path)
        self.src_col = src_col
        self.tgt_col = tgt_col

        # Check if columns exist
        if src_col not in self.df.columns:
            raise ValueError(f"Source language column '{src_col}' not found in dataset")
        if tgt_col not in self.df.columns:
            raise ValueError(f"Target language column '{tgt_col}' not found in dataset")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = str(self.df.iloc[idx][self.src_col])
        tgt_text = str(self.df.iloc[idx][self.tgt_col])

        return {
            "source_text": src_text,
            "target_text": tgt_text
        }

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

def evaluate_bleu(model, test_dataloader, src_tokenizer, tgt_tokenizer, device, max_samples=None):
    """Evaluate BLEU score on test data"""
    references = []
    hypotheses = []
    sample_count = 0

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        # Extract source and target texts from the batch dictionary
        src_texts = batch['source_text']
        tgt_texts = batch['target_text']

        for src_text, tgt_text in zip(src_texts, tgt_texts):
            # Generate translation
            translated = translate(model, src_text, src_tokenizer, tgt_tokenizer, device)
            hypotheses.append(translated)
            references.append([tgt_text])  # sacrebleu expects a list of references

            sample_count += 1
            if max_samples and sample_count >= max_samples:
                break

        if max_samples and sample_count >= max_samples:
            break

    # Calculate BLEU score
    bleu = corpus_bleu(hypotheses, references)
    return bleu.score, hypotheses, references

def evaluate_examples(model, examples, src_tokenizer, tgt_tokenizer, device):
    """Evaluate translation on example sentences"""
    results = []
    for src_text in examples:
        translation = translate(model, src_text, src_tokenizer, tgt_tokenizer, device)
        results.append((src_text, translation))
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer Translation Model')
    parser.add_argument('--model_path', type=str, default='./models/transformer_ewe_english_final.pt',
                        help='Path to the trained model')
    parser.add_argument('--src_tokenizer', type=str, default='./data/processed/ewe_sp.model',
                        help='Path to the source tokenizer model')
    parser.add_argument('--tgt_tokenizer', type=str, default='./data/processed/english_sp.model',
                        help='Path to the target tokenizer model')
    parser.add_argument('--test_data', type=str, default='./data/processed/ewe_english_test.csv',
                        help='Path to the test data')
    parser.add_argument('--src_col', type=str, default='source',
                        help='Column name for source language in test data')
    parser.add_argument('--tgt_col', type=str, default='target',
                        help='Column name for target language in test data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to evaluate (for faster testing)')
    parser.add_argument('--examples_file', type=str, default='./Attention_Is_All_You_Need/example_sentences.txt',
                        help='Path to file with example sentences to translate')
    parser.add_argument('--output_file', type=str, default='./evaluation_results.txt',
                        help='Path to save evaluation results')

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

    # Create test dataloader
    try:
        test_dataset = SimpleTranslationDataset(
            args.test_data,
            src_col=args.src_col,
            tgt_col=args.tgt_col
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        # Evaluate BLEU score
        print(f"Evaluating BLEU score on {min(args.max_samples, len(test_dataset))} samples...")
        bleu_score, hypotheses, references = evaluate_bleu(
            model, test_dataloader, src_tokenizer, tgt_tokenizer, device, args.max_samples
        )
        print(f"BLEU score: {bleu_score:.2f}")

        has_test_results = True
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        print("Skipping BLEU evaluation.")
        has_test_results = False
        bleu_score = None
        hypotheses = []
        references = []

    # Evaluate example sentences
    example_results = []
    if args.examples_file and os.path.exists(args.examples_file):
        with open(args.examples_file, 'r', encoding='utf-8') as f:
            examples = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Evaluating {len(examples)} example sentences...")
        example_results = evaluate_examples(
            model, examples, src_tokenizer, tgt_tokenizer, device
        )

        print("\nExample Translations:")
        for src, tgt in example_results:
            print(f"Source: {src}")
            print(f"Translation: {tgt}")
            print("-" * 50)
    else:
        print(f"Examples file not found: {args.examples_file}")

    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        if has_test_results:
            f.write(f"BLEU Score: {bleu_score:.2f}\n\n")

        if example_results:
            f.write("Example Translations:\n")
            for src, tgt in example_results:
                f.write(f"Source: {src}\n")
                f.write(f"Translation: {tgt}\n")
                f.write("-" * 50 + "\n")

        if has_test_results:
            f.write("\nDetailed Translation Samples from Test Set:\n")
            for i, (hyp, ref) in enumerate(zip(hypotheses[:20], references[:20])):
                f.write(f"Example {i+1}:\n")
                f.write(f"Reference: {ref[0]}\n")
                f.write(f"Hypothesis: {hyp}\n")
                f.write("-" * 50 + "\n")

    print(f"Evaluation results saved to {args.output_file}")

if __name__ == "__main__":
    main()
