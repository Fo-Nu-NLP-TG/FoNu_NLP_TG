#!/usr/bin/env python3
"""
Script to train tokenizers for all available languages using the TokenizerTrainer class.
"""

from tokenizer_trainer import TokenizerTrainer
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train tokenizers for translation")
    parser.add_argument("--data_dir", default="./data/processed", help="Directory with corpus files")
    parser.add_argument("--method", choices=["sentencepiece", "huggingface", "both"], 
                        default="both", help="Tokenizer method to use")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size")
    args = parser.parse_args()
    
    # Initialize the tokenizer trainer
    trainer = TokenizerTrainer(data_dir=args.data_dir)
    
    # Check which language corpora are available
    languages = []
    for lang in ['ewe', 'english', 'french']:
        corpus_path = os.path.join(args.data_dir, f"{lang}_corpus.txt")
        if os.path.exists(corpus_path):
            languages.append(lang)
            print(f"Found corpus for {lang}")
        else:
            print(f"No corpus found for {lang}")
    
    if not languages:
        print("No corpus files found. Please run data_cleaner.py first.")
        return
    
    # Train tokenizers based on method
    if args.method in ["sentencepiece", "both"]:
        print("\nTraining SentencePiece tokenizers...")
        for lang in languages:
            print(f"\nTraining SentencePiece tokenizer for {lang}...")
            trainer.train_sentencepiece(lang, vocab_size=args.vocab_size)
    
    if args.method in ["huggingface", "both"]:
        print("\nTraining Hugging Face tokenizers...")
        for lang in languages:
            print(f"\nTraining Hugging Face tokenizer for {lang}...")
            trainer.train_huggingface_tokenizer(lang, vocab_size=args.vocab_size)
    
    print("\nTokenizer training complete!")
    if args.method in ["huggingface", "both"]:
        print("Hugging Face tokenizers are saved in:", os.path.join(args.data_dir, "tokenizers"))
    if args.method in ["sentencepiece", "both"]:
        print("SentencePiece models are saved in:", args.data_dir)

if __name__ == "__main__":
    main()