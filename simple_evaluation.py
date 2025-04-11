import sys
import os

# Add the project root to the path to import custom modules
sys.path.append('.')

# Import the translate function from the existing script
from Attention_Is_All_You_Need.translate import main as translate_main

def main():
    # Run the translate script with example sentences
    translate_main()
    
    # Get some real examples from the test set
    try:
        import pandas as pd
        import torch
        import sentencepiece as spm
        from Attention_Is_All_You_Need.translate import translate
        
        # Load the model and tokenizers
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizers
        src_tokenizer = spm.SentencePieceProcessor()
        tgt_tokenizer = spm.SentencePieceProcessor()
        src_tokenizer.load('./data/processed/ewe_sp.model')
        tgt_tokenizer.load('./data/processed/english_sp.model')
        
        # Load model
        model_path = './models/transformer_ewe_english_final.pt'
        checkpoint = torch.load(model_path, map_location=device)
        
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
        
        # Load test data
        test_df = pd.read_csv('./data/processed/ewe_english_test.csv')
        print("\nReal Test Data Translations:")
        print("-" * 50)
        
        for i in range(min(5, len(test_df))):
            src_text = test_df.iloc[i]['Ewe']
            ref_text = test_df.iloc[i]['English']
            translation = translate(model, src_text, src_tokenizer, tgt_tokenizer, device)
            print(f"Source (Ewe): {src_text}")
            print(f"Reference (English): {ref_text}")
            print(f"Translation (English): {translation}")
            print("-" * 50)
    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
