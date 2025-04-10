import sentencepiece as spm
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

class TokenizerTrainer:
    """Train tokenizers for Ewe and target languages (English/French)"""
    
    def __init__(self, data_dir="./data/processed"):
        self.data_dir = data_dir
    
    def train_sentencepiece(self, lang, vocab_size=8000, model_type='unigram'):
        """Train a SentencePiece tokenizer
        
        Args:
            lang: Language code ('ewe', 'english', or 'french')
            vocab_size: Size of vocabulary
            model_type: 'unigram' or 'bpe'
        """
        input_file = os.path.join(self.data_dir, f"{lang}_corpus.txt")
        model_prefix = os.path.join(self.data_dir, f"{lang}_sp")
        
        if not os.path.exists(input_file):
            print(f"Corpus file not found: {input_file}")
            return None
        
        print(f"Training SentencePiece tokenizer for {lang} with {model_type} model")
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,  # For languages with small character sets
            model_type=model_type,
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc',  # Normalization for NMT
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']
        )
        
        # Load the model
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        
        print(f"Trained {lang} tokenizer with vocabulary size {sp.get_piece_size()}")
        
        # Test the tokenizer
        test_sentences = {
            'ewe': "Ŋdi nyuie, èfɔ̀ nyuiê",
            'english': "Good morning, how are you?",
            'french': "Bonjour, comment allez-vous?"
        }
        
        if lang in test_sentences:
            test_text = test_sentences[lang]
            tokens = sp.encode_as_pieces(test_text)
            print(f"Test tokenization for '{test_text}':")
            print(tokens)
        
        return sp
    
    def train_huggingface_tokenizer(self, lang, vocab_size=8000, model_type='bpe'):
        """Train a Hugging Face tokenizer
        
        Args:
            lang: Language code ('ewe', 'english', or 'french')
            vocab_size: Size of vocabulary
            model_type: 'bpe' or 'wordpiece'
        """
        input_file = os.path.join(self.data_dir, f"{lang}_corpus.txt")
        output_dir = os.path.join(self.data_dir, "tokenizers")
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(input_file):
            print(f"Corpus file not found: {input_file}")
            return None
        
        # Initialize tokenizer based on model type
        if model_type.lower() == 'bpe':
            tokenizer = Tokenizer(models.BPE())
        elif model_type.lower() == 'wordpiece':
            tokenizer = Tokenizer(models.WordPiece())
        else:
            print(f"Unsupported model type: {model_type}")
            return None
        
        # Use whitespace pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Prepare trainer
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]
        if model_type.lower() == 'bpe':
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens
            )
        else:  # wordpiece
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens
            )
        
        # Train the tokenizer
        tokenizer.train(files=[input_file], trainer=trainer)
        
        # Add post-processor
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>"))
            ]
        )
        
        # Save the tokenizer
        output_path = os.path.join(output_dir, f"{lang}_tokenizer.json")
        tokenizer.save(output_path)
        
        # Create a Hugging Face compatible tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=output_path,
            bos_token="<BOS>",
            eos_token="<EOS>",
            unk_token="<UNK>",
            pad_token="<PAD>",
            mask_token="<MASK>"
        )
        
        # Save the Hugging Face tokenizer
        hf_tokenizer.save_pretrained(os.path.join(output_dir, f"{lang}_hf_tokenizer"))
        
        print(f"Trained {lang} tokenizer with vocabulary size {len(tokenizer.get_vocab())}")
        
        # Test the tokenizer
        test_sentences = {
            'ewe': "Ŋdi nyuie, èfɔ̀ nyuiê",
            'english': "Good morning, how are you?",
            'french': "Bonjour, comment allez-vous?"
        }
        
        if lang in test_sentences:
            test_text = test_sentences[lang]
            encoding = tokenizer.encode(test_text)
            print(f"Test tokenization for '{test_text}':")
            print(encoding.tokens)
        
        return tokenizer, hf_tokenizer
    
    def train_all_tokenizers(self, method='sentencepiece', vocab_size=8000):
        """Train tokenizers for all languages
        
        Args:
            method: 'sentencepiece' or 'huggingface'
            vocab_size: Size of vocabulary
        """
        languages = []
        
        # Check which language corpora are available
        for lang in ['ewe', 'english', 'french']:
            if os.path.exists(os.path.join(self.data_dir, f"{lang}_corpus.txt")):
                languages.append(lang)
        
        tokenizers = {}
        
        for lang in languages:
            if method.lower() == 'sentencepiece':
                tokenizers[lang] = self.train_sentencepiece(lang, vocab_size)
            else:  # huggingface
                tokenizers[lang] = self.train_huggingface_tokenizer(lang, vocab_size)
        
        return tokenizers