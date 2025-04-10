# Data Preprocessing for Neural Machine Translation

This guide covers essential data preprocessing techniques for NMT systems, with a focus on low-resource language pairs like Ewe-English.

## Table of Contents
- [1. Data Collection and Cleaning](#1-data-collection-and-cleaning)
- [2. Tokenization Approaches](#2-tokenization-approaches)
- [3. Data Augmentation Techniques](#3-data-augmentation-techniques)
- [4. Handling Low-Resource Languages](#4-handling-low-resource-languages)
- [5. Preprocessing Pipeline](#5-preprocessing-pipeline)
- [6. Research Papers and Resources](#6-research-papers-and-resources)

## 1. Data Collection and Cleaning

### Common Issues
- **Misaligned pairs**: Source and target sentences don't match
- **Duplicates**: Redundant sentence pairs
- **Noise**: HTML tags, special characters, inconsistent formatting
- **Length disparities**: Extremely short or long sentences

### Approaches

#### Basic Cleaning
```python
def clean_text(text):
    """Basic text cleaning"""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (customize as needed)
    text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    return text.strip()
```

#### Length Filtering
```python
def filter_by_length(df, src_col, tgt_col, min_len=3, max_len=100):
    """Filter out sentence pairs that are too short or too long"""
    mask = (df[src_col].str.split().str.len() >= min_len) & \
           (df[src_col].str.split().str.len() <= max_len) & \
           (df[tgt_col].str.split().str.len() >= min_len) & \
           (df[tgt_col].str.split().str.len() <= max_len)
    return df[mask]
```

#### Duplicate Removal
```python
def remove_duplicates(df, src_col, tgt_col):
    """Remove duplicate sentence pairs"""
    return df.drop_duplicates(subset=[src_col, tgt_col])
```

#### Language Detection
```python
def filter_by_language(df, src_col, tgt_col, src_lang='ewe', tgt_lang='en'):
    """Filter out sentence pairs with wrong languages using langdetect"""
    from langdetect import detect
    
    def is_correct_language(row):
        try:
            src_detected = detect(row[src_col])
            tgt_detected = detect(row[tgt_col])
            return src_detected == src_lang and tgt_detected == tgt_lang
        except:
            return False
    
    return df[df.apply(is_correct_language, axis=1)]
```

**When to use**: Always perform basic cleaning. Apply length filtering to remove outliers. Remove duplicates to prevent memorization. Use language detection for datasets collected from the web.

## 2. Tokenization Approaches

### Word-level Tokenization
- **Pros**: Intuitive, preserves word boundaries
- **Cons**: Large vocabulary, many OOV words, especially for morphologically rich languages

```python
def word_tokenize(text):
    """Simple word-level tokenization"""
    return text.split()
```

### Subword Tokenization
- **BPE (Byte Pair Encoding)**: Merges frequent character pairs iteratively
- **WordPiece**: Similar to BPE but uses likelihood instead of frequency
- **SentencePiece**: Language-agnostic subword tokenization
- **Unigram Language Model**: Probabilistic subword segmentation

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_bpe_tokenizer(corpus_file, vocab_size=32000):
    """Train a BPE tokenizer"""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )
    tokenizer.train(files=[corpus_file], trainer=trainer)
    return tokenizer
```

### Character-level Tokenization
- **Pros**: No OOV issues, smaller model size
- **Cons**: Longer sequences, may lose semantic meaning

```python
def char_tokenize(text):
    """Character-level tokenization"""
    return list(text)
```

**When to use**: 
- For low-resource languages like Ewe, subword tokenization (especially BPE) is recommended
- Use SentencePiece for languages with non-standard word boundaries
- Character-level can be useful for very low-resource scenarios or as a fallback

## 3. Data Augmentation Techniques

### Back-translation
Generate synthetic parallel data by translating target language sentences back to the source language.

```python
def back_translate(tgt_sentences, reverse_model):
    """Generate synthetic source sentences using back-translation"""
    synthetic_src = reverse_model.translate(tgt_sentences)
    return synthetic_src
```

### Word Dropout
Randomly drop words to create more robust models.

```python
def word_dropout(sentence, dropout_prob=0.1):
    """Randomly drop words with given probability"""
    words = sentence.split()
    kept_words = [w for w in words if random.random() > dropout_prob]
    return ' '.join(kept_words)
```

### Word Replacement
Replace words with synonyms or similar words.

```python
def replace_with_synonyms(sentence, synonym_dict, replace_prob=0.1):
    """Replace words with synonyms with given probability"""
    words = sentence.split()
    for i, word in enumerate(words):
        if random.random() < replace_prob and word in synonym_dict:
            words[i] = random.choice(synonym_dict[word])
    return ' '.join(words)
```

### Noise Addition
Add spelling errors or other noise to make the model more robust.

```python
def add_noise(sentence, noise_prob=0.05):
    """Add character-level noise"""
    chars = list(sentence)
    for i in range(len(chars)):
        if random.random() < noise_prob:
            # Random operations: insert, delete, replace, swap
            op = random.choice(['insert', 'delete', 'replace', 'swap'])
            if op == 'insert' and i < len(chars) - 1:
                chars.insert(i, random.choice('abcdefghijklmnopqrstuvwxyz'))
            elif op == 'delete':
                chars[i] = ''
            elif op == 'replace':
                chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
            elif op == 'swap' and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)
```

**When to use**: 
- Back-translation is essential for low-resource pairs like Ewe-English
- Word dropout helps prevent overfitting
- Synonym replacement when you have a synonym dictionary
- Noise addition for deployment in noisy environments (e.g., OCR, speech-to-text)

## 4. Handling Low-Resource Languages

### Transfer Learning
Leverage high-resource language pairs to improve low-resource translation.

```python
def create_multilingual_corpus(ewe_english_df, related_pairs_df):
    """Combine Ewe-English data with related language pairs"""
    # Ensure column names match
    related_pairs_df = related_pairs_df.rename(
        columns={
            'source': 'ewe',  # Placeholder for the related source language
            'target': 'english'
        }
    )
    # Add language tags if needed
    ewe_english_df['ewe'] = '<ewe> ' + ewe_english_df['ewe']
    related_pairs_df['ewe'] = '<rel> ' + related_pairs_df['ewe']
    
    # Combine datasets
    return pd.concat([ewe_english_df, related_pairs_df])
```

### Data Synthesis
Create synthetic parallel data using templates or rules.

```python
def generate_template_data(templates, vocabulary, n_samples=1000):
    """Generate synthetic data from templates"""
    synthetic_data = []
    
    for _ in range(n_samples):
        template = random.choice(templates)
        # Fill in template slots with vocabulary
        ewe_sent = template['ewe']
        eng_sent = template['english']
        
        for slot in template['slots']:
            word = random.choice(vocabulary[slot])
            ewe_sent = ewe_sent.replace(f"[{slot}]", word['ewe'])
            eng_sent = eng_sent.replace(f"[{slot}]", word['english'])
        
        synthetic_data.append({'ewe': ewe_sent, 'english': eng_sent})
    
    return pd.DataFrame(synthetic_data)
```

### Monolingual Data Utilization
Leverage monolingual data through self-supervised techniques.

```python
def create_denoising_data(monolingual_texts, noise_function):
    """Create denoising data for self-supervised learning"""
    noisy_texts = [noise_function(text) for text in monolingual_texts]
    return pd.DataFrame({
        'source': noisy_texts,
        'target': monolingual_texts
    })
```

**When to use**:
- Transfer learning when related high-resource languages are available
- Data synthesis when you have linguistic knowledge of the language
- Monolingual data utilization when parallel data is extremely limited

## 5. Preprocessing Pipeline

### Complete Pipeline Example

```python
def preprocess_translation_data(raw_df, src_col='ewe', tgt_col='english'):
    """Complete preprocessing pipeline for translation data"""
    # 1. Basic cleaning
    df = raw_df.copy()
    df[src_col] = df[src_col].apply(clean_text)
    df[tgt_col] = df[tgt_col].apply(clean_text)
    
    # 2. Remove empty rows
    df = df[(df[src_col].str.strip() != '') & (df[tgt_col].str.strip() != '')]
    
    # 3. Length filtering
    df = filter_by_length(df, src_col, tgt_col, min_len=3, max_len=100)
    
    # 4. Remove duplicates
    df = remove_duplicates(df, src_col, tgt_col)
    
    # 5. Train tokenizers
    src_texts = df[src_col].tolist()
    tgt_texts = df[tgt_col].tolist()
    
    # Write texts to files for tokenizer training
    with open('src_corpus.txt', 'w') as f:
        f.write('\n'.join(src_texts))
    with open('tgt_corpus.txt', 'w') as f:
        f.write('\n'.join(tgt_texts))
    
    # Train tokenizers
    src_tokenizer = train_bpe_tokenizer('src_corpus.txt', vocab_size=16000)
    tgt_tokenizer = train_bpe_tokenizer('tgt_corpus.txt', vocab_size=16000)
    
    # 6. Data augmentation (optional)
    augmented_df = augment_translation_data(df, src_col, tgt_col)
    
    # 7. Create train/val/test splits
    train_df, val_df, test_df = create_data_splits(augmented_df)
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'src_tokenizer': src_tokenizer,
        'tgt_tokenizer': tgt_tokenizer
    }
```

## 6. Research Papers and Resources

### Key Papers

1. **Subword Tokenization**:
   - [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf) - Original BPE paper
   - [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf)

2. **Data Augmentation**:
   - [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709.pdf) - Back-translation
   - [Understanding Back-Translation at Scale](https://arxiv.org/pdf/1808.09381.pdf)

3. **Low-Resource NMT**:
   - [Neural Machine Translation for Low-Resource Languages](https://arxiv.org/pdf/1811.00350.pdf)
   - [Massively Multilingual Neural Machine Translation in the Wild](https://arxiv.org/pdf/1907.05019.pdf)

4. **Transfer Learning**:
   - [Transfer Learning for Low-Resource Neural Machine Translation](https://arxiv.org/pdf/1604.02201.pdf)
   - [Multilingual Neural Machine Translation with Knowledge Distillation](https://arxiv.org/pdf/1902.10461.pdf)

### Online Resources

1. [HuggingFace Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
2. [Google's Neural Machine Translation System](https://arxiv.org/pdf/1609.08144.pdf)
3. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
4. [Masakhane - NLP for African Languages](https://www.masakhane.io/)

### Ewe-Specific Resources

1. [Ewe-English Dictionary](https://www.webonary.org/ewe/)
2. [Masakhane Ewe Translation Project](https://github.com/masakhane-io/masakhane-community)
3. [NLLB: No Language Left Behind](https://arxiv.org/pdf/2207.04672.pdf) - Includes Ewe