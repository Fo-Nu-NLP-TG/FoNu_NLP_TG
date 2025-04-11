# Quick Start Guide

This guide will help you quickly get started with the FoNu NLP TG project for Ewe-English translation.

## Prerequisites

Make sure you have:

- Completed the [Installation](installation.md) process
- Activated your virtual environment
- Downloaded the pre-trained models (if using them)

## Basic Translation

### Using the Pre-trained Model

To translate a sentence from Ewe to English using the pre-trained model:

```bash
python Attention_Is_All_You_Need/translate.py --text "Ŋdi nyuie"
```

This will output the English translation of the Ewe text "Ŋdi nyuie" (Good morning).

### Interactive Mode

You can also use the interactive mode to translate multiple sentences:

```bash
python Attention_Is_All_You_Need/translate.py --interactive
```

This will start an interactive session where you can type Ewe sentences and get their English translations.

## Training Your Own Model

If you want to train your own model, follow these steps:

### Step 1: Prepare Your Data

First, prepare your parallel data in CSV format:

```bash
python data_processing/prepare_data.py --input-file your_data.csv --output-dir data/processed
```

### Step 2: Train Tokenizers

Train SentencePiece tokenizers for both languages:

```bash
python data_processing/tokenizer_trainer.py --train-file data/processed/ewe_train.txt --model-prefix data/processed/ewe_sp --vocab-size 8000
python data_processing/tokenizer_trainer.py --train-file data/processed/english_train.txt --model-prefix data/processed/english_sp --vocab-size 8000
```

### Step 3: Train the Model

Train the transformer model:

```bash
python Attention_Is_All_You_Need/training.py --train-file data/processed/train.csv --val-file data/processed/val.csv --src-tokenizer data/processed/ewe_sp.model --tgt-tokenizer data/processed/english_sp.model --output-dir models
```

This will train the model and save checkpoints to the `models/` directory.

## Evaluating the Model

To evaluate the model's performance:

```bash
python evaluation/run_evaluation.py --test-data data/processed/test.csv --model-path models/transformer_ewe_english_final.pt --src-tokenizer data/processed/ewe_sp.model --tgt-tokenizer data/processed/english_sp.model
```

This will calculate BLEU scores and other metrics on the test set.

## Visualizing Attention

To visualize the attention patterns in the model:

```bash
python Attention_Is_All_You_Need/visualization.py --model-path models/transformer_ewe_english_final.pt --text "Ŋdi nyuie"
```

This will generate an HTML file with attention visualizations.

## Next Steps

Now that you've got the basics, you can:

- Explore the [Project Structure](../documentation/project-structure.md) to understand the codebase
- Learn about the [Transformer Architecture](../model/transformer-architecture.md) used in the project
- Read our [Research Report](../research/ewe-english-translation.md) on Ewe-English translation
- Check out the [Blog](../blog/introduction.md) for updates and insights
