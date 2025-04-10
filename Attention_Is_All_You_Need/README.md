# Transformer Model for Ewe-English Translation

This directory contains an implementation of the Transformer model based on the "Attention Is All You Need" paper, adapted for Ewe-English translation.

## Files

- `model_utils.py`: Core transformer components (LayerNorm, Encoder, Decoder, etc.)
- `encode_decode.py`: The EncodeDecode model that combines encoder and decoder
- `inference.py`: Inference script for testing the transformer model
- `train_transformer_cli.py`: Command-line script for training the transformer model
- `train_transformer_fixed.ipynb`: Jupyter notebook for training the transformer model
- `visualization.py`: Utilities for visualizing transformer components

## Training the Model

### Using the Command-Line Script

The easiest way to train the model is to use the command-line script:

```bash
# From the project root directory
python train_transformer.py --data_dir ./data/processed --src_lang ewe --tgt_lang english
```

Or you can run the CLI script directly:

```bash
# From the project root directory
python Attention_Is_All_You_Need/train_transformer_cli.py --data_dir ./data/processed --src_lang ewe --tgt_lang english
```

### Command-Line Arguments

- `--data_dir`: Directory with processed data (default: "./data/processed")
- `--src_lang`: Source language (default: "ewe")
- `--tgt_lang`: Target language (default: "english")
- `--tokenizer_type`: Type of tokenizer to use, either "sentencepiece" or "huggingface" (default: "sentencepiece")
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 0.0001)
- `--d_model`: Model dimension (default: 512)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--heads`: Number of attention heads (default: 8)
- `--layers`: Number of encoder/decoder layers (default: 6)
- `--dropout`: Dropout rate (default: 0.1)
- `--max_len`: Maximum sequence length (default: 128)
- `--save_dir`: Directory to save models (default: "./models")

### Using the Jupyter Notebook

Alternatively, you can use the Jupyter notebook:

```bash
# From the project root directory
jupyter notebook Attention_Is_All_You_Need/train_transformer_fixed.ipynb
```

## Inference

To run inference with a trained model:

```bash
python Attention_Is_All_You_Need/inference.py --model-path models/transformer_ewe_english_final.pt --test-file test.txt
```

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you're running the scripts from the project root directory. The scripts add the necessary directories to the Python path, but they need to be run from the correct location.

### Data Preparation

Before training, make sure you have prepared the data and trained the tokenizers. See the `data_processing` directory for scripts to prepare the data and train tokenizers.

### CUDA Out of Memory

If you encounter CUDA out of memory errors, try reducing the batch size, model dimension, or maximum sequence length.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
