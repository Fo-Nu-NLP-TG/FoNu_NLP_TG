# Semantics Visualizer (SV)

A powerful tool for visualizing and analyzing word embeddings and semantic relationships in transformer models during training and inference.

## Features

- Visualize pre-trained word embeddings (Word2Vec, GloVe, etc.)
- Extract and visualize embeddings from transformer models
- Track embedding changes during training
- Compare embeddings across different model versions
- Support for both PyTorch and TensorFlow models
- Interactive visualizations with dimensionality reduction (PCA, t-SNE, UMAP)
- Create animations showing how embeddings evolve during training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SV.git
cd SV

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Visualize Pre-trained Embeddings

```python
from SV import SemanticsVisualizer

# Create visualizer
sv = SemanticsVisualizer()

# Load pre-trained embeddings
sv.load_pretrained_embeddings('path/to/embeddings.bin')

# Visualize some words
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'boy', 'girl']
sv.visualize_pretrained(words, method='pca')
```

### Visualize Transformer Embeddings

```python
# Load your transformer model and tokenizer
model = load_model('path/to/model.pt')
tokenizer = load_tokenizer('path/to/tokenizer.model')

# Extract embeddings
words = ['hello', 'world', 'language', 'model', 'transformer']
embeddings = sv.extract_embeddings_from_model(model, tokenizer, words)

# Visualize embeddings
sv.visualize_model_embeddings(embeddings, method='tsne')
```

### Track Embeddings During Training

```python
# Register a hook to track embeddings during training
hook = sv.register_training_hook(model, tokenizer, words)

# In your training loop
for epoch in range(num_epochs):
    train_epoch(...)
    hook(epoch)  # Call the hook after each epoch

# Create an animation after training
sv.create_embedding_animation('embeddings_evolution.gif')
```

## Integration with Transformer Models

The Semantics Visualizer is designed to work seamlessly with transformer models. We provide scripts to integrate it with your training process:

- `transformer_sv_integration.py`: Functions to extract embeddings from transformer models and register training hooks
- `train_with_sv.py`: Extended training script that integrates the Semantics Visualizer

### Training with Semantics Visualization

```bash
python train_with_sv.py \
    --data-dir data/processed \
    --src-lang ewe \
    --tgt-lang english \
    --epochs 10 \
    --sv-output-dir visualizations \
    --track-words hello world language model transformer attention neural network translation
```

## Visualization Methods

The Semantics Visualizer supports multiple dimensionality reduction methods:

- **PCA**: Fast and deterministic, good for initial exploration
- **t-SNE**: Better at preserving local structure, good for finding clusters
- **UMAP**: Preserves both local and global structure, good for detailed analysis

## Examples

### Comparing Source and Target Embeddings

```python
# Extract embeddings from source and target sides
src_embeddings = sv.extract_embeddings_from_model(model, src_tokenizer, words, layer='src_embed')
tgt_embeddings = sv.extract_embeddings_from_model(model, tgt_tokenizer, words, layer='tgt_embed')

# Compare embeddings
sv.compare_embeddings(src_embeddings, tgt_embeddings, labels=('Source', 'Target'))
```

### Creating Embedding Animations

```python
# After training with the registered hook
sv.create_embedding_animation('embeddings_evolution.gif', method='pca')
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- scikit-learn
- matplotlib
- numpy
- (Optional) plotly for interactive 3D visualizations
- (Optional) umap-learn for UMAP dimensionality reduction

## License

MIT License
