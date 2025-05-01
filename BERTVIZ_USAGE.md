# Using BERTViz with FoNu_NLP_TG

This guide explains how to use BERTViz to visualize attention patterns in your transformer model for Ewe-English translation.

## What is BERTViz?

BERTViz is a tool for visualizing attention in transformer models like BERT, GPT-2, and other transformer-based architectures. It provides three visualization types:

1. **Head View**: Visualizes attention patterns of individual attention heads
2. **Model View**: Visualizes attention patterns across all layers and heads
3. **Neuron View**: Visualizes individual neuron activations (not implemented in this integration)

## Prerequisites

BERTViz is already installed in your environment. If you need to install it in a different environment, you can use:

```bash
pip install bertviz
```

## Quick Start

The simplest way to visualize attention in your model is to use the `visualize_attention.py` script:

```bash
python visualize_attention.py --src-text "Your Ewe text here" --open-browser
```

This will:
1. Load your trained model from `models/transformer_ewe_english_final.pt`
2. Generate attention visualizations for the provided text
3. Save the visualizations as HTML files in the `bertviz_visualizations` directory
4. Open the visualizations in your browser (if `--open-browser` is specified)

## Advanced Usage

### Visualizing Attention for a Specific Text Pair

```bash
python visualize_attention.py \
  --model-path models/transformer_ewe_english_final.pt \
  --src-text "Your Ewe text here" \
  --tgt-text "Your English translation here" \
  --output-dir my_visualizations \
  --device cuda \
  --open-browser
```

### Using the Enhanced Semantics Visualizer

The `SV_bertviz.py` script combines the Semantics Visualizer (SV) with BERTViz, allowing you to visualize both embeddings and attention patterns:

```bash
python SV_bertviz.py \
  --model-path models/transformer_ewe_english_final.pt \
  --words hello world language model transformer attention \
  --src-text "Your Ewe text here" \
  --tgt-text "Your English translation here" \
  --output-dir enhanced_visualizations \
  --open-browser
```

This will:
1. Visualize embeddings for the specified words using PCA and t-SNE
2. Compare source and target embeddings
3. Generate attention visualizations for the provided text pair
4. Save all visualizations in the specified output directory

## Programmatic Usage

You can also use the BERTViz integration in your own code:

```python
from bertviz_integration import visualize_attention_for_text

# Visualize attention for a text
viz_paths = visualize_attention_for_text(
    model_path='models/transformer_ewe_english_final.pt',
    src_text='Your Ewe text here',
    tgt_text='Your English translation here',
    output_dir='my_visualizations',
    device='cpu'
)

# The viz_paths dictionary contains paths to the generated HTML files
print(viz_paths)
```

Or use the enhanced Semantics Visualizer:

```python
from SV_bertviz import EnhancedSemanticsVisualizer

# Create visualizer
visualizer = EnhancedSemanticsVisualizer(output_dir='my_visualizations')

# Analyze model
results = visualizer.analyze_model(
    model_path='models/transformer_ewe_english_final.pt',
    words=['hello', 'world', 'language', 'model', 'transformer', 'attention'],
    src_text='Your Ewe text here',
    tgt_text='Your English translation here',
    device='cpu'
)

# The results dictionary contains paths to all generated visualizations
print(results)
```

## Integration with Training

You can also integrate BERTViz with your training process to visualize attention patterns during training. This can be done by modifying the `train_with_sv.py` script to include attention visualization after each epoch.

## Understanding the Visualizations

### Head View

The Head View shows attention patterns for each attention head in each layer. It displays a grid where:
- Each row represents a layer in the transformer
- Each column represents an attention head
- Each cell shows the attention pattern for that specific head

### Model View

The Model View aggregates attention across all heads and layers, allowing you to see the overall attention pattern of the model. It provides a more holistic view of how the model attends to different tokens.

## Troubleshooting

If you encounter any issues:

1. Make sure your model checkpoint is accessible and contains the expected state dictionaries
2. Check that your tokenizers are properly loaded
3. Ensure the input text is in the correct language (Ewe for source, English for target)
4. If visualizations don't appear in the browser, try opening the HTML files manually from the output directory

## References

- [BERTViz GitHub Repository](https://github.com/jessevig/bertviz)
- [BERTViz Paper](https://arxiv.org/abs/1906.05714)
