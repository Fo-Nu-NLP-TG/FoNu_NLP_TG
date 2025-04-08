# Experiments-On-Transformers (FoNu_NLP_TG)

FoNu_NLP_TG ("Fo Nu" means "speak" in Ewe, and TG stands for Togo) is a research project focused on experimenting, exploring, and fine-tuning transformers, with a special emphasis on applications for Togolese languages.

## Project Blog

We've started a blog to document our progress and share insights about transformer models and NLP. The blog is available in multiple formats:

- [GitHub Pages](https://lemniscate-world.github.io/FoNu_NLP_TG/) (automatically updated)
- [Source files](blog/) in the repository
- Selected posts on [Medium](https://medium.com/) (coming soon)

## Transformer Architecture Standard

1. Encoder: N layers (usually 6) with self-attention and feed-forward networks.
2. Decoder: N layers with self-attention, source-attention (to encoder), and feed-forward networks.
3. Attention: Mechanism to weigh word importance.
4. Forward Pass: Input → Encoder → Memory → Decoder → Output.

## Methods

Standard: Encoder-Decoder with multi-head attention. (Harvard)
Variants: BERT (encoder-only), GPT (decoder-only).
Customization: You can adjust N, hidden size, or attention heads, but the structure is usually fixed.

## Attention Mechanism
- How It Works: Attention calculates "scores" between words. For "Hello world", it checks how much "Hello" relates to "world" using their hidden states.
- Training: The model learns these relationships from data (e.g., "Hello" often precedes "world").
- Multi-Head Attention: Looks at multiple relationships at once (e.g., syntax, meaning).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Experiments-On-Transformers.git
cd Experiments-On-Transformers

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy models (if needed)
python -m spacy download en_core_web_sm
```

## Project Structure

- `Attention_Is_All_You_Need/`: Implementation based on the original paper
  - `Harvard_Annotated_Transformers.py`: Transformer implementation following Harvard NLP's annotated guide
  - `model_utils.py`: Core transformer components (LayerNorm, Encoder, Decoder, etc.)
  - `encode_decode.py`: The EncodeDecode model that combines encoder and decoder
  - `visualization.py`: Utilities for visualizing transformer components
  - `transformer_explanation.ipynb`: Jupyter notebook with visualizations and explanations
- `Tensorflow_Ytb_Channel/`: TensorFlow-based implementations
- `Kaggle_dataset/`: Scripts for working with datasets from Kaggle
- `Ewe_News_Dataset/`: Processing scripts for the Ewe News Dataset
- `SV(Semantics_Visualizer)/`: Tools for visualizing word embeddings
  - `SV.py`: Local implementation of word embedding visualization
  - `kaggle_semantics_visualizer.ipynb`: Kaggle notebook implementation that can be run in the cloud
  - Link to Kaggle notebook: [https://www.kaggle.com/kuroio/semantics-visualizer](https://www.kaggle.com/kuroio/semantics-visualizer)
- `blog/`: Project blog documenting our progress and insights
  - `index.md`: Main blog index page
  - `first_post.md`: Introduction to the project and transformer models
  - `images/`: Directory containing images for blog posts

## Usage Examples

### Basic Transformer Usage

```python
import torch
from Attention_Is_All_You_Need.encode_decode import EncodeDecode
from Attention_Is_All_You_Need.model_utils import Generator, Encoder

# Create a transformer model
model = EncodeDecode(
    encoder=encoder,
    decoder=decoder,
    src_embed=src_embed,
    tgt_embed=tgt_embed,
    generator=generator
)

# Forward pass
output = model(src, tgt, src_mask, tgt_mask)
```

### Visualizing Attention Masks

```python
from Attention_Is_All_You_Need.visualization import visualize_mask, display_chart

# Create and display a visualization of the subsequent mask
mask_chart = visualize_mask()
display_chart(mask_chart, "mask_visualization.html")
```

## Requirements

This project requires Python 3.8+ and the following packages:
- torch>=2.2.0
- torchtext>=0.16.0
- pandas>=1.5.0
- altair>=5.0.0
- spacy>=3.6.0
- matplotlib>=3.5.0
- jupyter>=1.0.0

See `requirements.txt` for the complete list.

## Papers

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/#prelims)
![Transformer Architecture](https://nlp.seas.harvard.edu/images/the_transformer_architecture.jpg)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
