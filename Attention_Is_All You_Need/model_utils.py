import copy
import torch
import torch.nn as nn
import pandas as pd
import altair as alt
from torch.nn.functional import log_softmax

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#### GENERATOR ###########
## The Generator turns the decoder’s output into probabilities over words.
## Probabilities are used to select the next word in the sequence.
## The Generator is a linear layer followed by a softmax function.

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab) # Liner layer maps the model's hidden state to the vocabulary size.
        # d_model: Size of the input (e.g., 512 numbers per word).
        # vocab: Size of the output (e.g., 10,000 words in the vocabulary).

    def forward(self, x):
        """Perform the forward pass of the generator module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear projection and softmax activation.
        """
        return log_softmax(self.proj(x), dim=-1) # Output Probabilities
        # log_softmax: Converts raw scores into probabilities (sums to 1).

### LAYER NORMALIZATION ###########
## Keeps the numbers in a reasonable range, stable.

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """Perform the forward pass of the layer normalization module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying layer normalization.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

######## ENCODER ########### N = 6

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # Stack N Identical Layers
        self.norm = LayerNorm(layer.size) # LayerNorm: Normalizes the output of each layer.
        # LayerNorm: Adjusts numbers so they’re not too big or small (like tuning a radio).

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask) # Pass the input through each layer.
        return self.norm(x)
    
#### SUBLAYERS ###########
# Normalizes the input (LayerNorm). 
# Applies the sublayer (e.g., attention).
# Adds the original input back (residual connection).
# Applies dropout (randomly ignores some numbers to prevent overfitting).

class SublayerConnection(nn.Module):
        """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
        """
        def __init__(self, size, dropout):
            super(SublayerConnection, self).__init__()
            self.norm = LayerNorm(size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, sublayer):
            """Apply residual connection to any sublayer with the same size."""
            return x + self.dropout(sublayer(self.norm(x)))
        # Deep networks (like 6-layer encoders) can "forget" early information as data passes through. 
        # Adding the input back (a "residual connection") preserves it, making training easier.


        
        # Each layer has two sub-layers. The first is a multi-head self-attention mechanism, 
        # and the second is a simple, position-wise fully connected feed-forward network.

class EncoderLayer(nn.Module):
            """Encoder is made up of self-attn and feed forward (defined below)"""
            def __init__(self, size, self_attn, feed_forward, dropout):
                super(EncoderLayer, self).__init__()
                self.self_attn = self_attn # self_attn: Self-attention mechanism.
                # Self-Attention: Lets each word "look" at other words in the input 
                # to understand context (e.g., "world" relates to "hello").
                self.feed_forward = feed_forward # Simple Neural Network
                # Feed-Forward: Processes each word independently.
                self.sublayer = clones(SublayerConnection(size, dropout), 2) # Two Sublayers
                # SublayerConnection: Adds the input back to the output 
                # (residual connection) and normalizes it.

                self.size = size

            def forward(self, x, mask):
                """Follow Figure 1 (left) for connections."""
                x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # Self-Attention
                return self.sublayer[1](x, self.feed_forward) # Feed Forward
        
###### DECODER ########### N = 6

class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Pass the input (and mask) through each layer in turn."""
        # Memory: The encoder’s output.
        # tgt_mask: Ensures the decoder only looks at previous words 
        # (e.g., when predicting "mundo", it can’t see "mundo").
        #  Translating "Hello world" to "Hola mundo". When predicting "Hola",
        #  if the decoder looks at "mundo" (the next word), 
        #  it’s not learning to translate—it’s just copying the answer.
        #  Masks act like blindfolds. 
        #  They block the model from seeing parts of the data it shouldn’t,
        #  forcing it to rely on what it’s learned, not the full answer.
        # src_mask: Ensures the decoder doesn’t pay attention to padding.
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
##### ATTENTION MASK ###########

# Masks ensure the model behaves correctly.
# Ensures the decoder predicts words one at a time, without seeing the future.
# [1, 0, 0]  # Predicting "Hola": sees only itself
# [1, 1, 0]  # Predicting "mundo": sees "Hola" but not future
# [1, 1, 1]  # After finishing: sees all

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def example_mask():
    """Create a visualization of the subsequent mask."""
    LS_data = pd.concat(
        [pd.DataFrame(
            {"Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
             "Window": y,
             "Masking": x,
            }
            )
            for y in range(20)
            for x in range(20)
            ]
    )
    return (alt.Chart(LS_data).mark_rect().properties(height=250, width=250).encode(
        x='Window:O',
        y='Masking:O',
        color=alt.Color('Subsequent Mask:Q', scale=alt.Scale(scheme='viridis'))
    ).interactive())

def show_example(example):
    """Display an example visualization."""
    return example.display()

chart = example_mask()
chart.save("mask_visualization.html")
print("Visualization saved to mask_visualization.html")


##### PADDING #####
## "Hello" → [1, 0, 0] (padded to length 3). (Right Padding)
## "Hello world" → [1, 2, 0].
## Padding isn’t meaningful, so the src_mask tells the model to ignore it (0s in the mask).


##### SOFTMAX #####
## The model learns the probabilities during training


#### TRAINING ####
## Training: The model compares its predictions (e.g., "Hola" = 0.9, "Adios" = 0.1) 
# to the correct answer ("Hola") and tweaks itself to improve.

#### PREPROCESSING ####
## Counting unique words, creating a list.

#### HIDDEN STATE #####
## It’s like a "summary" of the word in context.

