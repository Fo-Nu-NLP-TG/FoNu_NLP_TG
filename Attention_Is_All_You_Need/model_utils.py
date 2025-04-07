import copy
import torch
import torch.nn as nn
import pandas as pd
import altair as alt
import math
from torch.nn.functional import log_softmax
import os
import sys

# Remove the circular import
# from encode_decode import EncodeDecode

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
        # d_model: Size of the input (e.g., 512 numbers per word). || Word Vector Size
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
                # Multi-head self-attention
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
                # splits x into heads, computes attention and recombines.
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


### ATTENTION ###########
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


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
    if callable(example):
        return example().display()
    else:
        return example.display()

chart = example_mask()
chart.save("mask_visualization.html")
print("Visualization saved to mask_visualization.html")


### MULTI-HEAD ATTENTION #######

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        del query
        del key
        del value
        return self.linears[-1](x)

## Feed Forward Network ####

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


### EMBEDDING ####

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)





### POSITIONAL ENCODING ####

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )

class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

show_example(example_positional())


#### FULL MODEL ####

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1) -> "EncodeDecode":
    """Helper: Construct a model from hyperparameters."""
    # Import EncodeDecode here to avoid circular imports
    from encode_decode import EncodeDecode
    
    c = copy.deepcopy
    # It’s modular so attn, ff, and position are reused with deepcopy
    # to avoid sharing weights unintentionally
    # Sharing weights unintentionally happens when the same object (e.g., a layer or module)
    # is reused across parts of a model without copying it, 
    # so changes to one instance affect all others. 
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncodeDecode(
        # A single encoder layer with multi-headed attention (attn), feed-forward network (ff), and dropout.
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # A single decoder layer. It uses two attention mechanisms
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # Embeddings and positional encoding for source and target sequences.
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # A linear layer (and often softmax) that maps the decoder’s d_model-sized outputs 
        # to probabilities over the target vocabulary, predicting the next token.
        Generator(d_model, tgt_vocab))
    # This line runs the forward function on one batch of data to initialize the model.
    for p in model.parameters():
        ## INITIALIZATION ##
        # Model initialization is setting the starting values of a neural network’s
        # internal parameters (weights and biases) before training begins.
        # built-in methods like Xavier or He initialization
        if p.dim() > 1:
            # Skips 1D parameters (e.g., biases), 
            # focusing on weight matrices or higher-dimensional tensors.
            # Initializes the weights of these parameters using the Xavier uniform distribution.
            nn.init.xavier_uniform_(p)
    # Returns the initialized model.
    return model

### SOME OTHER INITIALIZATIONS FUNCTIONS ####
# nn.init.xavier_uniform_(tensor): Xavier with uniform distribution.
# nn.init.xavier_normal_(tensor): Xavier with normal distribution.
# nn.init.kaiming_uniform_(tensor): He initialization (good for ReLU).
# nn.init.constant_(tensor, value): Set all to a constant (e.g., zeros).
# nn.init.normal_(tensor, mean, std): Normal distribution with mean and standard deviation.



# This code builds a standard Transformer (like in "attention is All You Need").
# The encoder processes the source sequence into a rich representation using N layers of attention 
# and feed-forward networks. 
# The decoder generates the target sequence, attending to both its own previous outputs 
# and the encoder’s result. Positional encoding ensures the model knows token order, 
# and embeddings map tokens to vectors. The generator predicts the final output tokens.


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

