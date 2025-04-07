# Standard library imports
import os                  # Operating system interface for file/directory operations
import time                # Time access and conversions for benchmarking
from os.path import exists # Check if files/directories exist

# PyTorch core imports
import torch               # Main PyTorch deep learning library
import torch.nn as nn      # Neural network modules (layers, loss functions)
import torch.nn.functional as F  # Functional interface (activation functions, etc.)
from torch.nn.functional import log_softmax, pad  # Specific functions for NLP tasks

# Math and data manipulation
import math                # Mathematical functions
import copy                # Deep copying objects
import pandas as pd        # Data manipulation and analysis

# Visualization
import altair as alt       # Declarative statistical visualization library

# PyTorch data handling and text processing
from torch.utils.data import DataLoader            # Batch data loading utility
from torch.utils.data.distributed import DistributedSampler  # For distributed training
# Distributed training is the process of training machine learning algorithms using several machines.
from torch.optim.lr_scheduler import LambdaLR     # Learning rate scheduler
# Speed at which the neural network learning rate changes over time
from torchtext.data.functional import to_map_style_dataset  # Dataset conversion
from torchtext.vocab import build_vocab_from_iterator  # Vocabulary building
import torchtext.datasets as datasets              # Standard NLP datasets

# NLP processing
import spacy               # Industrial-strength NLP library for tokenization

# Distributed training
import torch.distributed as dist      # Distributed communication
import torch.multiprocessing as mp    # Multiprocessing for PyTorch
from torch.nn.parallel import DistributedDataParallel as DDP  # Distributed model wrapper

# Hardware monitoring
import GPUtil               # GPU monitoring utility

# Miscellaneous
import warnings             # Warning control for suppressing warnings


class EncodeDecode(nn.Module):
    """
    EncodeDecode is a base class for encoder-decoder architectures in sequence-to-sequence models.
    It provides a structure for models that need to encode input sequences and decode them into output sequences.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Initializes the EncodeDecode model.

        Args:
            encoder: The encoder module.
            decoder: The decoder module.
            src_embed: The embedding layer for the source sequence.
            tgt_embed: The embedding layer for the target sequence.
            generator: The generator module for producing the output.
        """
        super(EncodeDecode, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Forward pass of the EncodeDecode model.

        Args:
            src: The source sequence.
            tgt: The target sequence.
            src_mask: The mask for the source sequence.
            tgt_mask: The mask for the target sequence.

        Returns:
            The output of the model.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.

        Args:
            src: The source sequence.
            src_mask: The mask for the source sequence.

        Returns:
            The encoded representation of the source sequence.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence.

        Args:
            memory: The encoded representation of the source sequence.
            src_mask: The mask for the source sequence.
            tgt: The target sequence.
            tgt_mask: The mask for the target sequence.

        Returns:
            The decoded representation of the target sequence.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    class Generator(nn.Module):
        """
        The Generator module is responsible for producing the output from the model's hidden state.
        It typically involves a linear transformation followed by a softmax activation.
        """

        def __init__(self, d_model, vocab):
            """
            Initializes the Generator module.

            Args:
                d_model: The dimensionality of the model's hidden state.
                vocab: The size of the vocabulary.
            """
            super(Generator, self).__init__()
            self.proj = nn.Linear(d_model, vocab)

        def forward(self, x):
            """
            Forward pass of the Generator module.

            Args:
                x: The input tensor.

            Returns:
                The output tensor after applying the linear transformation and softmax activation.
            """
            return log_softmax(self.proj(x), dim=-1)
        
        ###### Encoder & Decoder Stacks

        def clones(module, N):
            """
            Creates N identical layers using the provided module.

            Args:
                module: The module to be cloned.
                N: The number of clones to create.

            Returns:
                A list of N identical modules.
            """
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
        
        class Encoder(nn.Module):
            """
            The Encoder module is responsible for encoding the input sequence.
            It typically involves applying a series of layers to the input.
            """

            def __init__(self, layer, N):
                """
                Initializes the Encoder module.

                Args:
                    layer: The layer to be applied in the encoder.
                    N: The number of layers in the encoder.
                """
                super(Encoder, self).__init__()
                self.layers = clones(layer, N)
                self.norm = LayerNorm(layer.size)

            def forward(self, x, mask):
                """
                Forward pass of the Encoder module.

                Args:
                    x: The input tensor.
                    mask: The mask for the input sequence.

                Returns:
                    The encoded representation of the input sequence.
                """
                for layer in self.layers:
                    x = layer(x, mask)
                return self.norm(x)
            
        class LayerNorm(nn.Module):
            """
            The LayerNorm module applies layer normalization to the input tensor.
            It typically involves normalizing the input across the feature dimension.
            """ 

            def __init__(self, features, eps=1e-6):
                """
                Initializes the LayerNorm module.

                Args:
                    features: The number of features in the input tensor.
                    eps: A small value added to the denominator for numerical stability.
                """
                super(LayerNorm, self).__init__()
                self.a_2 = nn.Parameter(torch.ones(features))
                self.b_2 = nn.Parameter(torch.zeros(features))
                self.eps = eps
            
            def forward(self, x):
                """
                Forward pass of the LayerNorm module.

                Args:
                    x: The input tensor.

                Returns:
                    The normalized input tensor.
                """
                mean = x.mean(-1, keepdim=True)
                std = x.std(-1, keepdim=True)
                return self.a_2 * (x - mean) / (std + self.eps) + self.b_2