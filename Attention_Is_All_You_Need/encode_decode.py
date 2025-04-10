import torch.nn as nn
import os
import sys

# Try different import approaches to handle both local and Kaggle environments
try:
    # Try direct import first
    from model_utils import Generator
except ImportError:
    # If that fails, try with the full path
    try:
        from Attention_Is_All_You_Need.model_utils import Generator
    except ImportError:
        # If that also fails, try to find the file and add its path
        import glob
        model_utils_files = glob.glob("**/model_utils.py", recursive=True)
        if model_utils_files:
            model_dir = os.path.dirname(model_utils_files[0])
            sys.path.append(model_dir)
            from model_utils import Generator
        else:
            raise ImportError("Could not find model_utils.py")

class EncodeDecode(nn.Module):
    """EncodeDecode is a base class for encoder-decoder architectures in sequence-to-sequence models."""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator: Generator):
        super(EncodeDecode, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Perform the forward pass of the encoder-decoder model.

        Args:
            src (torch.Tensor): Source sequence.
            tgt (torch.Tensor): Target sequence.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Output of the model.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        # encode(src, src_mask): Turns the input into a "memory" 
        # (a numerical representation).
        # decode(memory, src_mask, tgt, tgt_mask): 
        # Uses that memory to generate the output.
        # Masks are like blindfolds, ensuring the model only sees what it’s supposed to.

    def encode(self, src, src_mask):
        """Encode the source sequence.

        Args:
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor): Source mask.

        Returns:
            torch.Tensor: Encoded representation of the source sequence.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """Decode the target sequence.

        Args:
            memory (torch.Tensor): Encoded representation of the source sequence.
            src_mask (torch.Tensor): Source mask.
            tgt (torch.Tensor): Target sequence.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Decoded representation of the target sequence.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
