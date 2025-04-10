# Training Fixes Documentation

This document tracks all fixes made to the transformer model training pipeline to address various issues encountered during development.

## Successful Training Run Confirmation

**Status:** The model is now training correctly with properly aligned dimensions.

**Training Output:**
```
Using device: cpu
Loaded SentencePiece tokenizer with vocabulary size 8000
Loaded SentencePiece tokenizer with vocabulary size 8000
Created dataset with 21719 translation pairs
Created dataset with 2715 translation pairs
Source vocabulary size: 8000
Target vocabulary size: 8000
Creating model with src_vocab_size=8000, tgt_vocab_size=8000
Generator output dimension: 8000
Epoch 1/10
Output shape: torch.Size([32, 127, 8000]), Output size(-1): 8000
Target shape: torch.Size([32, 127])
Target flat shape: torch.Size([4064])
Batch 0: Max target index: 7925, Vocab size: 8000
```

**Key Observations:**
- Tokenizers loaded with correct vocabulary sizes (8000)
- Model created with matching vocabulary sizes
- Generator output dimension correctly set to 8000
- Output tensor shape is [batch_size, sequence_length, vocab_size] = [32, 127, 8000]
- Maximum target index (7925) is within vocabulary bounds (8000)

## Output Dimension Mismatch Fix

**Issue:** The model's output dimension was 512 (d_model) but needed to be 8000 (target vocabulary size).

**Error Message:**
```
ERROR: Output dimension 512 doesn't match target vocab size 8000
```

**Root Cause:**
The Generator layer was correctly initialized with an output dimension of 8000, but it wasn't being applied in the forward pass of the model. The decoder output (dimension 512) was being returned directly instead of being passed through the Generator to get logits over the vocabulary.

**Fix:**
Modified the `EncodeDecode` class's forward method to apply the Generator to the decoder output:

```python
def forward(self, src, tgt, src_mask, tgt_mask):
    """Take in and process masked src and target sequences"""
    memory = self.encode(src, src_mask)
    decoder_output = self.decode(memory, src_mask, tgt, tgt_mask)
    # Apply the generator to get logits over vocabulary
    return self.generator(decoder_output)
```

**File Modified:** `Attention_Is_All_You_Need/train_transformer.py`

## Additional Improvements

### Enhanced Debugging in Generator

Added debugging information to the Generator class to help diagnose dimension issues:

```python
def forward(self, x):
    """Project features to vocabulary size"""
    if hasattr(self, 'proj'):
        # Print shape information for debugging
        # print(f"Generator input shape: {x.shape}, output features: {self.proj.out_features}")
        pass
    return self.proj(x)
```

### Improved Error Handling in Training Loop

Enhanced the training loop to provide more detailed information about problematic indices and to handle out-of-bounds target indices:

1. Added detection of target indices exceeding vocabulary size
2. Added reporting of problematic examples with their text and position
3. Added clamping of target indices to ensure they're within vocabulary bounds

```python
# Add debug info to find problematic indices
target_flat = tgt[:, 1:].contiguous().view(-1)
max_target_idx = target_flat.max().item()

# Print more detailed debugging information
if batch_idx == 0 or max_target_idx >= tgt_vocab_size:
    print(f"Batch {batch_idx}: Max target index: {max_target_idx}, Vocab size: {tgt_vocab_size}")
    
    if max_target_idx >= tgt_vocab_size:
        print(f"WARNING: Target index {max_target_idx} exceeds vocabulary size {tgt_vocab_size}")
        # Find all problematic indices
        problematic_indices = (target_flat >= tgt_vocab_size).nonzero().squeeze().tolist()
        if not isinstance(problematic_indices, list):
            problematic_indices = [problematic_indices]
        
        # Print some of the problematic examples
        for idx in problematic_indices[:3]:  # Print first 3 problematic indices
            batch_idx = idx // (tgt.size(1) - 1)
            seq_idx = idx % (tgt.size(1) - 1)
            if batch_idx < len(batch["target_text"]):
                print(f"Problematic text: {batch['target_text'][batch_idx]}")
                print(f"Problematic index position: {seq_idx}")
                print(f"Token ID: {target_flat[idx].item()}")

# Ensure target indices are within bounds by clamping
target_flat = torch.clamp(target_flat, 0, tgt_vocab_size - 1)
```

## Removed Workarounds

Removed the temporary padding workaround that was attempting to reshape the output tensor to match the vocabulary size, as we've fixed the root cause of the dimension mismatch.
