
from model_utils import subsequent_mask
import time

# This file provides utilities for training a Transformer-like model
# Batch handling: Prepares source and target sequences with appropriate masks for training.
# Training loop: Manages the training process, 
# including forward passes, loss computation, backpropagation, and optimization
# Learning rate scheduling: Implements a custom learning rate policy tailored 
# for Transformer models.
# Progress tracking: Keeps tabs on training statistics like loss and tokens processed.


#  A batch object that holds the src and target sentences for training,
#  as well as constructing the masks.
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

# Generic training and scoring function to keep track of loss

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

    def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1, # Number of iterations to accumulate gradients before updating weights
    train_state=TrainState(),):
        """Train a single epoch"""
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        for i, batch in enumerate(data_iter):
        # For each batch, passes the batch through the model, computes the loss, and updates the model's parameters.
            out = model.forward(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            # Compute the loss using loss_compute.
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            # loss_node = loss_node / accum_iter
            if mode == "train" or mode == "train+log":
                loss_node.backward() # Backpropagates the loss
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens
                if i % accum_iter == 0:  
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                scheduler.step() # Adjusts the learning rate.

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 40 == 1 and (mode == "train" or mode == "train+log"):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                start = time.time()
                tokens = 0
            del loss
            del loss_node
        return total_loss / total_tokens, train_state

### RATE #######

# Define a learning rate schedule
# step : current training step
# model_size : The size of the model 
# factor : A scaling factor for the learning rate
# warmup : The number of warmup steps during which the learning rate increases linearly

def rate(step, model_size, factor, warmup):
    """ we have to default the step to  for LambdaLR function 
    to avoid zero raising to negative power."""
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

