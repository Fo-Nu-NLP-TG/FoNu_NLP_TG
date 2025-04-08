
import torch
from model_utils import subsequent_mask, show_example, make_model

def inference_test():
    test_model = make_model(11, 11, 2) # Vocab size 11, 2 layers
    # Input: A sequence of 10 tokens (1 to 10), 
    # representing a "sentence" with vocab size 11 (0 to 10).
    test_model.eval() # Evaluation mode (no dropout)
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # Input sequence
    # Allows the encoder to see all 10 positions (no padding).
    src_mask = torch.ones(1, 1, 10) # No padding, all positions visible

    # Memory: The encoder’s output, a contextual representation of the input.
    memory = test_model.encode(src, src_mask)
    # Passes src through src_embed (embedding + positional encoding)
    # Runs it through 2 encoder layers (self-attention + feed-forward).


    # Decoder: Generates the output sequence one token at a time.
    # Starts with a single token (0) and builds the output step by step.
    ys = torch.zeros(1, 1).type_as(src)
    # Goal: Generate a 10-token output sequence, one token at a time.
    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1) # Picks the token with the highest probability.
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


if __name__ == "__main__":
    # Run the tests directly instead of using show_example
    run_tests()