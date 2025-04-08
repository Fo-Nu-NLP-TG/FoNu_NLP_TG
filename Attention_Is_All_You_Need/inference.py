import torch
import sys
import os
import importlib.util

# Get the absolute path to the model_utils.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, 'model_utils.py')

# Import the module dynamically
spec = importlib.util.spec_from_file_location('model_utils', module_path)
model_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_utils)

# Get the required functions from the module
subsequent_mask = model_utils.subsequent_mask
show_example = model_utils.show_example
make_model = model_utils.make_model

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

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
