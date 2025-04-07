# Scaled Dot-Product Attention: Step-by-Step

Inputs:
- Query (Q): What you’re asking about (e.g., "sleeps"). Dimension: dk (e.g., 64 numbers per head in multi-head attention).
- Keys (K): All words to compare against (e.g., "The," "cat," "sleeps"). Also dk.
- Values (V): Information about those words (e.g., their meanings). Dimension: dv (often same as dk).

In practice, these are matrices (grids) because the Transformer processes many queries at once (e.g., all words in a sentence).

How It Works

1. Dot Product:
Measure how similar the query is to each key by computing their dot product (a mathematical operation).

Example: If "sleeps" (Q = [1, 2]) and "cat" (K = [2, 1]) are vectors:
Dot product = (1 * 2) + (2 * 1) = 2 + 2 = 4.

Higher number = more similar.

2. Scale by √dk:
Divide the dot product by the square root of dk (e.g., √64 = 8).

Why? Big dot products can get out of control (especially with large dk), making later steps unstable. Scaling keeps things manageable.

So: 4 / √2 = 2.83 (if dk = 2).

3. Softmax:
Turn these scaled scores into weights that sum to 1 (like probabilities).

Example: Scores for "The," "cat," "sleeps" might be [0.5, 2.83, 1.0].

After softmax: [0.1, 0.7, 0.2] (70% weight on "cat").

This is what log_softmax in the Generator relates to, but here it’s regular softmax.

4. Weighted Sum:
Multiply each value by its weight and add them up.

Example: If values are:
"The" = [0.1, 0.2], "cat" = [0.5, 0.6], "sleeps" = [0.3, 0.4],

Output = (0.1 * [0.1, 0.2]) + (0.7 * [0.5, 0.6]) + (0.2 * [0.3, 0.4]).

Result: A new vector combining them, emphasizing "cat."

# Matrix Form

Instead of one query, process a whole sentence at once:
Q: Matrix of all queries (rows = words, columns = dk).

K: Matrix of all keys.

V: Matrix of all values.

Attention(Q, K, V) = softmax((Q @ K.T) / √dk) @ V
@ is matrix multiplication in PyTorch.
K.T is the transpose of K (flips rows and columns).


# Intuitive Analogy

Imagine you’re at a party:
Query: You ask, "Who likes cats?"

Keys: Everyone’s interests (e.g., "dogs," "cats," "food").

Values: Their stories.

Dot Product: How well their interests match your question.

Scale: Keep scores reasonable (no one yells too loud).

Softmax: Decide who to listen to most (e.g., 70% cat person).

Output: A mix of stories, weighted by relevance.

# Additive Attention

What It Is: An older attention mechanism (e.g., from Bahdanau et al., 2014, for RNNs).

How It Works:

1. Query, Keys, Values: Same as before.

2. Additive Combination: Instead of dot product, combine query and key with a neural network (e.g., tanh(WqQ + WkK + b)).

3. Softmax: Turn these scores into weights.

4. Weighted Sum: Same as before.

- Formula (simplified):

score(Q, K) = W @ (Q + K)  # W is a learned weight matrix
Attention(Q, K, V) = softmax(score(Q, K)) @ V