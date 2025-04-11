# Mini-Course: Understanding Attention Scores in Transformers

## Introduction

Attention scores are at the heart of the attention mechanism, the key innovation that made Transformer models so powerful. This mini-course explains what attention scores are, how they're calculated, and why they're so important in natural language processing.

## Table of Contents

1. [What is Attention?](#what-is-attention)
2. [Calculating Attention Scores](#calculating-attention-scores)
3. [Visualizing Attention Scores](#visualizing-attention-scores)
4. [Multi-Head Attention](#multi-head-attention)
5. [Impact of Attention Scores](#impact-of-attention-scores)
6. [Practical Exercises](#practical-exercises)

## What is Attention?

Attention is a mechanism that allows a model to focus on certain parts of an input sequence when processing a specific element. It's similar to how you might focus on certain key words when reading a complex sentence to understand its overall meaning.

In Transformer models, attention enables:
- Capturing long-range dependencies between words
- Processing sequences in parallel (unlike RNNs)
- Creating rich contextual representations for each word

## Calculating Attention Scores

Attention scores are calculated in three main steps:

### 1. Projecting Query, Key, and Value Vectors

Each word in the sequence is first transformed into three different vectors:
- **Query (Q)**: What we're looking for
- **Key (K)**: What we're comparing against
- **Value (V)**: What we're retrieving

These vectors are obtained by linear projection of the initial embedding vector:

```
Q = W_Q * X
K = W_K * X
V = W_V * X
```

Where X is the embedding vector and W_Q, W_K, W_V are learnable weight matrices.

### 2. Computing Compatibility Scores

Compatibility scores are calculated by taking the dot product between each Query and all Keys:

```
Score = Q * K^T
```

This score indicates how relevant each word (represented by its Key) is to the current word (represented by its Query).

### 3. Normalization and Weighting

The scores are then normalized by the square root of the dimension of the Key vectors to stabilize learning:

```
Normalized_Score = Score / √d_k
```

Then, a softmax function is applied to obtain a probability distribution:

```
Attention_weights = softmax(Normalized_Score)
```

Finally, these weights are used to calculate a weighted sum of the Value vectors:

```
Output = Attention_weights * V
```

The complete formula for the attention mechanism is therefore:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

## Visualizing Attention Scores

Attention scores can be visualized as heatmaps to understand how the model focuses on different parts of the sequence.

Example visualization for the sentence "The cat sleeps on the mat":

```
   | The | cat | sleeps | on | the | mat |
---+-----+-----+--------+----+-----+-----|
The | 0.7 | 0.1 |  0.05  |0.05| 0.1 | 0.0 |
cat | 0.1 | 0.6 |  0.2   |0.05| 0.0 | 0.05|
sleeps| 0.0 | 0.3 |  0.5   |0.1 | 0.0 | 0.1 |
on  | 0.0 | 0.1 |  0.1   |0.6 | 0.1 | 0.1 |
the | 0.1 | 0.0 |  0.0   |0.1 | 0.3 | 0.5 |
mat | 0.0 | 0.05|  0.05  |0.1 | 0.2 | 0.6 |
```

In this visualization, each cell represents the attention weight between the word in the row and the word in the column. The higher the value (and the darker the color), the stronger the attention.

## Multi-Head Attention

In practice, Transformers use Multi-Head Attention, which involves running several attention mechanisms in parallel, each with its own projection matrices W_Q, W_K, and W_V.

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) * W_O

where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
```

Advantages of Multi-Head Attention:
- Each head can focus on different aspects of the sequence
- Some heads may capture syntax, others semantics
- Improves the model's ability to model complex relationships

## Impact of Attention Scores

Attention scores have a major impact on the performance of Transformer models:

1. **Contextualization**: They allow for creating contextual representations of words, taking into account their environment
2. **Disambiguation**: They help resolve ambiguities by focusing on relevant words
3. **Translation**: In machine translation, they implicitly align words between source and target languages
4. **Interpretability**: They provide insight into how the model "reasons" about text

## Practical Exercises

### Exercise 1: Manual Calculation of Attention Scores

Consider a sequence of 3 words with simplified embedding vectors of dimension 2:
- Word 1: [1, 0]
- Word 2: [0, 1]
- Word 3: [1, 1]

With simplified projection matrices:
- W_Q = [[1, 0], [0, 1]]
- W_K = [[1, 0], [0, 1]]
- W_V = [[1, 0], [0, 1]]

Calculate the attention scores for this sequence.

### Exercise 2: Analyzing an Attention Map

Observe the following attention map for the translation of "The cat sits on the mat" to "Le chat est assis sur le tapis":

```
            | The | cat | sits | on  | the | mat |
------------+-----+-----+------+-----+-----+-----|
Le          | 0.8 | 0.1 | 0.0  | 0.0 | 0.1 | 0.0 |
chat        | 0.1 | 0.8 | 0.1  | 0.0 | 0.0 | 0.0 |
est         | 0.0 | 0.1 | 0.4  | 0.3 | 0.1 | 0.1 |
assis       | 0.0 | 0.1 | 0.7  | 0.1 | 0.0 | 0.1 |
sur         | 0.0 | 0.0 | 0.1  | 0.8 | 0.1 | 0.0 |
le          | 0.1 | 0.0 | 0.0  | 0.1 | 0.7 | 0.1 |
tapis       | 0.0 | 0.0 | 0.0  | 0.1 | 0.1 | 0.8 |
```

Questions:
1. Which words are strongly linked in the translation?
2. Are there any words that pay attention to multiple source words?
3. How does attention help in correctly translating "sits" to "est assis" (two words in French)?

## Conclusion

Attention scores are a fundamental mechanism that has transformed the field of NLP. By allowing models to dynamically focus on different parts of a sequence, they have paved the way for more powerful and interpretable architectures.

In modern Transformer models like BERT, GPT, and T5, attention mechanisms have become even more sophisticated, with variants such as sparse attention, local attention, or attention with structural constraints.

Understanding attention scores is essential for anyone who wants to master the inner workings of modern language models and develop advanced NLP applications.

---

## Additional Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper on Transformers
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Annotated implementation of the Transformer
- [Visualizing Attention in Transformer-Based Language Models](https://towardsdatascience.com/visualizing-attention-in-transformer-based-language-models-9a1d0c2c4c10) - Article on visualizing attention
