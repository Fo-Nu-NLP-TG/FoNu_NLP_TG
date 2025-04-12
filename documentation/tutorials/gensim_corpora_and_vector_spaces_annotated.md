# Annotated Tutorial: Corpora and Vector Spaces in Gensim

This is an annotated version of the [Gensim Corpora and Vector Spaces tutorial](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py), with additional explanations and connections to our FoNu_NLP_TG project.

## Introduction

Gensim is a powerful library for topic modeling, document indexing, and similarity retrieval. This tutorial explores how to convert documents to vector space representations, which is a fundamental step in many NLP pipelines, including our transformer-based translation system.

## From Strings to Vectors

```python
from gensim import corpora
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]
```

**Annotation:** In our Ewe-English translation project, we similarly start with raw text documents. The difference is that we work with parallel texts in two languages rather than a single corpus.

## Tokenization and Stopword Removal

```python
# Remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# Remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]
```

**Annotation:** This simple preprocessing is similar to what we do in `data_processing/tokenizer_trainer.py`, though we use more sophisticated tokenization methods like SentencePiece for handling subword units, which is crucial for morphologically rich languages like Ewe.

## Creating a Dictionary

```python
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
print(dictionary.token2id)
```

**Annotation:** In our project, we create vocabulary dictionaries for both source and target languages. These dictionaries map tokens to IDs, which are then used by the transformer model. Our implementation can be found in the `data_processing` module.

## Document to Vector Transformation

```python
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk
```

**Annotation:** This transformation from documents to sparse vectors is conceptually similar to how we prepare data for our transformer model, though we use dense embeddings rather than sparse bag-of-words representations. The transformer's embedding layer converts token IDs to dense vectors.

## Corpus Streaming - One Document at a Time

```python
class MyCorpus:
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
```

**Annotation:** This memory-efficient approach is crucial for our project as well. In `data_processing/dataset.py`, we implement similar streaming approaches to handle large datasets without loading everything into memory at once.

## Compatibility with NumPy and SciPy

```python
import gensim
import numpy as np
numpy_matrix = np.random.randint(10, size=[5,2])  # random matrix as an example
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)

import scipy.sparse
scipy_sparse_matrix = scipy.sparse.random(5,2)  # random sparse matrix as example
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
```

**Annotation:** While we primarily use PyTorch tensors in our project, these conversion utilities are useful for data preprocessing and analysis. Our transformer model works with dense matrices, but understanding sparse representations is valuable for efficient data handling.

## Relevance to FoNu_NLP_TG

This tutorial demonstrates fundamental concepts in text processing that are relevant to our project:

1. **Document representation**: Converting text to numerical form
2. **Vocabulary management**: Creating and using dictionaries
3. **Memory efficiency**: Streaming data to handle large corpora
4. **Matrix operations**: Working with different numerical representations

In our transformer implementation, we build upon these concepts with more sophisticated approaches:

- **Subword tokenization**: Using SentencePiece instead of simple word splitting
- **Dense embeddings**: Using learned embeddings rather than sparse vectors
- **Attention mechanisms**: Going beyond bag-of-words to capture contextual relationships
- **Parallel corpora**: Working with aligned text in two languages

## Next Steps

To see how we implement these concepts in our project, explore:

- `data_processing/tokenizer_trainer.py` for tokenization
- `data_processing/dataset.py` for data handling
- `Attention_Is_All_You_Need/model_utils.py` for embedding layers