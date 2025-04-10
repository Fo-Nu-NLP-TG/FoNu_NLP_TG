# Data Preprocessing Flowcharts and Diagrams

This document provides visual representations of the data preprocessing workflows for Neural Machine Translation, with a focus on Ewe-English translation.

## Table of Contents
- [1. Complete Preprocessing Pipeline](#1-complete-preprocessing-pipeline)
- [2. Data Cleaning Workflow](#2-data-cleaning-workflow)
- [3. Tokenization Approaches](#3-tokenization-approaches)
- [4. Data Augmentation Techniques](#4-data-augmentation-techniques)
- [5. Low-Resource Language Strategies](#5-low-resource-language-strategies)

## 1. Complete Preprocessing Pipeline

```mermaid
flowchart TD
    A[Raw Parallel Data] --> B[Data Cleaning]
    B --> C[Length Filtering]
    C --> D[Duplicate Removal]
    D --> E[Tokenization]
    E --> F[Data Augmentation]
    F --> G[Train/Val/Test Split]
    G --> H[Final Processed Dataset]
    
    subgraph "Optional Steps"
        F
    end
```

## 2. Data Cleaning Workflow

```mermaid
flowchart TD
    A[Raw Text] --> B[HTML Tag Removal]
    B --> C[Whitespace Normalization]
    C --> D[Special Character Handling]
    D --> E[Language Detection]
    E --> F[Empty Row Removal]
    F --> G[Clean Text]
```

## 3. Tokenization Approaches

```mermaid
flowchart LR
    A[Text] --> B{Tokenization Method}
    B -->|Word-level| C[Word Tokenizer]
    B -->|Subword| D[BPE/WordPiece/SentencePiece]
    B -->|Character-level| E[Character Tokenizer]
    
    C --> F[Large vocabulary\nOOV issues]
    D --> G[Balanced approach\nHandles rare words]
    E --> H[No OOV issues\nLong sequences]
    
    F --> I[Tokenized Text]
    G --> I
    H --> I
```

## 4. Data Augmentation Techniques

```mermaid
graph TD
    A[Original Parallel Data] --> B[Back-translation]
    A --> C[Word Dropout]
    A --> D[Word Replacement]
    A --> E[Noise Addition]
    
    B --> F[Synthetic Source-Target Pairs]
    C --> F
    D --> F
    E --> F
    
    F --> G[Augmented Dataset]
```

### Back-translation Process

```mermaid
sequenceDiagram
    participant O as Original Data
    participant R as Reverse Model
    participant A as Augmented Data
    
    O->>R: Target sentences
    R->>A: Generate synthetic source
    A->>A: Combine with original targets
    A->>A: Add to training data
```

## 5. Low-Resource Language Strategies

```mermaid
mindmap
    root((Low-Resource NMT))
        Transfer Learning
            Related high-resource languages
            Cross-lingual embeddings
            Language tagging
        Data Synthesis
            Template-based generation
            Rule-based generation
            Dictionary-based substitution
        Monolingual Data
            Denoising autoencoding
            Back-translation
            Self-supervised learning
```

### Transfer Learning Workflow

```mermaid
flowchart TD
    A[Ewe-English Data] --> C[Combined Dataset]
    B[Related Language Pairs] --> C
    C --> D[Add Language Tags]
    D --> E[Train Multilingual Model]
    E --> F[Fine-tune on Ewe-English]
```