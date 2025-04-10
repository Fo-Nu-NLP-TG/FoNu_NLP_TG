# Transformer and NMT Model Training Pipelines

This document provides visual explanations of training pipelines for various neural machine translation models, with a focus on transformer architectures.

## Table of Contents
- [Transformer Training Pipeline](#transformer-training-pipeline)
- [BERT Pre-training Pipeline](#bert-pre-training-pipeline)
- [mBART Fine-tuning Pipeline](#mbart-fine-tuning-pipeline)
- [T5 Training Pipeline](#t5-training-pipeline)
- [Low-Resource NMT Pipeline](#low-resource-nmt-pipeline)

## Transformer Training Pipeline

```mermaid
flowchart TD
    A[Raw Parallel Data] --> B[Data Cleaning]
    B --> C[Train/Val/Test Split]
    C --> D[Tokenization]
    D --> E[Create Vocabulary]
    E --> F[Prepare Batches]
    
    F --> G[Initialize Transformer]
    G --> H[Forward Pass]
    H --> I[Calculate Loss]
    I --> J[Backward Pass]
    J --> K[Update Parameters]
    
    K --> L{Convergence?}
    L -->|No| H
    L -->|Yes| M[Save Model]
    M --> N[Evaluate on Test Set]
    
    subgraph "Data Preparation"
        A
        B
        C
        D
        E
        F
    end
    
    subgraph "Training Loop"
        G
        H
        I
        J
        K
        L
    end
    
    subgraph "Evaluation"
        M
        N
    end
```

### Key Components

1. **Data Preparation**:
   - Clean and normalize text
   - Split into train/validation/test sets
   - Train tokenizers (SentencePiece or Hugging Face)
   - Build vocabulary for source and target languages

2. **Transformer Architecture**:
   - Encoder: Self-attention + Feed-forward
   - Decoder: Self-attention + Cross-attention + Feed-forward
   - Multi-head attention mechanism
   - Positional encoding

3. **Training Process**:
   - Teacher forcing (using ground truth as input)
   - Label smoothing regularization
   - Learning rate scheduling with warmup
   - Gradient clipping

## BERT Pre-training Pipeline

```mermaid
flowchart TD
    A[Monolingual Data] --> B[Data Cleaning]
    B --> C[Tokenization]
    C --> D[Create Training Examples]
    D --> E[Masked Language Modeling]
    E --> F[Next Sentence Prediction]
    
    F --> G[Initialize BERT]
    G --> H[Forward Pass]
    H --> I[Calculate MLM & NSP Loss]
    I --> J[Backward Pass]
    J --> K[Update Parameters]
    
    K --> L{Convergence?}
    L -->|No| H
    L -->|Yes| M[Save Pre-trained Model]
    
    subgraph "Data Preparation"
        A
        B
        C
        D
        E
        F
    end
    
    subgraph "Pre-training Loop"
        G
        H
        I
        J
        K
        L
    end
```

## mBART Fine-tuning Pipeline

```mermaid
flowchart TD
    A[Parallel Data] --> B[Data Cleaning]
    B --> C[Train/Val/Test Split]
    C --> D[Tokenization with mBART Tokenizer]
    D --> E[Prepare Batches]
    
    F[Pre-trained mBART] --> G[Initialize Fine-tuning]
    E --> G
    G --> H[Forward Pass]
    H --> I[Calculate Loss]
    I --> J[Backward Pass]
    J --> K[Update Parameters]
    
    K --> L{Convergence?}
    L -->|No| H
    L -->|Yes| M[Save Fine-tuned Model]
    M --> N[Evaluate on Test Set]
```

## T5 Training Pipeline

```mermaid
flowchart TD
    A[Parallel Data] --> B[Data Cleaning]
    B --> C[Train/Val/Test Split]
    C --> D[Format as Text-to-Text]
    D --> E[Tokenization with T5 Tokenizer]
    E --> F[Prepare Batches]
    
    G[Pre-trained T5] --> H[Initialize Fine-tuning]
    F --> H
    H --> I[Forward Pass]
    I --> J[Calculate Loss]
    J --> K[Backward Pass]
    K --> L[Update Parameters]
    
    L --> M{Convergence?}
    M -->|No| I
    M -->|Yes| N[Save Fine-tuned Model]
    N --> O[Evaluate on Test Set]
```

## Low-Resource NMT Pipeline

```mermaid
flowchart TD
    A[Limited Parallel Data] --> B[Data Cleaning]
    B --> C[Data Augmentation]
    C --> D[Train/Val/Test Split]
    D --> E[Tokenization]
    E --> F[Prepare Batches]
    
    G[Related Language Model] --> H[Transfer Learning]
    F --> H
    H --> I[Forward Pass]
    I --> J[Calculate Loss]
    J --> K[Backward Pass]
    K --> L[Update Parameters]
    
    L --> M{Convergence?}
    M -->|No| I
    M -->|Yes| N[Save Model]
    N --> O[Evaluate on Test Set]
    
    subgraph "Data Enhancement"
        A
        B
        C
        D
        E
        F
    end
    
    subgraph "Training with Transfer"
        G
        H
        I
        J
        K
        L
        M
    end
```

### Key Techniques for Low-Resource Languages

1. **Data Augmentation**:
   - Back-translation
   - Data synthesis
   - Paraphrasing

2. **Transfer Learning**:
   - From related high-resource languages
   - From multilingual pre-trained models

3. **Regularization**:
   - Stronger dropout
   - Label smoothing
   - Early stopping

4. **Architecture Modifications**:
   - Shared encoders
   - Adapter modules
   - Smaller model variants