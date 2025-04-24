# Layer Normalization Diagrams

## Basic Normalization Concept

```mermaid
graph TD
    A[Raw Data] --> B[Normalization Process]
    B --> C[Normalized Data]
    
    subgraph "Before Normalization"
    D[Feature 1: Values 0-1000]
    E[Feature 2: Values 0-1]
    F[Feature 3: Values -50 to 50]
    end
    
    subgraph "After Normalization"
    G[Feature 1: Values -1 to 1]
    H[Feature 2: Values -1 to 1]
    I[Feature 3: Values -1 to 1]
    end
    
    D --> B
    E --> B
    F --> B
    B --> G
    B --> H
    B --> I
```

## Internal Covariate Shift

```mermaid
graph TD
    A[Input Layer] --> B[Hidden Layer 1]
    B --> C[Hidden Layer 2]
    C --> D[Hidden Layer 3]
    D --> E[Output Layer]
    
    subgraph "Without Normalization"
    F[Distribution shifts between layers]
    G[Training becomes unstable]
    H[Slower convergence]
    end
    
    subgraph "With Layer Normalization"
    I[Stable distribution between layers]
    J[More stable training]
    K[Faster convergence]
    end
    
    B --> F
    F --> G
    G --> H
    
    B --> I
    I --> J
    J --> K
```

## Layer Normalization vs Batch Normalization

```mermaid
graph TD
    A[Input Data] --> B[Batch Normalization]
    A --> C[Layer Normalization]
    
    subgraph "Batch Normalization"
    D[Normalize across batch dimension]
    E[Depends on batch size]
    F[Statistics computed across batch]
    end
    
    subgraph "Layer Normalization"
    G[Normalize across feature dimension]
    H[Independent of batch size]
    I[Statistics computed per sample]
    end
    
    B --> D
    D --> E
    E --> F
    
    C --> G
    G --> H
    H --> I
```

## Layer Normalization in Transformer

```mermaid
graph TD
    A[Input Embedding] --> B[Add Positional Encoding]
    B --> C[Layer Norm]
    C --> D[Multi-Head Attention]
    D --> E[Add & Layer Norm]
    E --> F[Feed Forward Network]
    F --> G[Add & Layer Norm]
    G --> H[Output]
    
    subgraph "Layer Normalization Details"
    I[Calculate Mean]
    J[Calculate Variance]
    K[Normalize: (x-mean)/sqrt(var+eps)]
    L[Scale and Shift: γ*x + β]
    end
    
    C --> I
    I --> J
    J --> K
    K --> L
```

## Pre-LayerNorm vs Post-LayerNorm

```mermaid
graph TD
    subgraph "Post-LayerNorm (Original)"
    A1[Input] --> B1[Sublayer]
    B1 --> C1[Add]
    A1 --> C1
    C1 --> D1[LayerNorm]
    D1 --> E1[Output]
    end
    
    subgraph "Pre-LayerNorm (Improved)"
    A2[Input] --> B2[LayerNorm]
    B2 --> C2[Sublayer]
    C2 --> D2[Add]
    A2 --> D2
    D2 --> E2[Output]
    end
```

## Layer Normalization Computation Flow

```mermaid
graph TD
    A[Input Tensor x] --> B[Calculate Mean μ]
    A --> C[Calculate Variance σ²]
    B --> D[Normalize: (x-μ)/sqrt(σ²+ε)]
    C --> D
    D --> E[Apply Scale γ]
    E --> F[Apply Shift β]
    F --> G[Output Tensor y]
    
    subgraph "Learnable Parameters"
    H[Scale Parameter γ]
    I[Shift Parameter β]
    end
    
    H --> E
    I --> F
```
