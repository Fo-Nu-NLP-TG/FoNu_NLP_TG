# FoNu_NLP_TG Project Structure Documentation

This document provides a detailed explanation of the project structure, components, and how they interact. Use this as a comprehensive guide to understand the codebase organization.

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
  - [Attention_Is_All_You_Need](#attention_is_all_you_need)
  - [SV(Semantics_Visualizer)](#svsemantics_visualizer)
  - [data_processing](#data_processing)
  - [Ewe_News_Dataset](#ewe_news_dataset)
  - [Kaggle_dataset](#kaggle_dataset)
  - [Tensorflow_Ytb_Channel](#tensorflow_ytb_channel)
  - [blog](#blog)
  - [documentation](#documentation)
  - [evaluation](#evaluation)
  - [Research](#research)
  - [tools](#tools)
- [Data Flow](#data-flow)
- [Development Workflow](#development-workflow)
- [Deployment](#deployment)

## Overview

FoNu_NLP_TG is organized around several key components:

1. **Transformer Implementation**: Core implementation of the transformer architecture based on the "Attention Is All You Need" paper
2. **Data Processing**: Tools for preparing, tokenizing, and managing datasets
3. **Visualization Tools**: Components for visualizing model internals and semantic relationships
4. **Ewe-English Translation**: Specialized components for our focus on Ewe-English translation
5. **Documentation & Blog**: Materials explaining our approach and findings

## Core Components

### Attention_Is_All_You_Need

This directory contains our implementation of the transformer architecture as described in the original paper.

#### Files:
- `model_utils.py`: Core building blocks of the transformer
  - `LayerNorm`: Normalization layer used throughout the model
  - `MultiHeadAttention`: Implementation of multi-head attention mechanism
  - `PositionwiseFeedForward`: Feed-forward networks used in encoder/decoder
  - `PositionalEncoding`: Adds positional information to embeddings
  - `EncoderLayer`/`DecoderLayer`: Single layers of the encoder/decoder
  - `Encoder`/`Decoder`: Full encoder/decoder stacks

- `encode_decode.py`: The complete transformer model
  - `EncodeDecode`: Combines encoder and decoder components
  - `Generator`: Final linear + softmax layer for output probabilities

- `training.py`: Training loop and optimization
  - `LabelSmoothing`: Label smoothing regularization
  - `NoamOpt`: Learning rate scheduler with warmup
  - `train_epoch`: Single training epoch function
  - `run_training`: Complete training pipeline

- `inference.py`: Model inference utilities
  - `greedy_decode`: Simple greedy decoding
  - `beam_search`: Beam search implementation for better translations
  - `translate`: High-level translation function

- `visualization.py`: Tools for visualizing model components
  - `visualize_attention`: Creates attention map visualizations
  - `visualize_mask`: Visualizes masking patterns
  - `display_chart`: Renders visualizations to HTML

- `transformer_explanation.ipynb`: Jupyter notebook with step-by-step explanations and visualizations of the transformer architecture

### SV(Semantics_Visualizer)

Tools for visualizing word embeddings and semantic relationships between words.

#### Files:
- `SV.py`: Core implementation of the Semantics Visualizer
  - `load_embeddings`: Loads pre-trained word embeddings
  - `reduce_dimensions`: Reduces embedding dimensions for visualization
  - `find_similar_words`: Identifies semantically similar words
  - `visualize_embeddings`: Creates 2D/3D visualizations of word relationships

- `kaggle_semantics_visualizer.ipynb`: Kaggle notebook implementation
  - Contains the same functionality as SV.py but optimized for Kaggle environment
  - Includes interactive visualizations using Plotly
  - Can be run in the cloud without local dependencies

- `semantic_relationships.png`: Example visualization output

### data_processing

Tools and utilities for data preparation, tokenization, and dataset management.

#### Files:
- `tokenizer_trainer.py`: Scripts for training tokenizers
  - `TokenizerTrainer`: Class for training tokenizers on multiple languages
  - `train_sentencepiece`: Function for training SentencePiece tokenizers
  - `train_bpe`: Function for training Byte-Pair Encoding tokenizers
  - `train_wordpiece`: Function for training WordPiece tokenizers

- `data_preprocessing_guide.md`: Comprehensive guide for data preprocessing
  - Text cleaning procedures
  - Normalization approaches
  - Handling of special characters in Ewe
  - Sentence segmentation guidelines

- `dataset.py`: Dataset classes for loading and batching data
  - `TranslationDataset`: PyTorch dataset for translation pairs
  - `collate_fn`: Function for batching variable-length sequences
  - `create_masks`: Creates attention masks for transformer models

### Ewe_News_Dataset

Processing scripts and utilities specific to the Ewe News Dataset.

#### Files:
- `scraper.py`: Web scraper for collecting Ewe news articles
  - `scrape_news`: Function to scrape news from specific sources
  - `clean_article`: Cleans and normalizes scraped text

- `preprocess.py`: Preprocessing for Ewe news data
  - `normalize_ewe`: Handles Ewe-specific characters and diacritics
  - `segment_sentences`: Splits text into sentences

- `statistics.py`: Analyzes dataset statistics
  - `compute_stats`: Calculates vocabulary size, sentence lengths, etc.
  - `plot_distributions`: Visualizes statistical distributions

### Kaggle_dataset

Scripts for working with datasets from Kaggle, particularly the Ewe-English bilingual pairs dataset.

#### Files:
- `download.py`: Utilities for downloading datasets from Kaggle
  - `download_dataset`: Downloads specified dataset using Kaggle API

- `prepare_ewe_english.py`: Preparation of Ewe-English parallel corpus
  - `load_and_clean`: Loads and cleans the bilingual pairs
  - `split_data`: Creates train/validation/test splits
  - `save_splits`: Saves processed data in appropriate format

- `augmentation.py`: Data augmentation techniques
  - `back_translation`: Implements back-translation for data augmentation
  - `word_replacement`: Simple word replacement augmentation

### Tensorflow_Ytb_Channel

TensorFlow-based implementations of transformer models, following tutorials from various YouTube channels.

#### Files:
- `tf_transformer.py`: TensorFlow implementation of transformer
  - Similar structure to PyTorch implementation but using TensorFlow API

- `tf_training.py`: Training utilities for TensorFlow models
  - Custom training loop with TensorFlow
  - TensorBoard integration

- `tf_inference.py`: Inference utilities for TensorFlow models

### blog

Project blog documenting progress, insights, and findings.

#### Files:
- `index.md`: Main blog index page
  - Links to all blog posts
  - Overview of the project

- `first_post.md`: Introduction to the project
  - Project goals and motivation
  - Initial approach to transformer implementation

- `tokenization.md`: (Upcoming) Post about tokenization approaches
  - Challenges in tokenizing Ewe
  - Comparison of tokenization methods

- `translation_approaches.md`: (Upcoming) Post about translation strategies
  - Different approaches to Ewe-English translation
  - Preliminary results and comparisons

- `images/`: Directory containing images for blog posts

- `convert.py`: Script for converting blog posts to different formats
  - Converts Markdown to HTML for GitHub Pages
  - Prepares posts for Medium publication

### documentation

Project documentation files providing detailed information about various aspects of the project.

#### Files:
- `PROJECT_STRUCTURE.md`: This document - detailed explanation of project structure
- `ewe_english_translation_approaches.md`: Documentation on translation approaches for Ewe-English
- `DATASETS.md`: Information about datasets used in the project
- `LARGE_FILES_HANDLING.md`: Guidelines for handling large files in the repository
- `IMPORT_TROUBLESHOOTING.md`: Solutions for common import issues

### evaluation

Scripts and tools for evaluating model performance.

#### Files:
- `run_evaluation.py`: Comprehensive evaluation script for measuring model performance
  - `evaluate_bleu`: Function to calculate BLEU scores
  - `evaluate_examples`: Function to test model on example sentences
- `simple_evaluation.py`: Simplified evaluation script for quick testing

### Research

Research papers, reports, and academic documentation.

#### Files:
- `ewe_english_transformer_research_updated.md`: Comprehensive research report on Ewe-English translation
  - Analysis of transformer performance on Ewe-English translation
  - Discussion of challenges and future directions

### tools

Utility scripts and tools for repository management.

#### Files:
- `bfg.jar`: BFG Repo-Cleaner for handling large files
- `remove_large_files.sh`: Script for removing large files from repository history
- `large-files.txt`: List of large files to be managed separately

## Data Flow

The typical data flow through the system is as follows:

1. **Data Collection**:
   - Raw text is collected from various sources (Kaggle datasets, web scraping)
   - Parallel Ewe-English text pairs are organized

2. **Preprocessing**:
   - Text cleaning and normalization (`data_processing/`)
   - Sentence segmentation and alignment

3. **Tokenization**:
   - Training tokenizers on cleaned data (`data_processing/tokenizer_trainer.py`)
   - Applying tokenization to create input sequences

4. **Model Training**:
   - Data is batched and fed to the transformer model (`Attention_Is_All_You_Need/training.py`)
   - Model parameters are optimized using the training loop

5. **Evaluation**:
   - Model is evaluated on test set
   - Translation quality metrics are computed

6. **Inference**:
   - Trained model is used for translation (`Attention_Is_All_You_Need/inference.py`)
   - Beam search is applied for better translation quality

7. **Visualization**:
   - Model internals are visualized (`Attention_Is_All_You_Need/visualization.py`)
   - Word embeddings are visualized (`SV(Semantics_Visualizer)/SV.py`)

## Development Workflow

The recommended development workflow is:

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   ```bash
   python Kaggle_dataset/download.py
   python Kaggle_dataset/prepare_ewe_english.py
   ```

3. **Train Tokenizers**:
   ```bash
   python data_processing/tokenizer_trainer.py --languages ewe english --method sentencepiece
   ```

4. **Train Model**:
   ```bash
   python Attention_Is_All_You_Need/training.py --config configs/base_transformer.yaml
   ```

5. **Evaluate and Visualize**:
   ```bash
   python Attention_Is_All_You_Need/inference.py --model-path checkpoints/latest.pt --test-file test.txt
   python Attention_Is_All_You_Need/visualization.py --model-path checkpoints/latest.pt
   ```

## Deployment

The project can be deployed in several ways:

1. **Translation API**:
   - Flask/FastAPI wrapper around the trained model
   - Dockerized service for easy deployment

2. **Web Demo**:
   - Simple web interface for translation
   - Visualization of attention patterns

3. **Offline Usage**:
   - Command-line interface for batch translation
   - Integration with text editors via plugins

For more details on deployment options, see the `deployment/` directory (planned for future development).