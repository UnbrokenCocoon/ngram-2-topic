# NGram2Top: A Novel Topic Modelling Pipeline Using N-Grams and Sentence Embeddings

## Overview

**NGram2Top** is a custom topic modelling pipeline that combines n-gram frequency analysis with sentence-level embeddings to generate interpretable and semantically rich topics. Unlike traditional BERTopic pipelines, this approach decouples embedding generation from term extraction and clustering, allowing for greater control and transparency over each step.

This notebook implements a **modular and interpretable topic modelling method** that:
- Uses `SentenceTransformers` to compute high-quality sentence embeddings.
- Applies `CountVectorizer` with custom n-gram ranges for extracting term frequency.
- Computes cosine similarity between embeddings to cluster related content.
- Outputs refined topic groupings based on both vector space similarity and linguistic patterns.

## Key Features

- 📌 **Novel Pipeline Structure**: By separating n-gram extraction and embedding-based clustering, this method avoids some of the black-box limitations of end-to-end topic modelling tools.
- 🔍 **Customisable N-Gram Range**: Built-in support for unigrams and bigrams with adjustable `min_df`, `token_pattern`, and stopword removal.
- 📊 **Embedding-Aware Similarity**: Topic clusters are refined using `cosine_similarity` on high-dimensional embeddings, ensuring semantic coherence.
- 🧠 **No HDBSCAN or UMAP Dependency for Clustering**: Instead, this method relies on direct vector space operations and manual cluster inspection, enabling greater interpretability.

## Workflow Summary

1. **Load Precomputed Sentence Embeddings**  
   Sentence embeddings (from BERT or any SentenceTransformer model) and corresponding text are loaded from `.pkl` files.

2. **Vectorisation of N-Grams**  
   Apply `CountVectorizer` with `(1, 2)` n-gram range to identify frequently occurring phrases.

3. **Cosine Similarity Matrix**  
   Compute cosine similarity across sentence embeddings to identify semantic similarity between texts.

4. **Manual or Custom Clustering**  
   Use similarity scores and n-gram frequency to manually assign topic labels or experiment with your own clustering logic.

5. **Export / Visualise Topics**  
   Topics are optionally visualised or saved for downstream use, e.g., exporting cluster terms, plotting frequency graphs, or manual annotation.

## Usage

This notebook is designed for use in **Google Colab**, with optional use of Google Drive for data storage. You will need:

- `.pkl` files containing:
  - `sentences`: a list of text data.
  - `embeddings`: corresponding sentence embeddings (e.g. from `'all-mpnet-base-v2'`).

## Installation

```bash
pip install sentence-transformers spacy
```

## Example Output

The final output includes:
- A DataFrame with each sentence and its assigned cluster/topic.
- A sorted list of key terms (n-grams) per cluster.
- Optional cosine similarity visualisations.

## Why Use This?

Traditional topic modelling tools (e.g. LDA, BERTopic) often mask internal operations or require complex parameter tuning. NGram2Top offers a **transparent, flexible, and research-friendly alternative** suited for small- to mid-scale corpora, particularly in academic or interpretive contexts.
