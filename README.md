# Knowledge-Aware Multimodal Hashtag Recommendation System

## Project Overview

This project develops a knowledge-aware multimodal hashtag recommendation system that combines visual information, textual information, and structured semantic knowledge to generate relevant hashtags for images and captions.

The system uses a Graph Attention Network (GAT) to learn relationships between knowledge graph concepts and Gated Cross-Attention (GCA) to integrate information from different modalities.

## Dataset

- **Dataset:** Flickr30k
- **Source:** Kaggle
- **Task:** Multi-label hashtag recommendation
- **Selected hashtag classes:** 73

The dataset contains images with corresponding captions. Keywords extracted from the captions are used to construct hashtag labels for supervised learning.

## Technologies Used

- Python and PyTorch
- CLIP ViT-B/32
- SentenceTransformer (all-MiniLM-L6-v2)
- WordNet and ConceptNet
- PyTorch Geometric
- Graph Attention Network (GAT)
- Gated Cross-Attention (GCA)

## Methodology

1. **Dataset Preparation:** Load images and captions, extract hashtags, and prepare training and validation samples.
2. **Visual Feature Extraction:** Extract 512-dimensional image embeddings using CLIP.
3. **Textual Feature Extraction:** Generate 384-dimensional caption embeddings using SentenceTransformer.
4. **Knowledge Graph Construction:** Select relevant concepts and expand them using WordNet and ConceptNet.
5. **Graph Representation Learning:** Apply a two-layer GAT to learn relationships between connected concepts.
6. **Multimodal Fusion:** Combine visual, textual, and knowledge graph representations using Gated Cross-Attention.
7. **Hashtag Prediction:** Predict scores for multiple hashtag classes.
8. **Personalization:** Re-rank candidate hashtags using simulated user preferences.
9. **Explainability:** Analyze modality contributions and knowledge graph information.

## Model Evaluation

The recommendation system is evaluated using:

- Precision@5
- Recall@5
- F1@5
- Mean Average Precision (MAP@5)
- Normalized Discounted Cumulative Gain (NDCG@5)

## How to Run

1. Open the main `.ipynb` notebook in Google Colab.
2. Enable a GPU runtime if available.
3. Execute the dependency installation cells.
4. Download and prepare the Flickr30k dataset.
5. Initialize the pretrained encoders and knowledge graph resources.
6. Run the feature extraction, graph construction, training, and evaluation cells in order.

**Note:** The dataset, ConceptNet database, and pretrained model files are downloaded during setup and are not included in this repository.

## Project Status

M.Tech Artificial Intelligence — Academic Project

The repository contains the implementation of the knowledge-aware multimodal hashtag recommendation pipeline, including model evaluation, personalization, and explainability.
