# HCE_ImageModeling

This repository provides a modular pipeline for household carbon emission (HCE) estimation based on aerial imagery and structured survey data. The approach integrates deep learning-based image feature extraction, unsupervised clustering, and supervised modeling using CatBoost.

## Directory Structure

## Workflow Overview

The full pipeline consists of the following steps:

1. **[Optional] Train ResNet50**  
   Train the image encoder using labeled or unsupervised image data.  
   `scripts/imagecluster/main.py`

2. **Generate Image Embeddings**  
   Use the trained ResNet50 to encode aerial images into numerical embeddings.  
   `scripts/imagecluster/predict.py`

3. **Cluster Image Embeddings**  
   Perform clustering (e.g., K-Means) to identify building layout types.  
   `scripts/imagecluster/cluster.py`

4. **Model HCE with Text + Image Data**  
   Combine clustering results and household survey data to train the final prediction model using CatBoost.  
   `scripts/model_training/catboost_image.py`

---

### Data Flow Summary

```text
Image Input → ResNet50 → Embedding → Clustering → Cluster Label
                                                  ↓
                          Household Survey Data + Cluster Label → CatBoost → HCE Prediction
