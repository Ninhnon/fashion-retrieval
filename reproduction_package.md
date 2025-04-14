# Vision Transformer for Fashion Image Retrieval A Comprehensive Evaluation on Real-World Datasets: Reproduction Package

This document provides comprehensive instructions for reproducing the research results from our image retrieval system.

## Table of Contents

1. [Dataset Processing](#dataset-processing)
2. [Feature Extraction Methods](#feature-extraction-methods)
3. [System Setup](#system-setup)
4. [Running Experiments](#running-experiments)
5. [Results & Evaluation](#results--evaluation)

## Dataset Processing

### Dataset Structure

The system requires the following directory structure:

```
dataset/
├── cloth/              # Contains all clothing images
├── groundtruth/        # Contains query image information
├── feature/           # Stores extracted feature indexes
└── evaluation/        # Stores evaluation results
```

### Data Preparation Steps

1. **Download the Dataset**

   - Option 1: VNIU-VNR50 Dataset from Kaggle:
     ```bash
     # Using Kaggle API
     kaggle datasets download ninhnguyentrong/vniu-vnr50
     ```
   - Option 2: Deep Fashion Dataset from official source
     - Visit [Deep Fashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
     - Download and extract the Consumer-to-shop Clothes Retrieval Benchmark

2. **Process Images**
   - Place clothing images in `dataset/cloth/`
   - Format ground truth files in `dataset/groundtruth/`
   - Each ground truth file should contain: `image_name left top right bottom`

## Feature Extraction Methods

Our system implements multiple feature extraction methods:

### Deep Learning Models

1. **ResNet50**

   - Feature vector: 2,048 dimensions
   - Pre-trained on ImageNet
   - Extracts high-level visual features

2. **Vision Transformer (ViT-B/16)**

   - Feature vector: 768 dimensions
   - Attention-based architecture
   - Effective at capturing global image context

3. **EfficientNetV2**
   - Feature vector: 1,280 dimensions
   - Optimized for both accuracy and efficiency
   - Strong performance on mobile devices

### Traditional Methods

1. **RGB Histogram**

   - Feature vector: 768 dimensions
   - Captures color distribution
   - Implementation: 256 bins per color channel

2. **Local Binary Patterns (LBP)**
   - Feature vector: 26 dimensions
   - Captures texture patterns
   - Rotation-invariant implementation

## System Setup

1. **Install Dependencies**

   ```bash
   # Install PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

   # Install FAISS for similarity search
   conda install -c conda-forge faiss-gpu

   # Install other requirements
   pip install -r requirements.txt
   ```

2. **Environment Requirements**
   - Python 3.10 or higher
   - CUDA 11.7 (for GPU support)
   - 8GB+ RAM recommended

## Running Experiments

### 1. Feature Extraction

```bash
# Extract features using different methods
python indexing.py --feature_extractor Resnet50
python indexing.py --feature_extractor VIT
python indexing.py --feature_extractor EfficientNetV2
python indexing.py --feature_extractor RGBHistogram
python indexing.py --feature_extractor LBP
```

### 2. Image Retrieval

```bash
# Run retrieval evaluation
python ranking.py --feature_extractor Resnet50 --top_k 11
```

### 3. Interactive Demo

```bash
# Launch web interface
streamlit run demo.py
```

## Results & Evaluation

### Evaluation Metrics

1. **Mean Average Precision (MAP)**

   - Measures ranking quality
   - Higher values indicate better retrieval performance

2. **Processing Times**
   - Indexing time: Time to extract and index features
   - Evaluation time: Time to process evaluation queries
   - Retrieval time: Average time per query retrieval

### Deep Fashion Dataset Results

| Method         | Indexing (s) | Evaluate (s) | Retrieve (s) | mAP   |
| -------------- | ------------ | ------------ | ------------ | ----- |
| RGB Histogram  | 16,340.03    | 43.467       | 13.088       | 0.439 |
| LBP            | 38,002.902   | 8.492        | 12.791       | 0.385 |
| ResNet50       | 61,564.061   | 132.337      | 11.483       | 0.679 |
| EfficientNetV2 | 63,914.953   | 75.518       | 9.832        | 0.621 |
| ViT            | 121,834.871  | 96.655       | 8.910        | 0.689 |

### VNIU-VNR50 Dataset Results

| Method         | Indexing (s) | Evaluate (s) | Retrieve (s) | mAP   |
| -------------- | ------------ | ------------ | ------------ | ----- |
| RGB Histogram  | 17.997       | 0.593        | 0.023        | 0.768 |
| LBP            | 38.373       | 1.651        | 0.056        | 0.601 |
| ResNet50       | 93.123       | 4.342        | 0.523        | 0.750 |
| EfficientNetV2 | 85.161       | 4.950        | 0.694        | 0.766 |
| ViT            | 466.702      | 22.054       | 1.744        | 0.791 |

### Key Findings

1. **Deep Fashion Dataset**

   - ViT achieves highest accuracy (mAP: 0.689) but longest indexing time
   - ResNet50 provides good balance of accuracy and speed
   - Traditional methods are faster but significantly less accurate

2. **VNIU-VNR50 Dataset**

   - ViT maintains best accuracy (mAP: 0.791)
   - RGB Histogram performs surprisingly well (mAP: 0.768)
   - All methods show improved accuracy on this dataset

3. **Performance Comparison**
   - Deep learning methods consistently outperform traditional approaches in accuracy
   - RGB Histogram offers fastest processing on small datasets
   - ViT shows best accuracy but requires significant processing time

### Experiment Notes

- All experiments run using batch size of 64
- Hardware: NVIDIA RTX 3080 GPU
- Deep Fashion Dataset: ~240K images
- VNIU-VNR50 Dataset: ~1,5K images
- Results averaged over multiple runs

## Reproduction Notes

1. **Common Issues & Solutions**

   - If CUDA out of memory: Reduce batch size
   - If slow indexing: Check CPU/GPU utilization
   - If missing features: Verify model downloads

2. **Quality Checks**

   - Verify feature extraction completion
   - Check evaluation output format
   - Monitor GPU memory usage

3. **Performance Tips**
   - Use SSD for faster data loading
   - Increase batch size if memory allows
   - Enable CUDA for GPU acceleration
