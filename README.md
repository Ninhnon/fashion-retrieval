# Image Retrieval in Fashion: A Content-Based Approach

## Abstract

This research implements and evaluates a Content-Based Image Retrieval (CBIR) system that combines traditional computer vision techniques with modern deep learning approaches. The system leverages Facebook's FAISS (Facebook AI Similarity Search) for efficient similarity search and indexing, while comparing multiple feature extraction methods including both classical computer vision algorithms and state-of-the-art deep learning models.

## Key Features

- **Multiple Feature Extraction Methods:**

  - Deep Learning Models:
    - ResNet50 (Feature vector: 2,048 dimensions)
    - Vision Transformer (ViT-B/16) (Feature vector: 768 dimensions)
    - EfficientNetV2 (Feature vector: 1,280 dimensions)
  - Traditional Methods:
    - RGB Histogram (Feature vector: 768 dimensions)
    - Local Binary Patterns (LBP) (Feature vector: 26 dimensions)

- **Efficient Similarity Search:**

  - Utilizes Facebook's FAISS library for fast and scalable similarity search
  - Supports large-scale image databases with efficient indexing

- **Comprehensive Evaluation:**
  - Built-in evaluation pipeline for the Deep Fashion dataset
  - Mean Average Precision (MAP) computation
  - Query set evaluation capabilities

## System Architecture

The system follows a two-phase approach:

1. **Indexing Phase:**
   - Feature extraction from the image database
   - Index creation using FAISS
2. **Retrieval Phase:**
   - Query image feature extraction
   - Similarity search using the created index
   - Ranked retrieval of similar images

## Technical Requirements

- Python 3.10
- PyTorch with CUDA 11.7 support
- FAISS-GPU
- Additional dependencies listed in `requirements.txt`

## Installation

1. Set up PyTorch with CUDA:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

2. Install FAISS:

```bash
conda install -c conda-forge faiss-gpu
```

3. Install other dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The system expects the following dataset organization:

```
dataset/
├── evaluation/
│   ├── crop/
│   │   ├── LBP/
│   │   ├── Resnet50/
│   │   ├── RGBHistogram/
│   └── original/
├── feature/
│   ├── LBP.index.bin
│   ├── Resnet50.index.bin
│   ├── RGBHistogram.index.bin
├── groundtruth/
├── cloth/
└── fashion/
```

## Usage

### Feature Extraction and Indexing

```bash
python indexing.py --feature_extractor Resnet50
```

### Evaluation

```bash
# Evaluate on query set
python ranking.py --feature_extractor Resnet50

# Compute Mean Average Precision (MAP)
python evaluate.py --feature_extractor Resnet50
```

### Interactive Demo

```bash
streamlit run demo.py
```

## Configuration Options

The system supports various configuration parameters:

- `feature_extractor`: Choose between RGBHistogram, LBP, Resnet50, ViT, or EfficientNetV2
- `batch_size`: Adjust batch size for feature extraction
- `top_k`: Number of similar images to retrieve

## Dataset

This research uses the [Deep Fashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), which contains 239,557 number of consumer/shop clothes images. The dataset includes ground truth files for evaluation.

# Dataset Download Guide for VNIU-VNR50

## Downloading Dataset

### Method 1: Using Web Interface

https://www.kaggle.com/datasets/ninhnguyentrong/vniu-vnr50

1. Visit the dataset page on Kaggle
2. Click the "Download" button
3. Save the files to your local machine
4. If downloaded as ZIP, extract the contents

### Method 2: Using curl

```bash
curl -L -o ~/Downloads/vniu-vnr50.zip\
  https://www.kaggle.com/api/v1/datasets/download/ninhnguyentrong/vniu-vnr50
```

After downloading, your dataset should be organized as follows:

```
dataset/
── clothes/
   ├── image1.jpg
   ├── image2.jpg
   ...


* Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research
- Pre-trained models from [torchvision.models](https://pytorch.org/vision/stable/models.html)
- [Deep Fashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) by the Visual Geometry Group
```
