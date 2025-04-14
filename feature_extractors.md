# Feature Extractors Technical Documentation

This document provides detailed technical information about each feature extraction method implemented in the system.

## Table of Contents

1. [Implementation Overview](#implementation-overview)
2. [Deep Learning Models](#deep-learning-models)
3. [Traditional Methods](#traditional-methods)
4. [Usage Guidelines](#usage-guidelines)
5. [Implementation Details](#implementation-details)

## Implementation Overview

All feature extractors follow a common interface with two key components:

- `shape` property: Defines the output feature vector dimension
- `extract_features(image)` method: Processes input images and returns feature vectors

## Deep Learning Models

### 1. EfficientNetV2 (`MyEfficientNetV2`)

```python
Input: RGB Image Tensor (B, 3, H, W)
Output: Feature Vector (B, 1280)
```

**Implementation Details:**

- Uses EfficientNetV2-S variant
- Pre-trained on ImageNet1K
- Removes classification layer
- Applies standard ImageNet normalization
- Feature dimension: 1280

**Preprocessing:**

```python
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

### 2. Vision Transformer (`MyViT`)

```python
Input: RGB Image Tensor (B, 3, 224, 224)
Output: Feature Vector (B, 768)
```

**Implementation Details:**

- Uses ViT-B/16 architecture
- Pre-trained on ImageNet1K
- Extracts CLS token embedding
- Feature dimension: 768

**Key Components:**

- Modified forward pass to extract embeddings
- Custom preprocessing pipeline
- Input size fixed at 224x224

### 3. ResNet50 (`MyResnet50`)

```python
Input: RGB Image Tensor (B, 3, 224, 224)
Output: Feature Vector (B, 2048)
```

**Implementation Details:**

- Uses ResNet50 architecture
- Pre-trained on ImageNet1K V2
- Removes final classification layer
- Feature dimension: 2048

**Preprocessing Steps:**

1. Resize to 224x224
2. Convert to tensor
3. Apply ImageNet normalization

## Traditional Methods

### 1. RGB Histogram (`RGBHistogram`)

```python
Input: RGB Image Tensor (B, 3, H, W)
Output: Feature Vector (B, 768)
```

**Implementation Details:**

- Computes histogram for each color channel
- 256 bins per channel
- Normalized using Min-Max scaling
- Total feature dimension: 768 (256 \* 3)

**Processing Steps:**

1. Convert tensor to numpy array
2. Scale to 0-255 range
3. Compute histogram per channel
4. Normalize histograms
5. Concatenate channels

### 2. Local Binary Patterns (`LBP`)

```python
Input: RGB Image Tensor (B, 3, H, W)
Output: Feature Vector (B, 26)
```

**Implementation Details:**

- Uses scikit-image LBP implementation
- Parameters:
  - n_points = 24
  - radius = 3
- Feature dimension: 26

**Processing Steps:**

1. Convert to grayscale
2. Compute LBP
3. Create histogram
4. Normalize histogram

## Usage Guidelines

### 1. Memory Considerations

```python
Feature Dimensions:
- ResNet50: 2048 * 4 bytes = 8,192 bytes per image
- ViT: 768 * 4 bytes = 3,072 bytes per image
- EfficientNetV2: 1280 * 4 bytes = 5,120 bytes per image
- RGB Histogram: 768 * 4 bytes = 3,072 bytes per image
- LBP: 26 * 4 bytes = 104 bytes per image
```

### 2. Batch Processing

Recommended batch sizes:

- GPU (8GB+): 64-128 images
- GPU (4GB): 32-64 images
- CPU: 16-32 images

### 3. Input Requirements

```python
Image Requirements:
- Format: RGB
- Type: torch.Tensor
- Range: [0, 1]
- Normalization: Model-specific
```

## Implementation Details

### 1. Feature Extraction Pipeline

```python
def extract_features(image):
    1. Preprocess image
    2. Forward pass through model
    3. Flatten output
    4. Convert to numpy array
    5. Return features
```

### 2. Error Handling

Common issues and solutions:

1. CUDA out of memory:

   - Reduce batch size
   - Use CPU fallback

2. Input format:

   - Ensure correct normalization
   - Verify tensor dimensions

3. Performance:
   - Enable GPU acceleration
   - Use appropriate batch size

### 3. Best Practices

1. **Model Loading:**

   ```python
   # Load once, reuse for multiple images
   extractor = MyResnet50()
   ```

2. **Batch Processing:**

   ```python
   # Process multiple images efficiently
   features = extractor.extract_features(batch_images)
   ```

3. **Memory Management:**
   ```python
   # Clear GPU memory when done
   torch.cuda.empty_cache()
   ```

### 4. Customization Options

Each extractor can be customized:

1. **ResNet50:**

   - Change base model (ResNet18, ResNet101)
   - Modify preprocessing
   - Add feature normalization

2. **ViT:**

   - Adjust model size
   - Modify patch size
   - Change embedding extraction

3. **EfficientNetV2:**

   - Switch model variant
   - Customize preprocessing
   - Add post-processing

4. **RGB Histogram:**

   - Modify bin count
   - Change color space
   - Adjust normalization

5. **LBP:**
   - Change radius
   - Adjust point count
   - Modify histogram computation
