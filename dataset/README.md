# Dataset Setup Instructions

## Download Instructions

### 1. VNIU-VNR50 Dataset

#### Option 1: Using Kaggle Web Interface

1. Visit [VNIU-VNR50 Dataset on Kaggle](https://www.kaggle.com/datasets/ninhnguyentrong/vniu-vnr50)
2. Click "Download" button
3. Save the ZIP file to your local machine

#### Option 2: Using Kaggle API

```bash
# First, configure your Kaggle API credentials
# Place kaggle.json in ~/.kaggle/

# Download the dataset
kaggle datasets download ninhnguyentrong/vniu-vnr50
```

### 2. Deep Fashion Dataset

1. Visit [Deep Fashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
2. Download the Consumer-to-shop Clothes Retrieval Benchmark
3. Extract the downloaded files

## Directory Structure

After downloading, organize the files in the following structure:

```
dataset/
├── cloth/                  # All clothing images
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
├── feature/               # Feature indexes (created during indexing)
│   ├── Resnet50.index.bin
│   ├── VIT.index.bin
│   ├── EfficientNetV2.index.bin
│   ├── RGBHistogram.index.bin
│   └── LBP.index.bin
│
├── groundtruth/          # Query information files
│   ├── query1.txt
│   ├── query1_good.txt
│   ├── query1_ok.txt
│   └── ...
│
└── evaluation/          # Evaluation results
    ├── crop/           # Results with cropped images
    │   ├── Resnet50/
    │   ├── VIT/
    │   └── ...
    └── original/       # Results with original images
        ├── Resnet50/
        ├── VIT/
        └── ...
```

## File Formats

### 1. Query Files (in groundtruth/)

```
# Format: image_name left top right bottom
example_image.jpg 100 150 300 400
```

### 2. Feature Index Files (in feature/)

- Binary files created during indexing process
- Named according to feature extractor used
- Example: `VIT.index.bin`

### 3. Evaluation Results (in evaluation/)

- Text files containing ranked lists of similar images
- Organized by feature extractor and image type (crop/original)

## Dataset Statistics

### VNIU-VNR50 Dataset

- Total images: 1,311
- Image format: JPG
- Resolution: Variable (will be resized to 224x224)
- File size: ~60MB compressed

### Deep Fashion Dataset

- Total images: 239,557
- Image format: JPG
- Resolution: Variable (will be resized to 224x224)
- File size: ~2,5GB compressed

## Usage Notes

1. **Image Preprocessing**

   - All images will be automatically resized to 224x224
   - RGB format is required
   - Ensure proper file permissions

2. **Memory Requirements**

   - Minimum 16GB RAM recommended
   - ~50GB free disk space for full setup

3. **Validation**
   - Verify all images are readable
   - Check groundtruth file formats
   - Ensure directory structure is correct

## Troubleshooting

1. **Missing Files**

   ```bash
   # Check file counts
   find dataset/cloth -type f | wc -l
   find dataset/groundtruth -type f | wc -l
   ```

2. **Directory Structure**

   ```bash
   # Create missing directories
   mkdir -p dataset/{cloth,feature,groundtruth,evaluation/{crop,original}}
   ```

3. **Common Issues**
   - Invalid image files: Remove or replace corrupted images
   - Wrong file permissions: Use `chmod -R 644` for files
   - Missing directories: Create using mkdir command above

## Citation

If using these datasets, please cite:

```bibtex
@article{10.5281/zenodo.15151481,
    title={Vision Transformer for Fashion Image Retrieval: A Comprehensive Evaluation on Real-World Datasets},
    author={Truc Nguyen, Ninh Nguyen, Vu Tran, Qui Nguyen, Trinh Huynh},
    journal={The Visual Computer},
    year={2025},
    publisher={Springer}
}
```
