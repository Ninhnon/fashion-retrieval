import time
import os
from argparse import ArgumentParser
from pathlib import Path

import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.feature_extraction import MyResnet50, RGBHistogram, LBP, MyCLIP, MyEfficientNetV2, MyViT
from src.indexing import get_faiss_indexer
from src.dataloader import MyDataLoader

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Default paths relative to project root
DEFAULT_IMAGE_ROOT = Path('dataset/cloth')
DEFAULT_FEATURE_ROOT = Path('dataset/feature')

def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet50',
                      help="Feature extractor to use: Resnet50, EfficientNetV2, VIT, RGBHistogram, or LBP")
    parser.add_argument("--batch_size", required=False, type=int, default=64,
                      help="Batch size for feature extraction")
    parser.add_argument("--image_root", type=str, default=str(DEFAULT_IMAGE_ROOT),
                      help="Path to directory containing images to index")
    parser.add_argument("--feature_root", type=str, default=str(DEFAULT_FEATURE_ROOT),
                      help="Path to directory for storing feature indexes")

    print('Start indexing...')
    start = time.time()

    args = parser.parse_args()
    batch_size = args.batch_size

    # Create feature directory if it doesn't exist
    Path(args.feature_root).mkdir(parents=True, exist_ok=True)

    # Load module feature extraction 
    if (args.feature_extractor == 'Resnet50'):
        extractor = MyResnet50()
    elif (args.feature_extractor == 'EfficientNetV2'):
        extractor = MyEfficientNetV2()
    elif (args.feature_extractor == 'VIT'):
        extractor = MyViT()
    elif (args.feature_extractor == 'RGBHistogram'):
        extractor = RGBHistogram()
    elif (args.feature_extractor == 'LBP'):
        extractor = LBP()
    
    else:
        print("No matching model found")
        return

    dataset = MyDataLoader(args.image_root)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)

    indexer = get_faiss_indexer(extractor.shape)

    for images, image_paths in dataloader:
        images = images.to('cpu')
        features = extractor.extract_features(images)
        batch_size = 5000  # Thử nghiệm với giá trị nhỏ hơn
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            # indexer.train(batch)
            indexer.add(batch)
        # print(features.shape)
        # indexer.add(features)
    
    # Save features
    # Save features using Path for proper path joining
    index_path = Path(args.feature_root) / f"{args.feature_extractor}.index.bin"
    faiss.write_index(indexer, str(index_path))
    
    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()
