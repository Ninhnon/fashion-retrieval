import time
import json
import requests
from argparse import ArgumentParser
from io import BytesIO
from PIL import Image

import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.feature_extraction import MyResnet50, MyVGG16, RGBHistogram, LBP
from src.indexing import get_faiss_indexer
from src.dataloader import MyDataLoader
from src.dataloader import get_transformation

feature_root = './test'
json_file_path = './dataset/image.json'

def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print(f"Failed to download image from {url}")
            return None
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def main():

    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=False, type=str, default='Resnet50')
    parser.add_argument("--batch_size", required=False, type=int, default=64)

    print('Start indexing .......')
    start = time.time()

    args = parser.parse_args()
    batch_size = args.batch_size

    # Load module for feature extraction 
    if args.feature_extractor == 'Resnet50':
        extractor = MyResnet50()
    elif args.feature_extractor == 'VGG16':
        extractor = MyVGG16()
    elif args.feature_extractor == 'RGBHistogram':
        extractor = RGBHistogram()
    elif args.feature_extractor == 'LBP':
        extractor = LBP()
    else:
        print("No matching model found")
        return

    # Load JSON file
    with open(json_file_path, 'r') as f:
        products = json.load(f)

    indexer = get_faiss_indexer(extractor.shape)

    for product in products:
        image_url = product["ProductImageUrl"]
        image = download_image(image_url)
        if image is not None:
            transform = get_transformation()
            img = image.convert('RGB')
            image_tensor = transform(img)
            image_tensor = image_tensor.unsqueeze(0).to('cpu')
            # Extract features
            features = extractor.extract_features(image_tensor)
            indexer.add(features)

    # Save features
    faiss.write_index(indexer, feature_root + '/' + args.feature_extractor + '.index.bin')

    end = time.time()
    print(f"Finished indexing in {end - start:.2f} seconds")


if __name__ == '__main__':
    main()
