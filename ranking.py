import os
import time
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
import torch
import faiss

from src.feature_extraction import MyResnet50, RGBHistogram, LBP, MyCLIP, MyEfficientNetV2, MyViT
from src.dataloader import get_transformation

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Default paths relative to project root
DEFAULT_QUERY_ROOT = Path('dataset/groundtruth')
DEFAULT_IMAGE_ROOT = Path('dataset/cloth')
DEFAULT_FEATURE_ROOT = Path('dataset/feature')
DEFAULT_EVALUATE_ROOT = Path('dataset/evaluation')

ACCEPTED_IMAGE_EXTS = ['.jpg', '.jpeg', '.png']

def get_image_list(image_root):
    image_list = []
    image_root = pathlib.Path(image_root)
    
    # Recursively get all image files in the directory and subdirectories
    for image_path in image_root.rglob('*.*'):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_list.append(str(image_path))
    
    image_list = sorted(image_list, key=lambda x: x)
    return image_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet50',
                      help="Feature extractor to use: Resnet50, EfficientNetV2, VIT, RGBHistogram, or LBP")
    parser.add_argument("--top_k", required=False, type=int, default=11,
                      help="Number of similar images to retrieve")
    parser.add_argument("--crop", required=False, type=bool, default=False,
                      help="Whether to crop images using bounding box coordinates")
    parser.add_argument("--query_root", type=str, default=str(DEFAULT_QUERY_ROOT),
                      help="Path to directory containing query images")
    parser.add_argument("--image_root", type=str, default=str(DEFAULT_IMAGE_ROOT),
                      help="Path to directory containing database images")
    parser.add_argument("--feature_root", type=str, default=str(DEFAULT_FEATURE_ROOT),
                      help="Path to directory containing feature indexes")
    parser.add_argument("--evaluate_root", type=str, default=str(DEFAULT_EVALUATE_ROOT),
                      help="Path to directory for storing evaluation results")

    print('Ranking .......')
    start = time.time()

    args = parser.parse_args()
    # device = torch.device(args.device)

    if (args.feature_extractor == 'Resnet50'):
        extractor = MyResnet50()
    elif (args.feature_extractor == 'RGBHistogram'):
        extractor = RGBHistogram()
    elif (args.feature_extractor == 'LBP'):
        extractor = LBP()
    elif (args.feature_extractor == 'VIT'):
        extractor = MyViT()
    elif (args.feature_extractor == 'EfficientNetV2'):
        extractor = MyEfficientNetV2()
    else:
        print("No matching model found")
        return

    args = parser.parse_args()

    # Create evaluation directory if it doesn't exist
    evaluate_path = Path(args.evaluate_root)
    evaluate_path.mkdir(parents=True, exist_ok=True)

    img_list = get_image_list(args.image_root)
    transform = get_transformation()

    for path_file in os.listdir(args.query_root):
        if path_file.endswith('query'):
            rank_list = []

            query_path = Path(args.query_root) / path_file
            with open(query_path, "r") as file:
                img_query, left, top, right, bottom = file.read().split()
                print("🚀 ~ img_query:", img_query)

            test_image_path = Path(args.image_root) / f"{img_query}.jpg"
            pil_image = Image.open(test_image_path)
            pil_image = pil_image.convert('RGB')

            path_crop = 'original'
            if (args.crop):
                pil_image=pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'

            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to('cpu')
            feat = extractor.extract_features(image_tensor)

            index_path = Path(args.feature_root) / f"{args.feature_extractor}.index.bin"
            indexer = faiss.read_index(str(index_path))

            _, indices = indexer.search(feat, k=args.top_k)  

            for index in indices[0]:
                rank_list.append(str(img_list[index]))

            # Create subdirectories for evaluation results
            result_dir = evaluate_path / path_crop / args.feature_extractor
            result_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = result_dir / f"{path_file[:-10]}.txt"
            with open(result_path, "w") as file:
                file.write("\n".join(rank_list))

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()
