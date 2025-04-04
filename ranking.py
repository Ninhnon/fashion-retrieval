import os
import time
import pathlib
from PIL import Image
from argparse import ArgumentParser
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import faiss

from src.feature_extraction import MyResnet50, MyVGG16, RGBHistogram, LBP, MyCLIP, MyEfficientNetV2, MyViT
from src.dataloader import get_transformation

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']

query_root = './dataset/groundtruth'
image_root = './dataset/cloth'
feature_root = './dataset/feature'
evaluate_root = './dataset/evaluation'

# query_root = './dataset/groundtruthFashion'
# image_root = './dataset/images2'
# feature_root = './dataset/featureFashion'
# evaluate_root = './dataset/evaluationFashion'


## Folder have many sub folders
# def get_image_list(image_root):
#     def list_images(root):
#         image_list = []
#         for entry in root.iterdir():
#             if entry.is_file() and entry.suffix.lower() in ['.jpg', '.jpeg', '.png']:
#                 image_list.append(entry)
#             elif entry.is_dir():
#                 # Recursively scan subdirectories
#                 image_list.extend(list_images(entry))
#         return image_list
    
#     image_root = pathlib.Path(image_root)
#     if not image_root.exists():
#         print(f"Error: Directory {image_root} does not exist.")
#         return []
    
#     image_list = list_images(image_root)
#     image_list = sorted(image_list, key=lambda x: x.name)
#     return image_list

# Get the list of images in 1 folder
# Get the list of images in 1 folder
def get_image_list(image_root):
    image_list = []
    image_root = pathlib.Path(image_root)
    
    # Recursively get all image files in the directory and subdirectories
    for image_path in image_root.rglob('*.*'):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_list.append(str(image_path))
    
    image_list = sorted(image_list, key=lambda x: x)
    return image_list
# def get_image_list(image_root):
#     image_root = pathlib.Path(image_root)
#     image_list = list()
#     for image_path in os.listdir(image_root):
#         image_list.append(image_path[:-4])
#     image_list = sorted(image_list, key = lambda x: x)
#     return image_list


def main():

    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet50')
    # parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--top_k", required=False, type=int, default=11)
    parser.add_argument("--crop", required=False, type=bool, default=False)

    print('Ranking .......')
    start = time.time()

    args = parser.parse_args()
    # device = torch.device(args.device)

    if (args.feature_extractor == 'Resnet50'):
        extractor = MyResnet50()
    elif (args.feature_extractor == 'VGG16'):
        extractor = MyVGG16()
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

    img_list = get_image_list(image_root)
    transform = get_transformation()

    for path_file in os.listdir(query_root):
        if (path_file[-9:-4] == 'query'):
            rank_list = []

            with open(query_root + '/' + path_file, "r") as file:
                img_query, left, top, right, bottom = file.read().split()
                print("🚀 ~ img_query:", img_query)

            test_image_path = pathlib.Path( image_root+ '/' + img_query + '.jpg')
            pil_image = Image.open(test_image_path)
            pil_image = pil_image.convert('RGB')

            path_crop = 'original'
            if (args.crop):
                pil_image=pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'

            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to('cpu')
            feat = extractor.extract_features(image_tensor)

            indexer = faiss.read_index(feature_root + '/' + args.feature_extractor + '.index.bin')

            _, indices = indexer.search(feat, k=args.top_k)  

            for index in indices[0]:
                rank_list.append(str(img_list[index]))

            with open(evaluate_root + '/' + path_crop + '/' + args.feature_extractor + '/' + path_file[:-10] + '.txt', "w") as file:
                file.write("\n".join(rank_list))

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()