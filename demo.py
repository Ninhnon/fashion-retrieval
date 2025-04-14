import time
import torch
import faiss
import pathlib
from PIL import Image
import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.feature_extraction import MyResnet50, RGBHistogram, LBP, MyEfficientNetV2, MyViT
from src.dataloader import get_transformation 
import streamlit as st
from streamlit_cropper import st_cropper

st.set_page_config(layout="wide")

# device = torch.device('cpu')
image_root = './dataset/cloth'
feature_root = './dataset/feature'
# image_root = './dataset/images2'
# feature_root = './dataset/featureFashion'

def get_image_list(image_root):
    image_list = []
    image_root = pathlib.Path(image_root)
    
    # Recursively get all image files in the directory and subdirectories
    for image_path in image_root.rglob('*.*'):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_list.append(str(image_path))
    
    image_list = sorted(image_list, key=lambda x: x)
    return image_list


def retrieve_image(img, feature_extractor):
    if feature_extractor == 'Resnet50':
        extractor = MyResnet50()
    elif feature_extractor == 'EfficientNetV2':
        extractor = MyEfficientNetV2()
    elif feature_extractor == 'VIT':
        extractor = MyViT()
    elif feature_extractor == 'RGBHistogram':
        extractor = RGBHistogram()
    elif feature_extractor == 'LBP':
        extractor = LBP()
    else:
        raise ValueError("Unknown feature extractor")

    transform = get_transformation()
    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to('cpu')
    feat = extractor.extract_features(image_tensor)
    indexer_path = feature_root + '/' + feature_extractor + '.index.bin'
    if not os.path.isfile(indexer_path):
        raise FileNotFoundError(f"Index file not found: {indexer_path}")

    indexer = faiss.read_index(indexer_path)
    _, indices = indexer.search(feat, k=11)
    return indices[0]


def main():
    st.title('CONTENT-BASED IMAGE RETRIEVAL')

    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ( 'Resnet50','EfficientNetV2','VIT', 'RGBHistogram', 'LBP'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option)
            image_list = get_image_list(image_root)

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(image_list[retriev[0]])
                st.image(image, use_column_width = 'always')

            with col4:
                image = Image.open(image_list[retriev[1]])
                st.image(image, use_column_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(image_list[retriev[u]])
                    st.image(image, use_column_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(image_list[retriev[u]])
                    
                    st.image(image, use_column_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(image_list[retriev[u]])
                    st.image(image, use_column_width = 'always')
                # for i in range(11):
                #     print(image_list[retriev[i]])
       

if __name__ == '__main__':
    main()


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