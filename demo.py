import os
import sys
import time
from pathlib import Path
from PIL import Image

import torch
import faiss
import streamlit as st
from streamlit_cropper import st_cropper

from src.feature_extraction import MyResnet50, RGBHistogram, LBP, MyEfficientNetV2, MyViT
from src.dataloader import get_transformation 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Default paths relative to project root
DEFAULT_IMAGE_ROOT = Path('dataset/cloth')
DEFAULT_FEATURE_ROOT = Path('dataset/feature')

# Configure Streamlit page
st.set_page_config(
    page_title="Content-Based Image Retrieval",
    page_icon="🔍",
    layout="wide"
)

def get_image_list(image_root):
    image_list = []
    image_root = pathlib.Path(image_root)
    
    # Recursively get all image files in the directory and subdirectories
    for image_path in image_root.rglob('*.*'):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_list.append(str(image_path))
    
    image_list = sorted(image_list, key=lambda x: x)
    return image_list


def retrieve_image(img, feature_extractor, feature_root=DEFAULT_FEATURE_ROOT):
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
    index_path = Path(feature_root) / f"{feature_extractor}.index.bin"
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    indexer = faiss.read_index(str(index_path))
    _, indices = indexer.search(feat, k=11)
    return indices[0]


def main(image_root=DEFAULT_IMAGE_ROOT, feature_root=DEFAULT_FEATURE_ROOT):
    st.title('Content-Based Image Retrieval System')

    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Select Feature Extractor')
        option = st.selectbox(
            'Choose a feature extraction method:',
            ('Resnet50', 'EfficientNetV2', 'VIT', 'RGBHistogram', 'LBP'),
            help="Select the algorithm to use for feature extraction"
        )

        st.subheader('Upload Query Image')
        img_file = st.file_uploader(
            label='Select an image file',
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to search for similar items"
        )

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

            retrieve_list = retrieve_image(cropped_img, option, feature_root)
            image_list = get_image_list(str(image_root))

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(image_list[retrieve_list[0]])
                st.image(image, use_column_width = 'always')

            with col4:
                image = Image.open(image_list[retrieve_list[1]])
                st.image(image, use_column_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(image_list[retrieve_list[u]])
                    st.image(image, use_column_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(image_list[retrieve_list[u]])
                    
                    st.image(image, use_column_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(image_list[retrieve_list[u]])
                    st.image(image, use_column_width = 'always')
       

if __name__ == '__main__':
    main()

