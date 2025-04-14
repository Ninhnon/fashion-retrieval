import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from skimage.feature import local_binary_pattern
from torchvision.models import vit_b_16, ViT_B_16_Weights  # You can replace with your desired ViT model

class MyEfficientNetV2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Load EfficientNetV2 with pre-trained weights
        self.model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        # Remove the classifier layer
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        # self.model = self.model.to('device')
        self.shape = 1280  # Output feature vector length

    def extract_features(self, image):
        """
        Directly extracts features from the image tensor.
        The input image is expected to be a tensor normalized for EfficientNetV2.
        """
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)

        return feature.cpu().detach().numpy()

class MyViT():
    def __init__(self):
        super().__init__()
        # Initialize ViT model
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Create a modified forward method to get embeddings instead of classification
        self.original_forward = self.model.forward
        self.model.forward = self._forward_features
        
        self.model.eval()        
        self.feature_dim = 768  # Fixed for ViT-B/16
        self.shape = 768
        
        # ViT specific transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        

    def _forward_features(self, x):
        """Modified forward pass to get embeddings instead of classification."""
        # Process input
        x = self.model._process_input(x)
        n = x.shape[0]

        # Add class token
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.model.encoder(x)

        # Return CLS token embedding
        return x[:, 0]

    @torch.no_grad()
    def extract_features(self, image) -> np.ndarray:
        """Extract features from a single image."""
        try:
            
            # Forward pass through model
            features = self.model(image)
            
            # Convert to numpy and normalize
            features = features.cpu().detach().numpy()

            
            # # L2 normalize the features
            # norm = np.linalg.norm(features)
            # if norm > 0:
            #     features = features / norm
            
            
            return features
            
        except Exception as e:
            raise

class MyResnet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(weights='IMAGENET1K_V2')
        # Get the layers of the model
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        self.shape = 2048  # the length of the feature vector

    def preprocess_image(self, image):
        """
        Preprocesses the image for ResNet50 by resizing, converting to tensor, 
        and normalizing it with ImageNet statistics.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def extract_features(self, image):
        """
        Extracts features from the preprocessed image.
        The input image is expected to be a tensor of shape (1, 3, 224, 224).
        """
        # Pass the image through the ResNet50 model and get the feature maps
        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)

        # Return features as a numpy array
        return feature.cpu().detach().numpy()


class RGBHistogram():
    def __init__(self):
        self.shape = 768 # the length of the feature vector

    def extract_features(self, image):
        image = image.cpu().numpy()
        features = []
        for img in image:
            # Convert to format when reading images from CV2
            img *= 255
            img = img.reshape(img.shape[1], img.shape[2], img.shape[0])

            # Calculate the histogram of each color channel
            hist_red = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_green = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_blue = cv2.calcHist([img], [2], None, [256], [0, 256])

            # Normalize histogram
            cv2.normalize(hist_red, hist_red, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_green, hist_green, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_blue, hist_blue, 0, 1, cv2.NORM_MINMAX)
            
            # Merge histograms of color channels into a feature vector
            feature_vector = np.concatenate((hist_red, hist_green, hist_blue))
            feature_vector.resize(len(feature_vector))
            features.append(feature_vector)
        return np.array(features)

class LBP():
    def __init__(self):
        self.shape = 26 # the length of the feature vector

    def extract_features(self, image):
        n_points = 24
        radius = 3

        image = image.cpu().numpy()
        features = []
        for img in image:
            # Convert to format when reading images from CV2
            img *= 255
            img = img.reshape(img.shape[1], img.shape[2], img.shape[0])

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(gray, n_points, radius, method="default")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-7)

            features.append(hist)

        return np.array(features)