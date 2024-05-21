import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

# Load the DeepLabV3 model
def load_deeplab_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    return model

# Segment the image to get the foreground mask
def segment_image(image, model, confidence_threshold=0.01):
    # Ensure the image has 3 channels (RGB)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).copy()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image_bgr)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # Get confidence scores
    confidence_scores = torch.softmax(output, dim=0).cpu().numpy()

    # Create a combined mask for all classes
    combined_mask = np.zeros(confidence_scores.shape[1:], dtype=np.uint8)
    # for class_index in range(confidence_scores.shape[0]):
    index_list = [2,6,7,14,15,19,20]
    for class_index in index_list:
        class_mask = (confidence_scores[class_index] > confidence_threshold).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, class_mask * 255)

    # Resize the mask back to the original image size
    combined_mask = cv2.resize(combined_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask

# Extract the foreground from the image using the mask
def extract_foreground(image, mask):
    # Ensure the mask is binary (0 or 255) and of type uint8
    mask = (mask > 0).astype(np.uint8) * 255
    foreground = cv2.bitwise_and(image, mask)
    return foreground