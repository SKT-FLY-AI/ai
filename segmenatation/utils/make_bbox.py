import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt

def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    return bbox

def draw_bounding_box(image, bbox):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=2)
    return image

if __name__ == '__main__':
    # Paths
    dataset_dir = '/root/ai/dataset/poo_1-299/train'
    output_dir = '/root/ai/dataset/poo_1-299/bbox'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Process each image and mask
    for filename in os.listdir(dataset_dir):
        if filename.endswith("_mask.png"):
            mask_path = os.path.join(dataset_dir, filename)
            image_path = mask_path.replace("_mask.png", ".jpg")

            # Load mask and image
            mask = Image.open(mask_path).convert("L")
            image = Image.open(image_path).convert("RGB")

            # Convert mask to tensor
            mask_tensor = transform(mask).squeeze().numpy()

            # Get bounding box
            bbox = get_bounding_box(mask_tensor)

            # Draw bounding box on image
            image_with_bbox = draw_bounding_box(image, bbox)

            # Save the image with bbox
            output_image_path = os.path.join(output_dir, filename.replace("_mask.png", "_bbox.png"))
            image_with_bbox.save(output_image_path)
            print(f"Save bbox : {output_image_path}")

    print("Processing complete.")
