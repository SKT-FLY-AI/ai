# Preprocess the images
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    return bbox

def get_bbox_coords(dataset_dir):
    # dataset_dir = '/root/ai/dataset/poo_1-299/train'
    bbox_coords = {}
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Process each image and mask
    for filename in os.listdir(dataset_dir):
        if filename.endswith("_mask.png"):
            mask_path = os.path.join(dataset_dir, filename)
            image_path = mask_path.replace("_mask.png", ".jpg")
            key, _ = os.path.splitext(os.path.basename(image_path))
            # Load mask and image
            mask = Image.open(mask_path).convert("L")
            image = Image.open(image_path).convert("RGB")
            # Convert mask to tensor
            mask_tensor = transform(mask).squeeze().numpy()
            # Get bounding box
            bbox = get_bounding_box(mask_tensor)

            bbox_coords[key] = np.array(bbox)
    print(f"len(bbox_coords.keys()) : {len(bbox_coords.keys())}")
    return bbox_coords

def get_ground_truth_masks(bbox_coords, dataset_dir):
    ground_truth_masks = {}
    for k in bbox_coords.keys():
        mask_path = os.path.join(dataset_dir, f"{k}_mask.png")
        if not os.path.exists(mask_path): # 파일 없을 경우, 에러 출력
            print(f"Exist Error : {mask_path}")
            return False            
        gt_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 원래 마스크가 검정 배경, 흰 객체 -> 흰 배경, 검정 객체로 변경 
        ground_truth_masks[k] = np.where(gt_grayscale == 0, 255, 0)
    print(f"len(ground_truth_masks.keys()) : {len(ground_truth_masks.keys())}")
    return ground_truth_masks

def preprocess(sam_model, bbox_coords, device, dataset_dir):
    transformed_data = defaultdict(dict)
    for k in bbox_coords.keys():
        image_path = os.path.join(dataset_dir, f"{k}.jpg")
        if not os.path.exists(image_path): # 파일 없을 경우, 에러 출력
            print(f"Exist Error : {image_path}")
            return False        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[k]['image'] = input_image
        transformed_data[k]['input_size'] = input_size
        transformed_data[k]['original_image_size'] = original_image_size
    print(f"make transformed_data")
    return transformed_data, transform