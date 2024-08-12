import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
import numpy as np
from segment_anything.utils.transforms import ResizeLongestSide

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

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Transformation
        transform_bbox = transforms.Compose([
            transforms.ToTensor()
        ])
        # Convert mask to tensor
        mask_tensor = transform_bbox(mask).squeeze().numpy()
        # Get bounding box
        bbox = get_bounding_box(mask_tensor)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            bbox = self.transform(bbox)
            
        return image, mask, bbox


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.base_path = base_path
        self.resize = transforms.Resize(
            (1024, 1024),
            interpolation=transforms.InterpolationMode.NEAREST
        )
        
        # 모든 이미지 파일 수집
        self.all_img_files = glob.glob(os.path.join(self.base_path, self.image_folder, "*.jpg"))
        
        # 대응되는 마스크 파일이 있는 이미지 파일만 필터링
        self.img_files = []
        for img_path in self.all_img_files:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.base_path, self.mask_folder, f"{filename}_mask.png")
            if os.path.exists(mask_path):
                self.img_files.append(img_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image_path = self.img_files[index]
        
        # get the mask path
        mask_name = os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        # mask_name = mask_name.replace("frame", "mask")
        mask_path = os.path.join(self.base_path, self.mask_folder, mask_name)
        
        # read both image and mask path
        image = io.read_image(image_path)
        mask = io.read_image(mask_path)
        
        # resizing the image and mask
        image = self.resize(image)
        mask = self.resize(mask)
        
        # chaging dtype of mask
        mask = mask.type(torch.float)
        image = image.type(torch.float)
        
        # standardizing the mask between 0 and 1
        mask = mask/255
    
        return image, mask