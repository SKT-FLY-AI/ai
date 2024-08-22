import os
import shutil
from PIL import Image
import numpy as np
import torch

# 원본 데이터셋 디렉토리
dataset_dir = '/root/ai/dataset/poo_1-299/train'

# 타겟 디렉토리
target_dir = '/root/ai/dataset/bristol_type_1-299'

# 클래스에 맞는 타겟 디렉토리 생성
for i in range(1, 8):
    os.makedirs(os.path.join(target_dir, f'type{i}'), exist_ok=True)

# 파일을 이동하는 함수
def move_files(image_path, mask_path, class_type):
    target_path = os.path.join(target_dir, f'type{class_type}')
    shutil.copy(image_path, target_path)
    shutil.copy(mask_path, target_path)

# 데이터셋 디렉토리 내 파일을 확인하고 파일 이동
for filename in os.listdir(dataset_dir):
    if filename.endswith('_mask.png'):
        mask_path = os.path.join(dataset_dir, filename)
        image_path = mask_path.replace('_mask.png', '.jpg')
        
        if not os.path.exists(image_path):
            continue

        # 마스크 이미지 로드
        mask_image = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_image)
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

        # 텐서 내의 고유한 값들 확인
        unique_values = torch.unique(mask_tensor)
        unique_values = unique_values[unique_values != 0]  # 0을 제외한 값

        if len(unique_values) == 1:
            class_type = int(unique_values.item())
            move_files(image_path, mask_path, class_type)

print("파일 이동이 완료되었습니다.")
