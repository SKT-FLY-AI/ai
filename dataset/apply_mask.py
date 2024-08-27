import os
import cv2
import numpy as np

def apply_mask_and_save(input_dir, output_dir):
    # 모든 하위 디렉토리를 순회
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                # 이미지 파일 경로
                image_path = os.path.join(root, file)
                
                # 마스크 파일 경로 설정 (여러 패턴 고려)
                if '_aug_' in file:
                    base_name = file.split('_aug_')[0]  # 'Type3_iter16_jpg.rf.9e027be09c5de406bceaefa6db73d601'
                    aug_part = file.split('_aug_')[1].split('.')[0]  # '32' or '131'
                    mask_file = f"{base_name}_mask_aug_{aug_part}.png"
                else:
                    base_name = file.rsplit('.', 1)[0]  # 'Type3_iter16_jpg.rf.9e027be09c5de406bceaefa6db73d601'
                    mask_file = f"{base_name}_mask.png"

                mask_path = os.path.join(root, mask_file)
                
                # 이미지와 마스크 불러오기
                if os.path.exists(mask_path):
                    image = cv2.imread(image_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # 마스크와 이미지 크기 일치화
                    if image.shape[:2] != mask.shape:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 마스크가 적용된 이미지를 만들기 위해 흰색 배경 이미지 생성
                    masked_image = np.ones_like(image) * 255  # 흰색 배경
                    masked_image[mask != 0] = image[mask != 0]  # 객체 부분은 원본 이미지에서 가져오기

                    # 출력 경로 만들기 (디렉토리 구조 유지)
                    relative_path = os.path.relpath(root, input_dir)
                    output_path = os.path.join(output_dir, relative_path)
                    os.makedirs(output_path, exist_ok=True)
                    
                    # 저장할 파일 이름
                    output_file_path = os.path.join(output_path, file)
                    
                    # 이미지 저장
                    cv2.imwrite(output_file_path, masked_image)
                    print(f"Saved: {output_file_path}")
                else:
                    print(f"Mask not found for: {image_path}")

# 예제 사용 방법
input_directory = '/root/ai/dataset/classification_aug_0816/test'
output_directory = '/root/ai/dataset/classification_aug_apply/test'

apply_mask_and_save(input_directory, output_directory)
