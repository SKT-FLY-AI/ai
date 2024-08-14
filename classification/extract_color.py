import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def rgb2hex(rgb):
    rgb = rgb[::-1]
    rgb_int = tuple(int(x) for x in rgb)
    return '#{:02x}{:02x}{:02x}'.format(rgb_int[0], rgb_int[1], rgb_int[2])

def calculate_average_color_path(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    mean_val = cv2.mean(masked_image, mask=mask)[:3]
    mean_val_bgr = np.array(mean_val)
    
    return mean_val_bgr, masked_image

def calAvgColor(masked_image):
    non_white_mask = np.all(masked_image != [255, 255, 255], axis=-1)
    # 흰색이 아닌 픽셀들만 평균 색상 계산
    selected_pixels = masked_image[non_white_mask]
    mean_val_rgb = np.mean(selected_pixels, axis=0)
    # RGB -> BGR 순서로 변경
    mean_val_bgr = mean_val_rgb[::-1]
    return mean_val_bgr, masked_image

def calMixtureColor(mean_val_bgr, red_color, alpha=0.5):
    mean_val_bgr = np.asarray(mean_val_bgr, dtype=float)
    red_color = np.asarray(red_color, dtype=float)
    mixed_color = (alpha * mean_val_bgr + (1 - alpha) * red_color).astype(int)
    return mixed_color

def checkBlood(mean_val_bgr, masked_image, thresholds, pixel_threshold=10, alpha=0.5):
    red_colors = [
        np.array([0, 0, 102]),  # #660000
        np.array([0, 0, 139]),  # #8B0000
        np.array([0, 0, 128]),  # #800000
        np.array([0, 17, 204]), # #CC1100
        np.array([60, 20, 220]) # #DC143C
    ]
    
    min_distance = float('inf')
    closest_mixed_color = None
    
    # 전체 평균 색상에 대한 검사
    for red_color in red_colors:
        mixed_color = calMixtureColor(mean_val_bgr, red_color, alpha)
        distance = np.linalg.norm(mean_val_bgr - mixed_color)
        if distance < min_distance:
            min_distance = distance
            closest_mixed_color = mixed_color
            
    # 특정 임계값 이하의 색상이 있으면 True 반환
    if min_distance < thresholds:
        return True, closest_mixed_color
    
    # 픽셀 단위 검사 (국소적 혈변 감지)
    red_count = 0
    for y in range(masked_image.shape[0]):
        for x in range(masked_image.shape[1]):
            pixel = masked_image[y, x, :]
            if not np.all(pixel == [255, 255, 255]):  # 배경이 아닌 픽셀만 고려
                for red_color in red_colors:
                    distance = np.linalg.norm(pixel - red_color)
                    if distance < thresholds:
                        red_count += 1
                        if red_count >= pixel_threshold:
                            return True, closest_mixed_color
    
    return False, closest_mixed_color

if __name__ == "__main__":
    
    image_path = "/root/ai/dataset/classification/train/4/Type3_iter348_jpg.rf.0996ba2bab045b26264af1a3418c7f19.jpg"
    mask_path = "/root/ai/dataset/classification/train/4/Type3_iter348_jpg.rf.0996ba2bab045b26264af1a3418c7f19_mask.png"

    threshold = 100

    mean_val_bgr, masked_image = calculate_average_color_path(image_path, mask_path)
    hemorrhage_present, closest_mixed_color = checkBlood(mean_val_bgr, masked_image, threshold)
