import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# # 25개의 색상 코드를 정의합니다.
color_codes = ["#F4F0E5", "#D2C6B4", "#AE9C88", "#8E7861", "#6E5841", "#513A28", 
               "#342113", "#180B05", "#010101", "#D7C384", "#B7A05D", "#9B7A4D", 
               "#845B3B", "#664B2E", "#534228", "#3D2C18", "#B49C1E", "#A07429", 
               "#8A4F2D", "#6F5232", "#4E5534", "#B26726", "#89392E", "#84432F", 
               "#AC192A", "#AD1A2A", "#1D4F36", "#000000", "#D0CECF", 
               "#840000", "#6D0000" # DarkRed
               ]

# 위험 색상 목록을 정의합니다.
danger_colors = [
    "#F4F0E5", # 하양
    "#D2C6B4", "#AE9C88", "#8E7861", "#D7C384", "#B7A05D", "#B49C1E", # 노란 계열 
    "#AD1A2A", "#AC192A", # 선홍 빨강
    "#840000", "#6D0000", # Dark Red
    "#1D4F36", "#4E5534", # 초록
    "#000000", "#342113", "#180B05", "#010101", "#534228", "#3D2C18", # 검은색 계열
    "#D0CECF" # 회색
]



def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

def perceptual_color_quantization(image, color_palette):
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 색변환
    quantized_image = image.copy()  # 입력 이미지를 복사하여 양자화된 이미지를 생성
    
    white_threshold = np.array([245, 245, 245])  # 하얀색으로 간주할 임계값 설정
    
    for y in range(height):
        for x in range(width):
            pixel_color = image[y, x]
            if np.all(pixel_color >= white_threshold):  # 하얀색 배경은 처리하지 않음
                continue
            closest_color = min(color_palette, key=lambda color: np.linalg.norm(pixel_color - color))
            quantized_image[y, x] = closest_color
    return quantized_image

def analyze_image_colors(quantized_image, danger_colors, threshold=5.0):
    # 하얀색으로 간주할 임계값 설정 (예: 약간의 여유를 둠)
    white_threshold = np.array([245, 245, 245])

    # 이미지를 2차원 배열로 평평하게 만듭니다.
    flat_image = quantized_image.reshape(-1, 3)
    
    # 하얀색이 아닌 픽셀만 선택
    non_white_pixels = [tuple(pixel) for pixel in flat_image if not np.all(pixel >= white_threshold)]
    
    # 각 색상의 출현 빈도를 계산합니다.
    color_counts = Counter(non_white_pixels)
    if not color_counts:
        # 비하얀 픽셀이 없을 경우 처리: 여기서는 None으로 반환하거나 기본값 설정
        return None, [], {}

    # 가장 많이 나타나는 색상(대표 색상)을 찾습니다.
    most_common_color = color_counts.most_common(1)[0][0]
    
    # 위험 색상 존재 여부 및 비율 확인
    detected_danger_colors = []
    danger_color_percentages = {}
    total_non_white_pixels = len(non_white_pixels)

    for color in danger_colors:
        color_tuple = tuple(color)
        if color_tuple in color_counts:
            # 위험 색상이 차지하는 비율 계산 (백분율)
            percentage = (color_counts[color_tuple] / total_non_white_pixels) * 100
            danger_color_percentages[color_tuple] = percentage
            
            # 임계값을 넘는 경우에만 위험 색상으로 추가
            if percentage >= threshold:
                detected_danger_colors.append(color_tuple)
    
    return most_common_color, detected_danger_colors, danger_color_percentages

def analyze_image(image, color_codes_rgb, danger_colors_rgb):
    # 이미지를 주어진 팔레트로 양자화
    quantized_image = perceptual_color_quantization(image, color_codes_rgb)
    
    # 함수 호출 예시
    threshold_value = 15.0  # 10% 이상의 색상만 위험 등급에 포함
    most_common_color, detected_danger_colors, danger_color_percentages = analyze_image_colors(quantized_image, danger_colors_rgb, threshold=threshold_value)

    print(f"대표 색상: {most_common_color}")
    print(f"감지된 위험 색상: {detected_danger_colors}")
    print(f"위험 색상 비율: {danger_color_percentages}")
    # RGB 값을 색상 코드로 변환
    most_common_color_hex = rgb_to_hex(most_common_color) if most_common_color else None
    detected_danger_colors_hex = [rgb_to_hex(color) for color in detected_danger_colors]
    
    # 결과를 반환합니다.
    return most_common_color, detected_danger_colors, most_common_color_hex, detected_danger_colors_hex


if __name__ == "__main__":
    # OpenCV에서는 BGR을 사용하지만, 여기서는 RGB로 반환합니다.
    color_codes_rgb = [hex_to_rgb(code) for code in color_codes]
    danger_colors_rgb = [hex_to_rgb(code) for code in danger_colors]
    image = cv2.imread("/root/ai/dataset/human_poo/classification_aug_apply/train/1/Type1_iter3_jpg.rf.120ab2fecf2da5c5636192b9a1f1e4af_aug_93.jpg")
    # 이미지에서 색상 분석을 수행합니다.
    mc, dc, most_common_color, detected_danger_colors = analyze_image(image, color_codes_rgb, danger_colors_rgb)

    print(f"대표 색상: {most_common_color}")
    print(f"감지된 위험 색상: {detected_danger_colors}")