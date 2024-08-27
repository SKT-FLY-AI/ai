import os
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# 화이트 밸런스 적용 함수
def apply_white_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_mean = np.mean(l)
    mask = l > l_mean  # Mask of bright areas
    avg_a = np.mean(a[mask])
    avg_b = np.mean(b[mask])
    a_adjusted = a - ((avg_a - 128) * (l / 255.0) * 1.1)
    b_adjusted = b - ((avg_b - 128) * (l / 255.0) * 1.1)
    a_adjusted = np.clip(a_adjusted, 0, 255).astype(np.uint8)
    b_adjusted = np.clip(b_adjusted, 0, 255).astype(np.uint8)
    lab_adjusted = cv2.merge([l, a_adjusted, b_adjusted])
    balanced_image = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    return balanced_image


# 특정 색상 계열로 부드럽게 제한하는 함수
def apply_soft_color_tone(image, target_colors, blend_factor_range=(0.2, 0.4)):
    image = image.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    chosen_color = (
        np.array(target_colors[np.random.randint(len(target_colors))], dtype=np.float32)
        / 255.0
    )
    chosen_color = np.ones((h, w, 3), dtype=np.float32) * chosen_color
    chosen_color = chosen_color[..., ::-1]
    blend_factor = np.random.uniform(*blend_factor_range)
    image = cv2.addWeighted(image, 1 - blend_factor, chosen_color, blend_factor, 0)
    return (image * 255).astype(np.uint8)


# 데이터 증강 함수
def augment_and_save(
    image,
    mask,
    target_dir,
    base_filename,
    start_idx,
    apply_color_tone=True,
    apply_color_tone_probability=0.2,
):
    target_colors = [
        (139, 69, 19),  # 갈색
        (255, 223, 0),  # 노란색
        (220, 20, 60),  # 붉은색
        (0, 128, 0),  # 녹색
        (0, 0, 0),  # 검은색
    ]

    transform = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomRotate90(),
                    A.Flip(),
                    A.Transpose(),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                ],
                p=0.5,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            ToTensorV2(),
        ]
    )

    # 화이트 밸런스 적용
    image = apply_white_balance(image)

    # 증강 적용
    augmented = transform(image=image)["image"]
    augmented_image = augmented.numpy().transpose(1, 2, 0)

    # 부드러운 색상 혼합 적용
    if apply_color_tone and np.random.rand() < apply_color_tone_probability:
        augmented_image = apply_soft_color_tone(augmented_image, target_colors)

    new_filename_img = f"{base_filename}_aug_{start_idx}.jpg"
    new_filename_mask = f"{base_filename}_mask_aug_{start_idx}.png"
    new_filepath_img = os.path.join(target_dir, new_filename_img)
    new_filepath_mask = os.path.join(target_dir, new_filename_mask)

    cv2.imwrite(new_filepath_img, augmented_image)
    cv2.imwrite(new_filepath_mask, mask)


# 데이터 복사 함수 (300개 이상인 경우)
def copy_images(class_dir, output_class_dir):
    images = [
        f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))
    ]
    for image_file in images:
        src_path = os.path.join(class_dir, image_file)
        dst_path = os.path.join(output_class_dir, image_file)
        shutil.copy2(src_path, dst_path)


# 데이터 증강을 적용할 함수
def apply_augmentation(
    dataset_dir,
    output_dir,
    target_class,
    target_count,
    apply_color_tone=True,
    apply_color_tone_probability=0.5,
):
    class_dir = os.path.join(dataset_dir, target_class)
    output_class_dir = os.path.join(output_dir, target_class)
    os.makedirs(output_class_dir, exist_ok=True)
    images = [
        f
        for f in os.listdir(class_dir)
        if f.endswith(".jpg") and os.path.isfile(os.path.join(class_dir, f))
    ]
    num_images = len(images)

    if num_images >= target_count:
        print(
            f"{target_class} 클래스는 이미 {target_count}장 이상 존재합니다. 이미지를 복사합니다."
        )
        copy_images(class_dir, output_class_dir)
        return

    num_augmentations = target_count - num_images
    if num_augmentations <= 0:
        return

    if num_augmentations > num_images:
        selected_images = random.choices(images, k=num_augmentations)  # 중복 허용
    else:
        selected_images = random.sample(images, num_augmentations)  # 중복 허용 안 함

    print(
        f"{target_class} 클래스에 대해 {num_augmentations}개의 증강 이미지 생성 중..."
    )

    for i, image_file in enumerate(tqdm(selected_images)):
        image_path = os.path.join(class_dir, image_file)
        mask_path = os.path.join(
            class_dir, os.path.splitext(image_file)[0] + "_mask.png"
        )

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        augment_and_save(
            image,
            mask,
            output_class_dir,
            os.path.splitext(image_file)[0],
            i,
            apply_color_tone=apply_color_tone,
            apply_color_tone_probability=apply_color_tone_probability,
        )


# 데이터셋의 각 클래스에 대해 증강 적용
dataset_dir = "dataset_noaug/train"
output_dir = "dataset_aug_0816/train"
target_classes = ["1", "2", "3", "4", "5", "6", "7"]
target_count = 300

for target_class in target_classes:
    apply_augmentation(
        dataset_dir,
        output_dir,
        target_class,
        target_count,
        apply_color_tone=True,
        apply_color_tone_probability=0.2,
    )
