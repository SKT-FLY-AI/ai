import os
import shutil
import random

# 데이터셋 디렉토리 경로
dataset_dir = "C:/Users/SKT007/Desktop/똥데이터/PoosSee_Dataset_Aug/mask/type6"
output_dir = "C:/Users/SKT007/Desktop/똥데이터/dataset_cls/"

# 비율 설정
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 파일 리스트 가져오기
files = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
random.shuffle(files)  # 파일을 랜덤하게 섞기

# 각 세트의 크기 계산
total_files = len(files)
train_size = int(total_files * train_ratio)
valid_size = int(total_files * valid_ratio)
test_size = total_files - train_size - valid_size

# 파일을 나누기
train_files = files[:train_size]
valid_files = files[train_size : train_size + valid_size]
test_files = files[train_size + valid_size :]

# 폴더 생성
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)


# 파일 복사 함수
def copy_files(file_list, folder_name):
    for file in file_list:
        base_name = os.path.splitext(file)[0]
        img_file = file
        mask_file = f"{base_name}_mask.png"

        shutil.copy(
            os.path.join(dataset_dir, img_file),
            os.path.join(output_dir, folder_name, img_file),
        )
        shutil.copy(
            os.path.join(dataset_dir, mask_file),
            os.path.join(output_dir, folder_name, mask_file),
        )


# 파일 복사
copy_files(train_files, "train")
copy_files(valid_files, "valid")
copy_files(test_files, "test")

print(f"Train set: {len(train_files)} files")
print(f"Validation set: {len(valid_files)} files")
print(f"Test set: {len(test_files)} files")
