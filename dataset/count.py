import os

# 디렉토리 내 파일 개수를 출력하는 함수
def count_files_in_directory(directory):
    # 각 type 디렉토리 탐색
    for type_dir in os.listdir(directory):
        type_dir_path = os.path.join(directory, type_dir)
        
        if os.path.isdir(type_dir_path):
            # 해당 디렉토리의 모든 파일 가져오기
            files = os.listdir(type_dir_path)
            
            image_files = [f for f in files if f.endswith('.jpg')]
            
            # 결과 출력
            print(f"{type_dir} 디렉토리:")
            print(f"  이미지 파일: {len(image_files)}개")

# 메인 실행 코드
dataset_dir = '/root/ai/dataset/classification_aug_apply/train'  # 데이터셋 경로 설정
count_files_in_directory(dataset_dir)
