import os
import gdown
import zipfile


def download_Zip(data_path, output, quiet=False):
    if os.path.exists(output):
        print(output + " already exists!")
        return
    gdown.download(data_path, output=output, quiet=quiet)


def extract_Zip(zip_path, output_path):
    print("Start extracting " + zip_path)
    with zipfile.ZipFile(zip_path) as file:
        if os.path.exists(output_path) and os.path.isdir(output_path):
            sub_dirs = [subDir for subDir in os.listdir(
                output_path) if os.path.isdir(os.path.join(output_path, subDir))]
            dir_name = zip_path.split('/')[-1].split('.')[0]
            for sub_dir in sub_dirs:
                if sub_dir == dir_name:
                    print(dir_name + " directory already exists")
                    return
        file.extractall(path=output_path)
        print("Successfully extracted " + zip_path)


if __name__ == "__main__":
    google_path = 'https://drive.google.com/uc?id='

    weights_save_folder = "/ai/"
    weights = '190gVtsvpkB6obW1p36Abns48LR1fWW1l'
    weights_name = 'pc_weights.zip'

    # 폴더 내에 파일이 이미 존재하는지 확인
    if os.path.exists(weights_save_folder) and os.listdir("weights"):
        print("Weights folder already contains files. Skipping download.")
    else:
        # 폴더가 없으면 생성
        # if not os.path.exists(weights_save_folder):
        #     os.makedirs(weights_save_folder)

        # 다운로드 및 압축 해제
        download_Zip(google_path + weights, weights_save_folder + weights_name)
        extract_Zip(weights_save_folder + weights_name, weights_save_folder)
