import gdown
import os
import shutil
import zipfile
import tarfile

def download_and_extract_drive_file(drive_url, dest_folder, extract_folder=None):
    """
    Google Drive에서 파일을 다운로드하고 지정된 폴더로 이동한 후 압축을 풉니다.

    :param drive_url: Google Drive 파일의 공유 링크 또는 gdrive 파일 ID
    :param dest_folder: 파일을 저장할 폴더 경로
    :param extract_folder: 압축을 풀 폴더 경로 (기본적으로 dest_folder와 동일)
    """
    # 파일 다운로드
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # gdown으로 파일 다운로드
    filename = gdown.download(drive_url, downloaded_file, quiet=False)
    downloaded_file = os.path.join(dest_folder, filename)

    # 압축 풀기
    if extract_folder is None:
        extract_folder = dest_folder

    if downloaded_file.endswith('.zip'):
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
    elif downloaded_file.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(downloaded_file, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_folder)
    elif downloaded_file.endswith('.tar'):
        with tarfile.open(downloaded_file, 'r') as tar_ref:
            tar_ref.extractall(extract_folder)
    else:
        print("압축 파일이 아닙니다. 다운로드된 파일은 이동만 됩니다.")
        shutil.move(downloaded_file, extract_folder)

    # 다운로드된 압축 파일 삭제
    os.remove(downloaded_file)
    print(f"{extract_folder}에 파일이 성공적으로 압축 해제되었습니다.")

def main():
    # 다운로드할 파일 리스트 (Google Drive 파일 ID 또는 링크)
    segment_url = "https://drive.google.com/uc?id=1zgTHF9F6wkhPgM3Xx8COsdOUAn868coc"
    swin_aug_url = "https://drive.google.com/uc?id=1wN2qqZhKs9pLp6tGmmmK8Yi7hsvpN-is"

    weights_path = 'weights/'  # 각 파일에 대해 다운로드 폴더 생성
    
    mm_weights_path = "mmpretrain/work_dir/"
    
    download_and_extract_drive_file(segment_url, weights_path, weights_path)
    download_and_extract_drive_file(swin_aug_url, mm_weights_path, mm_weights_path)

if __name__ == "__main__":
    main()
