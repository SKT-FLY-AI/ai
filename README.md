# Poopy-AI
- 애견 대변의 특징을 자동으로 분류하는 AI 모듈입니다.
- Custom SAM을 이용하여 배경과 대변을 분할하고, Dog Fecal Chart에 따라 7단계의 분류와 K-means 색상 양자화(n=5)를 진행합니다.
  
## Architecture
### Promptless-SAM Module
![image](https://github.com/user-attachments/assets/b48b034e-87b2-4095-b8fd-72e51aaf5909)
### Classification and Color Quantization
![image](https://github.com/user-attachments/assets/9df06529-506b-4da7-8120-8019a5f7de6b)

------------------
### Usage
1. Build Dockerfile
2. Run Docker
```
docker run -d -it --gpus all --ipc=host -p 8891:8891 {imagename:version}
```
