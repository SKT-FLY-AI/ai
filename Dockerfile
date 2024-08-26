FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

COPY ./ /ai

WORKDIR /ai
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install git
ENV DEBIAN_FRONTEND noninteractive
RUN apt install -y libgl1-mesa-glx libglib2.0-0
RUN cd mmpretrain && pip install -U openmim && mim install -e .
RUN pip install --no-cache-dir -r requirements.txt

# CMD로 쉘 명령어를 사용하여 두 개의 스크립트를 순차적으로 실행
CMD ["bash", "-c", "python download_weights.py && python main.py"]

