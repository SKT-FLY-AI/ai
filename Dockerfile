FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

COPY ./ /ai

WORKDIR /ai

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install git
RUN apt install -y libgl1-mesa-glx libglib2.0-0
RUN pip install -r requirements.txt