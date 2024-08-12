FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

COPY ./ /ai

WORKDIR /ai

RUN apt-get update & apt install -y libgl1-mesa-glx libglib2.0-0
RUN pip install -r requirements.txt