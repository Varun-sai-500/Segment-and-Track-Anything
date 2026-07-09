FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN pip install --upgrade pip

RUN pip install \
transformers==5.13.0 \
timm==1.0.27 \
opencv-python-headless==5.0.0.93 \
Pillow==12.3.0 \
gradio==6.20.0 \
gdown==6.1.0

RUN pip install git+https://github.com/facebookresearch/segment-anything.git

CMD ["python", "app.py"]