FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel

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

RUN python -m pip install --upgrade pip

RUN pip install \
transformers==5.14.1 \
timm==1.0.28 \
opencv-python-headless==5.0.0.93 \
Pillow==12.3.0 \
gradio==6.22.0 \
gdown==6.1.0

RUN pip install git+https://github.com/facebookresearch/segment-anything.git

CMD ["python", "app.py"]