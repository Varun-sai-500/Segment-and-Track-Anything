FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

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

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
