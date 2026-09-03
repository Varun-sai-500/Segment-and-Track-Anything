FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime

<<<<<<< HEAD
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
=======
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
>>>>>>> c4992261d0d1c6e0a6e3f2c0eec9e65c78474987

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --break-system-packages --upgrade pip && \
    python -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY . .

<<<<<<< HEAD
CMD ["python", "app.py"]
=======
RUN python -m pip install --upgrade pip setuptools==80.9.0 wheel

RUN pip install --no-cache-dir \
    transformers==4.30.2 \
    addict==2.4.0 \
    yapf==0.40.2 \
    timm==1.0.27 \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    Pillow==10.4.0 \
    matplotlib==3.9.2 \
    supervision==0.22.0 \
    pycocotools==2.0.8 \
    soundfile==0.13.1 \
    gradio==3.39.0 \
    gradio_client==0.5.0 \
    pydantic==1.10.13 \
    fastapi==0.100.1 \
    starlette==0.27.0 \
    gdown==6.1.0

RUN pip install -e sam

RUN git clone -b main https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && \
    pip install -e . --no-build-isolation


CMD ["python", "app.py"]
>>>>>>> c4992261d0d1c6e0a6e3f2c0eec9e65c78474987
