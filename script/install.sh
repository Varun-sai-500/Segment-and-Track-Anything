# better create a virtual environment with python 3.10 especially and activate it
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# If your CUDA version is different, use the matching command from:
# https://pytorch.org/get-started/locally/

python -m pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

python -m pip install \
transformers==5.14.1 \
timm==1.0.28 \
opencv-python-headless==5.0.0.93 \
pillow==12.3.0 \
gradio==6.22.0 \
gdown==6.1.0

