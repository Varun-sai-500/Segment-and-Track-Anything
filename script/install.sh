# better create a virtual environment with python 3.10 especially and activate it
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything

# or git clone https://github.com/facebookresearch/segment-anything and cd sam and pip install -e .

# If your CUDA version is different, use the matching command from:
# https://pytorch.org/get-started/locally/

python -m pip install torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
transformers==5.13.0 \
timm==1.0.27 \
opencv-python-headless==5.0.0.93 \
Pillow==12.3.0 \
gradio==6.20.0 \
gdown==6.1.0

