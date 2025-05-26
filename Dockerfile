FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV AM_I_DOCKER=true
ENV BUILD_WITH_CUDA=true
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH="/app/GroundingDINO:$PYTHONPATH"

# Install Python 3 + pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip wget git ffmpeg libsm6 libxext6 nano && \
    rm -rf /var/lib/apt/lists/*

# Set default python to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /app
COPY . .

# Install PyTorch for CUDA 11.6 (stable)
RUN pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install GroundingDINO + SAM + required packages
RUN pip install --no-cache-dir wheel setuptools \
 && pip install --no-cache-dir --no-build-isolation -e ./GroundingDINO \
 && pip install --no-cache-dir --no-build-isolation -e ./segment_anything \
 && pip install --no-cache-dir \
      diffusers[torch]==0.15.1 \
      opencv-python==4.7.0.72 \
      pycocotools==2.0.6 \
      matplotlib==3.5.3 \
      onnxruntime==1.14.1 \
      onnx==1.13.1 \
      scipy gradio openai \
 && pip cache purge

# Download pretrained weights
RUN mkdir -p /weights && \
    wget -O /weights/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
    wget -O /weights/groundingdino_swinb.pth https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swinb.pth

# Final sanity check
RUN python -c "import torch, groundingdino; print('✅ torch + groundingdino OK. CUDA:', torch.cuda.is_available())"
