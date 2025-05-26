FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set working env
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME /usr/local/cuda-11.6/

# Create base directory
RUN mkdir -p /home/appuser/Grounded-Segment-Anything
COPY . /home/appuser/Grounded-Segment-Anything
WORKDIR /home/appuser/Grounded-Segment-Anything

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 git nano vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python setup
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir --no-build-isolation -e GroundingDINO
RUN pip install --no-cache-dir -e segment_anything
RUN pip install --no-cache-dir \
    diffusers[torch]==0.15.1 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    matplotlib==3.5.3 \
    onnxruntime==1.14.1 \
    onnx==1.13.1 \
    ipykernel==6.16.2 \
    scipy gradio openai

# Download model weights
RUN mkdir -p /weights
RUN wget -O /weights/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget -O /weights/groundingdino_swinb.pth https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swinb.pth

# Simple sanity check to verify environment
RUN python -c "print('✅ Docker container built successfully. Torch version:'); import torch; print(torch.__version__)"
