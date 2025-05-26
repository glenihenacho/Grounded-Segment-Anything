FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set environment variables for CUDA + PyTorch compatibility
ENV DEBIAN_FRONTEND=noninteractive
ENV AM_I_DOCKER=true
ENV BUILD_WITH_CUDA=true
ENV CUDA_HOME=/usr/local/cuda-11.6/

# Create working directory
WORKDIR /home/appuser/Grounded-Segment-Anything
COPY . .

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 git nano vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Confirm directory structure
RUN echo "📁 Verifying build context:" && ls -la && ls -la GroundingDINO

# Python essentials
RUN pip install --no-cache-dir wheel setuptools

# Install local editable libraries
RUN pip install --no-cache-dir --no-build-isolation -e ./GroundingDINO
RUN pip install --no-cache-dir --no-build-isolation -e ./segment_anything

# Install remaining libraries
RUN pip install --no-cache-dir \
    diffusers[torch]==0.15.1 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    matplotlib==3.5.3 \
    onnxruntime==1.14.1 \
    onnx==1.13.1 \
    ipykernel==6.16.2 \
    scipy gradio openai

# Download model weights (SAM and GroundingDINO)
RUN mkdir -p /weights
RUN wget -O /weights/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget -O /weights/groundingdino_swinb.pth https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swinb.pth

# Optional: verify imports before shipping the image
RUN python -c "print('🔥 Import test...'); import torch, groundingdino; print('✅ Torch:', torch.__version__, '| groundingdino loaded')"
