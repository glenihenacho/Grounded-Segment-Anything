FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set build environment flags
ENV DEBIAN_FRONTEND=noninteractive
ENV AM_I_DOCKER=true
ENV BUILD_WITH_CUDA=true
ENV CUDA_HOME=/usr/local/cuda-11.6/
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /home/appuser/Grounded-Segment-Anything
COPY . .

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 git nano vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python tools
RUN pip install --no-cache-dir wheel setuptools

# 🛠 Install GroundingDINO with fail-fast check
ENV PYTHONPATH="/home/appuser/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"
RUN pip install --no-cache-dir --no-build-isolation -e ./GroundingDINO || (echo '❌ GroundingDINO install failed' && exit 1)
RUN pip list | grep groundingdino || (echo '❌ groundingdino not in pip list' && exit 1)

# 🛠 Install segment_anything (editable mode optional)
RUN pip install --no-cache-dir --no-build-isolation -e ./segment_anything || (echo '❌ segment_anything install failed' && exit 1)

# Runtime packages
RUN pip install --no-cache-dir \
    diffusers[torch]==0.15.1 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    matplotlib==3.5.3 \
    onnxruntime==1.14.1 \
    onnx==1.13.1 \
    ipykernel==6.16.2 \
    scipy gradio openai

# 🔁 Download pretrained weights
RUN mkdir -p /weights
RUN wget -O /weights/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget -O /weights/groundingdino_swinb.pth https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swinb.pth

# ✅ Final confirmation: validate everything works
RUN python -c "import torch, groundingdino; print('✅ CUDA:', torch.cuda.is_available(), '| groundingdino loaded ✔')"
