FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set environment flags
ENV DEBIAN_FRONTEND=noninteractive
ENV AM_I_DOCKER=true
ENV BUILD_WITH_CUDA=true
ENV CUDA_HOME=/usr/local/cuda-11.6/

# Set working directory
WORKDIR /app
COPY . .

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 git nano vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python build tools
RUN pip install --no-cache-dir wheel setuptools

# Install GroundingDINO (editable mode with path fix)
RUN pip install --no-cache-dir --no-build-isolation -e ./GroundingDINO || (echo "❌ GroundingDINO install failed" && exit 1)

# Confirm it's installed
RUN pip list | grep groundingdino || (echo "❌ groundingdino not found in pip list" && exit 1)

# Install segment-anything
RUN pip install --no-cache-dir --no-build-isolation -e ./segment_anything || (echo "❌ segment_anything install failed" && exit 1)

# Other Python libs
RUN pip install --no-cache-dir \
    diffusers[torch]==0.15.1 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    matplotlib==3.5.3 \
    onnxruntime==1.14.1 \
    onnx==1.13.1 \
    ipykernel==6.16.2 \
    scipy gradio openai

# Model Weights
RUN mkdir -p /weights
RUN wget -O /weights/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget -O /weights/groundingdino_swinb.pth https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swinb.pth

# Final sanity check
RUN python -c "import groundingdino; print('✅ groundingdino successfully imported')"
