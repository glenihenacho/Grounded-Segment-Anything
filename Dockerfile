FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV AM_I_DOCKER=true
ENV BUILD_WITH_CUDA=true
ENV CUDA_HOME=/usr/local/cuda-11.6/
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/home/appuser/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# Create working directory
WORKDIR /home/appuser/Grounded-Segment-Anything
COPY . .

# Install OS dependencies + clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 git nano vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install all Python deps together and clean cache after
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
      ipykernel==6.16.2 \
      scipy gradio openai \
 && pip cache purge

# Download model weights
RUN mkdir -p /weights && \
    wget -O /weights/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
    wget -O /weights/groundingdino_swinb.pth https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swinb.pth

# Final check
RUN python -c "import torch, groundingdino; print('✅ torch + groundingdino loaded. CUDA:', torch.cuda.is_available())"
