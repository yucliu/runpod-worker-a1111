FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      fonts-dejavu-core \
      rsync \
      git \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      libgoogle-perftools4 \
      libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Clone the repo
RUN git clone --depth=1 https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
# Create and activate venv
RUN cd stable-diffusion-webui
RUN python -m venv /workspace/venv
RUN source /workspace/venv/bin/activate

# Install Torch and xformers
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir xformers

# Install A1111 Web UI
RUN wget https://raw.githubusercontent.com/ashleykleynhans/runpod-worker-a1111/main/install-automatic.py
RUN python -m install-automatic --skip-torch-cuda-test

# Clone the ControlNet Extension
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet
# Install dependencies for ControlNet
RUN pip install -r extensions/sd-webui-controlnet/requirements.txt

RUN mkdir -p /workspace/stable-diffusion-webui/models/Stable-diffusion/qrcode
RUN wget -O /workspace/stable-diffusion-webui/models/Stable-diffusion/revAnimated_v122.safetensors https://civitai.com/api/download/models/46846
RUN wget -O /workspace/stable-diffusion-webui/models/ControlNet/control_v11f1e_sd15_tile.pth https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth
RUN wget -O /workspace/stable-diffusion-webui/models/ControlNet/control_v1p_sd15_brightness.safetensors https://huggingface.co/ioclab/ioc-controlnet/resolve/main/models/control_v1p_sd15_brightness.safetensors


# Add RunPod Handler and Docker container start script
RUN mkdir -p /workspace/logs
COPY start.sh rp_handler.py ./
COPY schemas /schemas

# Start the container
RUN chmod +x /start.sh
CMD /start.sh
