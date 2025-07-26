# Start with a base Ubuntu image with CUDA and cuDNN pre-installed
# Using CUDA 12.1 as it's compatible with PyTorch 2.5.1 and closest to the required 12.4
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip \
    git \
    vim \
    wget \
    cmake \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sv /usr/bin/python3.10 /usr/bin/python
RUN ln -svf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set the working directory
WORKDIR /exp

# Create a new virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create and set permissions for system directories
RUN mkdir -p /.cache /.local /.config /.jupyter && \
    chmod -R 777 /.cache && \
    chmod -R 777 /.local && \
    chmod -R 777 /.config && \
    chmod -R 777 /.jupyter

# Copy requirements file first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies with specific versions as mentioned in README
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    numpy==2.2.0 \
    opencv-python==4.10.0.84 \
    scikit-image>=0.20.0 \
    pyyaml \
    tensorboard \
    matplotlib \
    scipy \
    pillow \
    tqdm

# Copy project files into the container
COPY . /exp/

# Install the package in development mode
RUN pip install -e .

# Create directories for datasets and output
RUN mkdir -p /exp/datasets/train /exp/datasets/validation /exp/datasets/test /exp/output

# Set proper permissions
RUN chmod -R 755 /exp

# Set default command
CMD ["bash"]
