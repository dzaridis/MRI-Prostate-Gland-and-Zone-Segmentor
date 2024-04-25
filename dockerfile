# Use an NVIDIA CUDA base image with CUDA 11.7.0 and cuDNN support
FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

# Avoid interactive dialog (e.g., tzdata)
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    libgdbm-dev \
    wget \
    libc6-dev

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install Python 3.9.18 using pyenv
RUN pyenv install 3.9.18 && pyenv global 3.9.18

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory in the container to root
WORKDIR /

# Copy the current directory contents into the container at root
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define mountable directories
VOLUME ["/Pats", "/Outputs"]

# Run model.py when the container launches
CMD ["python", "./__main__.py"]

# Reset the frontend variable - good practice to keep Docker images clean
ENV DEBIAN_FRONTEND=