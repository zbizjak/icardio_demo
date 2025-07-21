FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3.10 \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# Install any python packages you need
RUN pip list

RUN python3 -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install wget
RUN pip install h5py
RUN pip install scikit-image
RUN pip install wandb
RUN pip install nibabel
RUN pip install nilearn
RUN pip install matplotlib
RUN pip install SimpleITK
RUN pip install openpyxl
RUN pip install evalutils
RUN pip install line_profiler
RUN pip install tqdm
RUN pip install opencv-python
RUN pip install torch torchvision torchaudio transformers


# Set the working directory
WORKDIR /icardio