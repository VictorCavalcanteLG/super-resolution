# Use uma imagem base com suporte a GPU
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        curl \
        libglib2.0-0

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to the PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create a new conda environment with Python 3.11 and install dependencies
COPY environment.yaml .
RUN /opt/conda/bin/conda env create -f environment.yaml && \
    /opt/conda/bin/conda clean --all --yes

# Ative o ambiente conda
SHELL ["/bin/bash", "--login", "-c"]
RUN source activate super-resolution

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

WORKDIR super-resolution/

CMD ["conda", "run", "--no-capture-output", "-n", "super-resolution", "python", "-m", "src.scripts.train_model"]
