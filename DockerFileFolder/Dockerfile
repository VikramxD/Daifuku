# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        git \
        curl \
        build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python${PYTHON_VERSION} get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && python3 --version \
    && pip3 --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy everything from DockerFileFolder and the rest of the directory structure
COPY DockerFileFolder/ /app/DockerFileFolder/
COPY src/ /app/src/
COPY requirements.txt /app/

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create log directory if it doesn't exist
RUN mkdir -p /app/src/api/logs

# Expose port
EXPOSE 8000

# Set working directory for the server
WORKDIR /app/src/api

# Command to run the server
CMD ["python3", "mochi_serve.py"]