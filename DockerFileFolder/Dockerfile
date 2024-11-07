# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python$PYTHON_VERSION get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python \
    && ln -sf /usr/local/bin/pip$PYTHON_VERSION /usr/local/bin/pip \
    && python --version \
    && pip --version \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*




COPY . .

RUN pip install uv
RUN uv pip install --no-cache-dir -r requirements.txt
RUN uv pip install -e . --no-build-isolation

EXPOSE 8000

CMD ["python", "api/mochi_serve.py"]
