ARG CUDA_VER=12.6.0
ARG UBUNTU_VER=20.04
ARG PY_VER=3.10

FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-ubuntu${UBUNTU_VER}
# FROM ubuntu:${UBUNTU_VER}

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-dev \
    git \
    wget \
    sudo \
    ninja-build \
    gcc \
    cmake \
    curl \
    nano \
    cron \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    openjdk-11-jdk \
    zip

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# install miniconda
ENV PATH="/root/miniconda3/bin:$PATH"
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

# create conda environment
RUN conda init bash \
    && . ~/.bashrc \
    && conda create -y --name dbenv python=${PY_VER} \
    && conda activate dbenv \
    && pip install ipython

# install packages from conda 
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# downlaod and install nltk data
RUN pip install nltk==3.8.1
RUN python -m nltk.downloader punkt wordnet omw-1.4 stopwords

# clean up the apt cache to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# edit bashrc
RUN echo "PS1='\[\033[1;36m\]\u\[\033[1;34m\]@\[\033[1;36m\]\h:\[\033[1;32m\][\w]\[\033[1;31m\]\$\[\033[0m\] '" >> ~/.bashrc
RUN echo 'export PATH="$PATH:/root/.local/bin"' >> ~/.bashrc
RUN echo 'export TRANSFORMERS_CACHE=/transformers' >> ~/.bashrc
RUN echo 'conda activate base' >> ~/.bashrc
