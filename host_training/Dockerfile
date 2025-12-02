FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base_cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y \
      # see: the Hailo DFC user guide
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      python3.10-distutils \
      python3-pip \
      python3-tk \
      graphviz \
      libgraphviz-dev \
      libgl1-mesa-glx \
      # utilities
      python-is-python3 \
      build-essential \
      sudo \
      curl \
      git \
      nano && \
    # clean up
    rm -rf /var/lib/apt/lists/*

# update pip
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /workspace