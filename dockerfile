FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
LABEL maintainer "Yang <leibniz21c@gmail.com>"

# set environment variables
ENV WORKSPACE="/workspace" \
    DATASET_PATH="/datasets"

# set working directory
WORKDIR /workspace

# copy dependencies
COPY requirements.txt .

# Dependencies
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    mkdir /datasets 

# User and group setting
#USER appuser

ENV APP_VERSION="1.0.0" \
    APP_NAME="cifar10-classifier"