FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo ca-certificates && \
    apt-get clean

COPY ./requirements.txt /requirements.txt

RUN python -m pip install -r /requirements.txt

