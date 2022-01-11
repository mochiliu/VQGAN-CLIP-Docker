#!/bin/bash
docker run --rm --gpus 'device=1' -p 8888:8888 -v /home/mochi/github/VQGAN-CLIP-Docker/:/workspace/VQGAN-CLIP-Docker/ -it vqgan-clip
