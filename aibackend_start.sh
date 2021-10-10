#!/bin/zsh

eval "$(/home/legekka/anaconda3/bin/conda shell.zsh hook)"
conda activate env_pytorch
cd /opt/bots/ai-backend
python backend.py