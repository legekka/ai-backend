#!/bin/zsh

eval "$(/home/maiia@boltz.local/miniconda3/bin/conda shell.zsh hook)"
conda activate aibackend
cd /opt/bots/ai-backend
python backend.py
