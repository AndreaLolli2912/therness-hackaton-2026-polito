#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda create -n therness_env python=3.11 -y
conda activate therness_env
conda install -c conda-forge ffmpeg -y
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install torchcodec
pip install ipykernel
python -m ipykernel install --user --name therness_env
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
