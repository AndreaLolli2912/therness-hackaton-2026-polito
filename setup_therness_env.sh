conda create -n torch311 python=3.11 -y
conda activate torch311
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
