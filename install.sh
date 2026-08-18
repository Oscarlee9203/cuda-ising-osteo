#!/usr/bin/env bash
# ==========================================================================
# CUDA-Ising 骨質疏鬆 pipeline — DGX-B200 安裝腳本
# --------------------------------------------------------------------------
# 用法:
#   bash install.sh                # 只裝 CPU 相依(numpy/scipy/sklearn/GEOparse)
#   bash install.sh --with-torch   # 額外裝 Blackwell(B200)用的 PyTorch cu128
#
# 說明:B200 是 Blackwell(sm_100),需要 CUDA 12.8 + PyTorch cu128 wheel。
#       若你在 DGX 上用 NGC PyTorch 容器,torch 已內建,毋須 --with-torch。
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
WITH_TORCH=0
[[ "${1:-}" == "--with-torch" ]] && WITH_TORCH=1

# 偵測系統 Python 是否已有 torch(例如 NGC PyTorch 容器)。
# 若有,venv 要用 --system-site-packages 繼承它,否則 venv 裡看不到 torch。
HAS_TORCH=0
if "$PYTHON" -c "import torch" 2>/dev/null; then
  HAS_TORCH=1
  echo ">>> 偵測到系統已有 PyTorch(容器內建),venv 會繼承它,不重裝 torch。"
fi

echo ">>> 建立虛擬環境 $VENV"
if [[ "$HAS_TORCH" == "1" ]]; then
  "$PYTHON" -m venv --system-site-packages "$VENV"   # 繼承容器的 torch/CUDA
else
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

echo ">>> 安裝核心相依"
pip install "numpy>=1.24" "scipy>=1.10" "pyyaml>=6.0" "scikit-learn>=1.2" "GEOparse>=2.0"

echo ">>> (選用)線上通路富集用 gseapy"
pip install "gseapy>=1.1" || echo "    gseapy 安裝略過(可日後再裝;離線用本地 GMT 即可)"

if [[ "$WITH_TORCH" == "1" && "$HAS_TORCH" == "0" ]]; then
  echo ">>> 安裝 PyTorch(Blackwell / B200 = CUDA 12.8 / cu128)"
  pip install torch --index-url https://download.pytorch.org/whl/cu128
elif [[ "$WITH_TORCH" == "1" && "$HAS_TORCH" == "1" ]]; then
  echo ">>> 已有容器內建 torch,忽略 --with-torch(不重裝)。"
fi

echo ""
echo ">>> 安裝完成。啟用環境:  source $VENV/bin/activate"
echo ">>> 冒煙測試(離線、免 GPU):  bash smoke_test.sh"
echo ">>> GPU / torch 檢查:"
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.version.cuda, \
'| available:', torch.cuda.is_available())" \
  || echo "  (venv 內沒有 torch — GPU 求解需要;容器環境請確認系統 torch 可 import)"
