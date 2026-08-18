#!/usr/bin/env bash
# ==========================================================================
# 冒煙測試 — 用合成資料(loader=demo)離線跑完整條 pipeline,
# 含 GPU 求解器 solve_gpu.py(無 CUDA 時自動退回 NumPy)。
# 不需要網路、不需要 GPU。跑得過代表安裝正確。
# ==========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/pipeline"

CFG=/tmp/cuda_ising_demo.yaml
sed 's/loader: "gse56815"/loader: "demo"/' ../config.yaml > "$CFG"

echo "== step01 載入(demo 合成資料) =="; python step01_download_geo.py "$CFG"
echo "== step02 特徵 =="; python step02_features.py "$CFG"
echo "== step03 建 QUBO/Ising =="; python step03_build_qubo.py "$CFG"
echo "== solve_gpu 求解(device=cpu 走 NumPy fallback) =="
python solve_gpu.py --ising outputs/ising.npz --out outputs/states.txt \
    --n-temp 10 --chains 40 --sweeps 200 --device cpu --seed 1
echo "== step05 彙整候選 =="
python step05_postprocess.py "$CFG" --states outputs/states.txt --top-frac 0.15
echo "== step07 經典基線對照 =="; python step07_baselines.py "$CFG" | tail -9

echo ""
if [[ -s outputs/candidates.csv ]]; then
  echo "✅ 冒煙測試通過:outputs/candidates.csv 已產生。安裝無誤。"
else
  echo "❌ 冒煙測試失敗:未產生 candidates.csv"; exit 1
fi
