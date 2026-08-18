"""
Step 05 — 求解(或讀取 CUDA-Ising 結果)+ 彙整候選基因
==================================================================
低能量態不會只有一個。做法是收集「一整群」低能量解,
統計每個基因在這些解裡被選中的頻率 —— 頻率越高越穩健。

兩種來源:
  A) --states <file>  讀外部求解器(你的 CUDA-Ising)輸出的狀態檔。
       格式:每行一個解,空白分隔;值為 +1/-1(ising)或 1/0(qubo)。
  B) 不給 --states     用內建的 numpy 模擬退火跑一個 ensemble
       (免額外套件,可當基線 baseline,也方便離線驗證)。

輸出:outputs/candidates.csv  欄位: gene, select_freq, R
"""
from __future__ import annotations
import sys
import argparse
import numpy as np

from common import load_config, workdir, load_npz


def ising_energy(s, h, J):
    return h @ s + s @ np.triu(J, 1) @ s


def sa_sample(h, J, n_reads, n_sweeps, seed=0):
    """極簡單自旋翻轉模擬退火。回傳 (n_reads, n) 的 ±1 解集合。
    正式研究請用 CUDA-Ising 的 parallel tempering,這裡只作基線/驗證。"""
    rng = np.random.default_rng(seed)
    n = len(h)
    Jsym = np.triu(J, 1)
    Jsym = Jsym + Jsym.T
    states = np.empty((n_reads, n), dtype=int)
    temps = np.geomspace(5.0, 0.05, n_sweeps)
    for r in range(n_reads):
        s = rng.choice([-1, 1], size=n)
        for T in temps:
            for i in range(n):
                # 翻轉 s_i 的能量變化: dE = -2 s_i (h_i + sum_j J_ij s_j)
                local = h[i] + Jsym[i] @ s
                dE = -2.0 * s[i] * local
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    s[i] = -s[i]
        states[r] = s
    return states


def load_states(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()])
    arr = np.array(rows)
    # 轉成 ±1(若是 0/1 就映射)
    if arr.min() >= 0:
        arr = arr * 2 - 1
    return arr.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--states", default=None, help="外部求解器狀態檔")
    ap.add_argument("--reads", type=int, default=200, help="內建 SA 的解數")
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--top-frac", type=float, default=0.2,
                    help="只取能量最低的前這個比例的解來統計頻率")
    args = ap.parse_args()

    cfg = load_config(args.config)
    wd = workdir(cfg)
    ising = load_npz(f"{wd}/ising.npz")
    feat = load_npz(f"{wd}/features.npz")
    h, J, genes = ising["h"], ising["J"], ising["genes"]
    R = feat["R"]

    if args.states:
        states = load_states(args.states)
        print(f"讀入外部解 {states.shape}")
    else:
        states = sa_sample(h, J, args.reads, args.sweeps)
        print(f"內建 SA 產生解 {states.shape}")

    # 依能量排序,取最低的一批
    E = np.array([ising_energy(s, h, J) for s in states])
    order = np.argsort(E)
    keep = order[: max(1, int(len(order) * args.top_frac))]
    low = states[keep]
    print(f"最低能量={E.min():.4g},取前 {len(keep)} 個解統計頻率")

    # spin=+1 視為「選中」該基因
    selected = (low == 1)
    freq = selected.mean(axis=0)

    idx = np.argsort(freq)[::-1]
    out = f"{wd}/candidates.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("gene,select_freq,R\n")
        for i in idx:
            if freq[i] > 0:
                fh.write(f"{genes[i]},{freq[i]:.4f},{R[i]:.4f}\n")
    print(f"候選基因 -> {out}")
    print("Top 15:")
    for i in idx[:15]:
        print(f"  {str(genes[i]):12s} freq={freq[i]:.3f}  R={R[i]:.3f}")


if __name__ == "__main__":
    main()
