"""
Step 04 — 參數掃描
==================================================================
對 config 的 sweep 區塊做 alpha/beta/lam/k 的網格,
每組參數輸出一份 Ising 耦合檔到 outputs/sweep/<標籤>/。
之後你可以把每份餵給 CUDA-Ising,再用 step05 彙整。

用法:
    python step04_param_sweep.py [config.yaml]
"""
from __future__ import annotations
import sys
import os
import itertools
import numpy as np

from common import load_config, workdir, load_npz
import step03_build_qubo as q3


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    wd = workdir(cfg)
    d = load_npz(f"{wd}/features.npz")
    R, C, genes = d["R"], d["C"], d["genes"]
    s = cfg["sweep"]
    norm = cfg["qubo"]["normalize"]
    conv = cfg["export"]["spin_convention"]

    combos = list(itertools.product(s["alpha"], s["beta"], s["lam"], s["k"]))
    print(f"共 {len(combos)} 組參數")
    for alpha, beta, lam, k in combos:
        tag = f"a{alpha}_b{beta}_l{lam}_k{k}"
        out = os.path.join(wd, "sweep", tag)
        os.makedirs(out, exist_ok=True)
        Qdiag, Qoff, const = q3.build_qubo(R, C, alpha, beta, lam, k, norm)
        h, J, offset = q3.qubo_to_ising(Qdiag, Qoff)
        q3.export_coo_txt(f"{out}/ising_coupling.txt", h, J, conv)
        q3.export_npz(f"{out}/ising.npz", h, J, genes, offset + const)
        print(f"  {tag}: spins={len(h)}, edges={int((np.triu(J,1)!=0).sum())}")
    print(f"完成 -> {wd}/sweep/")


if __name__ == "__main__":
    main()
