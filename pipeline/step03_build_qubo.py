"""
Step 03 — 組 QUBO 目標函數,轉成 Ising,匯出耦合檔
==================================================================
目標函數 (x_i in {0,1}):

    E(x) = -alpha * sum_i R_i x_i
           + beta  * sum_{i<j} C_ij x_i x_j
           + lam   * (sum_i x_i - k)^2

展開基數約束 (x_i^2 = x_i):
    (sum x - k)^2 = sum_i (1-2k) x_i + 2 sum_{i<j} x_i x_j + k^2

得 QUBO(上三角)係數:
    Q_ii      = -alpha*R_i + lam*(1 - 2k)
    Q_ij(i<j) =  beta*C_ij + lam*2
    const     =  lam*k^2

QUBO -> Ising,令 x_i = (1+s_i)/2,能量寫成
    E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset   (dimod 慣例,求最小)
    J_ij = Q_ij / 4
    h_i  = Q_ii/2 + (1/4) * sum_{j!=i} Q_ij
    offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4 + const

⚠ 求解器慣例:本檔輸出採 dimod/最小化慣例 E = Σ h s + Σ J s s。
   若你的 CUDA-Ising 用物理慣例 H = -Σ J s s - Σ h s 並找基態,
   請把 J、h 各自「取負號」再餵入(見 README「慣例對齊」)。
"""
from __future__ import annotations
import sys
import numpy as np

from common import load_config, workdir, load_npz, save_npz, save_json


def build_qubo(R, C, alpha, beta, lam, k, normalize):
    R = R.astype(float).copy()
    C = C.astype(float).copy()
    if normalize:
        if R.max() > 0:
            R = R / R.max()
        if C.max() > 0:
            C = C / C.max()

    Qdiag = -alpha * R + lam * (1.0 - 2.0 * k)          # (n,)
    # 冗餘懲罰(beta*C)+ 基數約束成對項(2*lam,加到所有基因對)
    Qoff = beta * C.copy() + 2.0 * lam
    np.fill_diagonal(Qoff, 0.0)
    # 只保留上三角,避免重複計數
    Qoff = np.triu(Qoff, k=1)
    const = lam * k * k
    return Qdiag, Qoff, const


def qubo_to_ising(Qdiag, Qoff):
    # Qoff 為上三角;先做成對稱矩陣好算 row-sum
    Qsym = Qoff + Qoff.T
    J = Qoff / 4.0                                       # 上三角
    h = Qdiag / 2.0 + Qsym.sum(axis=1) / 4.0
    offset = Qdiag.sum() / 2.0 + Qoff.sum() / 4.0
    return h, J, offset


# ----------------------- 匯出器 -----------------------
def export_coo_txt(path, h, J, convention):
    """純文字 COO。第一行 'N M'(N=spin 數, M=非零耦合數)。
    其後每行 'i j value':i==j 代表場 h_i,i<j 代表耦合 J_ij。"""
    iu = np.argwhere(np.triu(J, 1) != 0)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# convention={convention} (E = sum h s + sum J s s, minimize)\n")
        fh.write(f"{len(h)} {len(iu)}\n")
        for i in range(len(h)):
            if h[i] != 0:
                fh.write(f"{i} {i} {h[i]:.8g}\n")
        for i, j in iu:
            fh.write(f"{i} {j} {J[i, j]:.8g}\n")


def export_npz(path, h, J, genes, offset):
    save_npz(path, h=h, J=J, genes=genes, offset=np.array([offset]))


def export_dimod_json(path, h, J):
    linear = {int(i): float(h[i]) for i in range(len(h)) if h[i] != 0}
    quad = {}
    iu = np.argwhere(np.triu(J, 1) != 0)
    for i, j in iu:
        quad[f"{int(i)},{int(j)}"] = float(J[i, j])
    save_json(path, {"vartype": "SPIN", "linear": linear, "quadratic": quad})


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    q = cfg["qubo"]
    d = load_npz(f"{workdir(cfg)}/features.npz")
    R, C, genes = d["R"], d["C"], d["genes"]

    Qdiag, Qoff, const = build_qubo(
        R, C, q["alpha"], q["beta"], q["lam"], q["k"], q["normalize"])
    h, J, offset = qubo_to_ising(Qdiag, Qoff)
    offset += const

    wd = workdir(cfg)
    fmts = cfg["export"]["formats"]
    conv = cfg["export"]["spin_convention"]
    if "coo_txt" in fmts:
        export_coo_txt(f"{wd}/ising_coupling.txt", h, J, conv)
    if "npz" in fmts:
        export_npz(f"{wd}/ising.npz", h, J, genes, offset)
    if "dimod_json" in fmts:
        export_dimod_json(f"{wd}/ising_bqm.json", h, J)

    # 也存一份基因對照表(spin index -> gene),後處理要用
    save_json(f"{wd}/gene_index.json",
              {int(i): str(g) for i, g in enumerate(genes)})

    print(f"spins={len(h)}, 非零耦合={int((np.triu(J,1)!=0).sum())}, offset={offset:.4g}")
    print(f"已輸出格式 {fmts} -> {wd}/")


if __name__ == "__main__":
    main()
