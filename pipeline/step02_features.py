"""
Step 02 — 計算相關性 R_i 與冗餘度 C_ij
==================================================================
輸入:outputs/expression.npz
輸出:outputs/features.npz  內含
    R      : (m,) 每個(預篩後)基因的相關性分數,已可選標準化
    C      : (m, m) 冗餘度矩陣(上三角有效,對角 0)
    genes  : (m,) 對應基因名

R_i 越大 = 越能區分病例/對照;C_ij 越大 = 兩基因越冗餘。
這兩者之後直接餵進 QUBO。
"""
from __future__ import annotations
import sys
import numpy as np
from scipy import stats

from common import load_config, workdir, load_npz, save_npz


def relevance(X, y, metric: str) -> np.ndarray:
    case, ctrl = X[y == 1], X[y == 0]
    if metric == "ttest":
        t, p = stats.ttest_ind(case, ctrl, axis=0, equal_var=False)
        p = np.nan_to_num(p, nan=1.0)
        return -np.log10(p + 1e-300)
    if metric == "corr":
        r = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
        return np.abs(np.nan_to_num(r))
    if metric == "auc":
        # 以 Mann-Whitney U 換算 AUC,再取 |AUC-0.5|*2(0~1,越大越有辨別力)
        out = np.empty(X.shape[1])
        n1, n0 = len(case), len(ctrl)
        for j in range(X.shape[1]):
            u, _ = stats.mannwhitneyu(case[:, j], ctrl[:, j], alternative="two-sided")
            auc = u / (n1 * n0)
            out[j] = abs(auc - 0.5) * 2.0
        return out
    raise ValueError(f"未知的 relevance_metric: {metric}")


def redundancy(X, metric: str, thr: float) -> np.ndarray:
    if metric == "pearson":
        C = np.corrcoef(X, rowvar=False)
    elif metric == "spearman":
        C, _ = stats.spearmanr(X)
        C = np.atleast_2d(C)
    else:
        raise ValueError(f"未知的 redundancy_metric: {metric}")
    C = np.abs(np.nan_to_num(C))
    np.fill_diagonal(C, 0.0)
    if thr > 0:
        C[C < thr] = 0.0
    return C


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    f = cfg["features"]
    d = load_npz(f"{workdir(cfg)}/expression.npz")
    X, y, genes = d["X"], d["y"], d["genes"]

    R = relevance(X, y, f["relevance_metric"])

    # 依相關性預篩,控制 QUBO 規模
    top = f.get("top_genes", 0)
    if top and top < len(R):
        idx = np.argsort(R)[::-1][:top]
        idx = np.sort(idx)
        X, genes, R = X[:, idx], genes[idx], R[idx]
        print(f"預篩:保留相關性前 {top} 個基因")

    C = redundancy(X, f["redundancy_metric"], f["redundancy_threshold"])

    nnz = int((C > 0).sum() // 2)
    print(f"R shape={R.shape}, C 非零邊={nnz}, "
          f"R範圍=[{R.min():.3f},{R.max():.3f}]")

    save_npz(f"{workdir(cfg)}/features.npz", R=R, C=C, genes=genes)
    print(f"已存檔 -> {workdir(cfg)}/features.npz")


if __name__ == "__main__":
    main()
