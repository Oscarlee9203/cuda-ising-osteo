"""
Step 07 — 經典特徵選擇基線對照
==================================================================
目的:證明 Ising/QUBO 選出的基因組「多找到了什麼」。
這是最能說服審稿人與合作者的一步。

在同一份 X, y 上,用同樣的基因數 k,跑幾種經典特徵選擇:
    - LASSO      : L1-logistic,取係數最大的 k 個
    - mRMR       : 最大相關-最小冗餘,貪婪選 k 個(自實作,免額外套件)
    - RandomForest: 依 impurity importance 取前 k 個
    - Univariate : 直接用 step02 的相關性 R 取前 k 個
    - Random     : 隨機 k 個(下限對照,informed 方法都該贏它)
再和 Ising 的選擇(candidates.csv 依 select_freq 取前 k)比:

  (A) 重疊度:與 Ising 的 Jaccard、共享基因數 -> 看方法間是否一致
  (B) 預測力:對「每個方法選出的基因組」做交叉驗證分類 AUC
              -> 同樣 k 個基因,誰組成的 panel 更能區分 high/low BMD
  (C) Ising 獨有:只有 Ising 選到、其他方法沒選到的基因 -> 潛在新發現

輸出:
    outputs/baseline_comparison.csv  每個方法一列:n, cv_auc, jaccard_vs_ising ...
    outputs/method_gene_sets.json    每個方法選到的基因 + Ising 獨有基因

⚠ 說明:各方法(含 Ising)都在全資料上做選擇,再用交叉驗證評估「該基因組的預測力」。
   這回答的是「給定這組基因,panel 有多強」。若要無偏估計選擇本身的泛化力,
   應把「選擇」也包進 CV 折內(nested CV)—— 見 README 註記。
"""
from __future__ import annotations
import sys
import csv
import json

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import load_config, workdir, load_npz


# ---------------- 各種選擇器,回傳「選到的欄位索引」----------------
def sel_lasso(X, y, k, seed):
    import warnings
    Xs = StandardScaler().fit_transform(X)
    # L1-logistic 取係數最大的 k 個。penalty='l1'+liblinear 在各版本皆可用
    # (sklearn 1.8 起對 penalty 有 FutureWarning,這裡靜音,功能不受影響)。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                 max_iter=2000, random_state=seed).fit(Xs, y)
    coef = np.abs(clf.coef_.ravel())
    return np.argsort(coef)[::-1][:k]


def sel_rf(X, y, k, seed):
    rf = RandomForestClassifier(n_estimators=400, random_state=seed,
                                n_jobs=-1).fit(X, y)
    return np.argsort(rf.feature_importances_)[::-1][:k]


def sel_univariate(X, y, k, R=None):
    if R is None:                     # 沒帶 R 就用 F 檢定當相關性
        F, _ = stats.f_oneway(*[X[y == c] for c in np.unique(y)])
        R = np.nan_to_num(F)
    return np.argsort(R)[::-1][:k]


def sel_mrmr(X, y, k):
    """最小冗餘-最大相關(MID 版):relevance 用 F 檢定,redundancy 用 |Pearson|。"""
    F, _ = stats.f_oneway(*[X[y == c] for c in np.unique(y)])
    rel = np.nan_to_num(F)
    n = X.shape[1]
    C = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    selected = [int(np.argmax(rel))]
    cand = set(range(n)) - set(selected)
    while len(selected) < min(k, n):
        best, best_score = None, -np.inf
        for j in cand:
            red = np.mean([C[j, s] for s in selected])
            score = rel[j] - red        # MID: 相關 - 平均冗餘
            if score > best_score:
                best, best_score = j, score
        selected.append(best)
        cand.discard(best)
    return np.array(selected)


def sel_random(X, y, k, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(X.shape[1], size=k, replace=False)


# ---------------- 評估 ----------------
def cv_auc(X, y, cols, seed, n_splits=5):
    if len(cols) == 0:
        return float("nan"), float("nan")
    Xc = X[:, cols]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    s = cross_val_score(clf, Xc, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


# ---------------- 主流程 ----------------
def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    wd = workdir(cfg)
    seed = cfg["data"].get("demo_seed", 0)
    k = cfg["qubo"]["k"]

    expr = load_npz(f"{wd}/expression.npz")
    X, y, genes = expr["X"], expr["y"], np.array([str(g) for g in expr["genes"]])
    gene2idx = {g: i for i, g in enumerate(genes)}

    # 若 step02 有做預篩(features.npz 的基因是子集),用它對齊 X 的欄位
    try:
        feat = load_npz(f"{wd}/features.npz")
        fgenes = np.array([str(g) for g in feat["genes"]])
        if len(fgenes) != len(genes):
            cols = [gene2idx[g] for g in fgenes if g in gene2idx]
            X, genes = X[:, cols], fgenes
            gene2idx = {g: i for i, g in enumerate(genes)}
        R = feat["R"]
    except Exception:
        R = None

    # ---- Ising 選擇:candidates.csv 依 select_freq 取前 k ----
    ising_genes = []
    with open(f"{wd}/candidates.csv", newline="", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh),
                      key=lambda r: float(r["select_freq"]), reverse=True)
        ising_genes = [r["gene"] for r in rows[:k]]
    ising_cols = np.array([gene2idx[g] for g in ising_genes if g in gene2idx])

    # ---- 跑各方法 ----
    methods = {
        "Ising_QUBO":  ising_cols,
        "LASSO":       sel_lasso(X, y, k, seed),
        "mRMR":        sel_mrmr(X, y, k),
        "RandomForest": sel_rf(X, y, k, seed),
        "Univariate":  sel_univariate(X, y, k, R),
        "Random":      sel_random(X, y, k, seed),
    }

    results, gene_sets = [], {}
    for name, cols in methods.items():
        cols = np.asarray(cols, dtype=int)
        auc_m, auc_s = cv_auc(X, y, cols, seed)
        jac = jaccard(cols, ising_cols)
        shared = len(set(cols) & set(ising_cols))
        results.append({"method": name, "n": len(cols),
                        "cv_auc_mean": auc_m, "cv_auc_std": auc_s,
                        "jaccard_vs_ising": jac, "n_shared_with_ising": shared})
        gene_sets[name] = [genes[c] for c in cols]

    # ---- Ising 獨有(其他方法都沒選到)----
    others = set()
    for name, cols in methods.items():
        if name != "Ising_QUBO":
            others |= {genes[c] for c in cols}
    ising_unique = [g for g in ising_genes if g not in others]
    gene_sets["_Ising_unique"] = ising_unique

    # ---- 輸出 ----
    outc = f"{wd}/baseline_comparison.csv"
    with open(outc, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({**r,
                        "cv_auc_mean": f"{r['cv_auc_mean']:.4f}",
                        "cv_auc_std": f"{r['cv_auc_std']:.4f}",
                        "jaccard_vs_ising": f"{r['jaccard_vs_ising']:.4f}"})
    with open(f"{wd}/method_gene_sets.json", "w", encoding="utf-8") as fh:
        json.dump(gene_sets, fh, ensure_ascii=False, indent=2)

    # ---- 摘要 ----
    print(f"每個方法各選 k={k} 個基因,5-fold CV AUC:\n")
    print(f"  {'method':13s} {'CV_AUC':>16s}  {'Jaccard_vs_Ising':>16s}  shared")
    for r in sorted(results, key=lambda x: x["cv_auc_mean"], reverse=True):
        print(f"  {r['method']:13s} {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f}   "
              f"{r['jaccard_vs_ising']:>14.3f}   {r['n_shared_with_ising']}")
    print(f"\nIsing 獨有基因({len(ising_unique)} 個,其他方法都沒選到):")
    print("  " + (", ".join(ising_unique) if ising_unique else "(無)"))
    print(f"\n-> {outc}\n-> {wd}/method_gene_sets.json")


if __name__ == "__main__":
    main()
