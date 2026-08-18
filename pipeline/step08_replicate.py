"""
Step 08 — 外部世代複製(External cohort replication)
==================================================================
把 discovery(GSE56815 / GPL96)選出的候選基因組,拿到一份「獨立、
不同平台」的世代(GSE56814 / GPL5175 Exon 陣列)上檢驗是否複製得出來。
這是把結果從「有趣」推到「可信」最有效的一步。

三個檢驗:
  (A) 跨世代預測轉移:只用候選基因,在 discovery 上訓練 logistic,
      到 external 上測 AUC(再做反向 external->discovery)。
      panel 若能跨平台/跨世代轉移,代表訊號穩健、非過度擬合。
  (B) 每個基因的效應方向一致性:在兩個世代各算 low vs high BMD 的效應
      (以 |AUC-0.5| 帶方向),看方向是否相同、效應量是否相關(Spearman)。
  (C) 候選在 external 世代自身的 5-fold CV AUC(panel 在新資料裡還準不準)。

輸入:
  outputs/candidates.csv      (discovery 的 Ising 候選)
  outputs/expression.npz      (discovery X,y,genes;symbol)
輸出:
  outputs/replication_panel.csv   每個候選基因兩世代的效應與方向是否一致
  outputs/replication_summary.txt 三項檢驗摘要

需要網路 + GEOparse(下載 GSE56814)。可先跑 step01~05 產生 discovery 候選。
"""
from __future__ import annotations
import sys
import csv
import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from common import load_config, workdir, load_npz
import step01_download_geo as s1


def signed_effect(X, y):
    """每個基因:low(y=1) vs high(y=0) 的帶方向效應量 = 2*(AUC-0.5)。
    >0 代表在 low BMD 較高。回傳 (n_genes,)。"""
    case, ctrl = X[y == 1], X[y == 0]
    n1, n0 = len(case), len(ctrl)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        u, _ = stats.mannwhitneyu(case[:, j], ctrl[:, j], alternative="two-sided")
        out[j] = 2.0 * (u / (n1 * n0) - 0.5)
    return out


def transfer_auc(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, prob)


def cv_auc(X, y, seed=0, n_splits=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    s = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    wd = workdir(cfg)
    rep = cfg["replication"]
    k = cfg["qubo"]["k"]

    # ---- discovery ----
    disc = load_npz(f"{wd}/expression.npz")
    Xd, yd = disc["X"], disc["y"]
    gd = np.array([str(g) for g in disc["genes"]])
    di = {g: i for i, g in enumerate(gd)}

    # ---- 候選 panel ----
    with open(f"{wd}/candidates.csv", newline="", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh),
                      key=lambda r: float(r["select_freq"]), reverse=True)
    panel = [r["gene"] for r in rows] if rep.get("panel") == "all" \
        else [r["gene"] for r in rows[:k]]

    # ---- external cohort(不同平台)----
    Xe, ye, ge = s1.load_series(
        rep["geo_accession"], rep["platform"], wd,
        rep.get("restrict_state"), collapse=True, tag="replication")
    ei = {g: i for i, g in enumerate(ge)}

    # ---- 對齊:panel 中兩世代都有的基因 ----
    matched = [g for g in panel if g in di and g in ei]
    missing = [g for g in panel if g not in ei]
    print(f"\npanel {len(panel)} 個,external 世代對得上 {len(matched)} 個,"
          f"缺 {len(missing)} 個")

    dcol = [di[g] for g in matched]
    ecol = [ei[g] for g in matched]

    # ---- (A) 跨世代轉移 ----
    auc_d2e = transfer_auc(Xd[:, dcol], yd, Xe[:, ecol], ye)
    auc_e2d = transfer_auc(Xe[:, ecol], ye, Xd[:, dcol], yd)
    # ---- (C) external 自身 CV ----
    ext_cv_m, ext_cv_s = cv_auc(Xe[:, ecol], ye)

    # ---- (B) 每基因方向一致性 ----
    eff_d = signed_effect(Xd[:, dcol], yd)
    eff_e = signed_effect(Xe[:, ecol], ye)
    same_dir = np.sign(eff_d) == np.sign(eff_e)
    concordance = float(np.mean(same_dir)) if len(matched) else float("nan")
    if len(matched) >= 3:
        rho, pval = stats.spearmanr(eff_d, eff_e)
    else:
        rho, pval = float("nan"), float("nan")

    # ---- 輸出:每基因 ----
    pf = f"{wd}/replication_panel.csv"
    with open(pf, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "discovery_effect", "external_effect", "same_direction"])
        for g, a, b, s in zip(matched, eff_d, eff_e, same_dir):
            w.writerow([g, f"{a:.4f}", f"{b:.4f}", bool(s)])

    # ---- 輸出:摘要 ----
    summary = {
        "discovery": {"accession": cfg["data"].get("geo_accession"),
                      "n": int(len(yd)), "platform": cfg["data"].get("platform")},
        "external": {"accession": rep["geo_accession"], "n": int(len(ye)),
                     "platform": rep["platform"]},
        "panel_size": len(panel), "matched_genes": len(matched),
        "transfer_auc_discovery_to_external": round(auc_d2e, 4),
        "transfer_auc_external_to_discovery": round(auc_e2d, 4),
        "external_internal_cv_auc": [round(ext_cv_m, 4), round(ext_cv_s, 4)],
        "direction_concordance": round(concordance, 4),
        "effect_size_spearman_rho": None if np.isnan(rho) else round(rho, 4),
        "effect_size_spearman_p": None if np.isnan(pval) else round(pval, 4),
    }
    sf = f"{wd}/replication_summary.txt"
    with open(sf, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n===== 複製檢驗摘要 =====")
    print(f"discovery {summary['discovery']['accession']} (n={len(yd)}, "
          f"{summary['discovery']['platform']})  vs  "
          f"external {rep['geo_accession']} (n={len(ye)}, {rep['platform']})")
    print(f"(A) 跨世代轉移 AUC  disc->ext={auc_d2e:.3f}  ext->disc={auc_e2d:.3f}")
    print(f"(C) external 自身 5-fold CV AUC = {ext_cv_m:.3f} ± {ext_cv_s:.3f}")
    print(f"(B) 效應方向一致 = {concordance:.1%}  "
          f"| 效應量 Spearman rho={rho:.3f} (p={pval:.3g})")
    print(f"\n-> {pf}\n-> {sf}")


if __name__ == "__main__":
    main()
