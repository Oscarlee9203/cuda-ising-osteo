"""Step 02b — replication-aware meta feature selection (replaces step02)."""
from __future__ import annotations
import sys
import numpy as np
from scipy import stats
from common import load_config, workdir, load_npz, save_npz
import step01_download_geo as s1
import step02_features as s2


def signed_auc_effect(X, y):
    case, ctrl = X[y == 1], X[y == 0]
    n1, n0 = len(case), len(ctrl)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        u, _ = stats.mannwhitneyu(case[:, j], ctrl[:, j], alternative="two-sided")
        out[j] = 2.0 * (u / (n1 * n0) - 0.5)
    return out


def _load_cohort(acc, platform, wd, restrict_state):
    X, y, genes = s1.load_series(acc, platform, wd, restrict_state,
                                 collapse=True, tag=acc)
    return X, y, np.array([str(g) for g in genes])


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    wd = workdir(cfg)
    f = cfg["features"]
    combine = cfg.get("meta", {}).get("combine", "min")
    dA, rB = cfg["data"], cfg["replication"]
    print("load discovery cohort...")
    XA, yA, gA = _load_cohort(dA["geo_accession"], dA.get("platform", "GPL96"),
                              wd, dA.get("restrict_state"))
    print("load external cohort...")
    XB, yB, gB = _load_cohort(rB["geo_accession"], rB["platform"],
                              wd, rB.get("restrict_state"))
    posA = {g: i for i, g in enumerate(gA)}
    posB = {g: i for i, g in enumerate(gB)}
    common = np.array([g for g in gA if g in posB])
    print(f"common genes: {len(common)}")
    iA = [posA[g] for g in common]
    iB = [posB[g] for g in common]
    effA = signed_auc_effect(XA[:, iA], yA)
    effB = signed_auc_effect(XB[:, iB], yB)
    relA, relB = np.abs(effA), np.abs(effB)
    concordant = np.sign(effA) == np.sign(effB)
    print(f"concordant-direction genes: {int(concordant.sum())} / {len(common)} "
          f"({concordant.mean():.1%})")
    if combine == "stouffer":
        zA = stats.norm.ppf((stats.rankdata(relA) - 0.5) / len(relA))
        zB = stats.norm.ppf((stats.rankdata(relB) - 0.5) / len(relB))
        Rmeta = (zA + zB) / np.sqrt(2)
        Rmeta = Rmeta - Rmeta.min()
    else:
        Rmeta = np.minimum(relA, relB)
    Rmeta = np.where(concordant, Rmeta, 0.0)
    keep = Rmeta > 0
    genes_c, R_c = common[keep], Rmeta[keep]
    print(f"replicable pool (concordant & R>0): {len(genes_c)}")
    top = f.get("top_genes", 0)
    order = np.argsort(R_c)[::-1]
    if top and top < len(order):
        order = order[:top]
    sel_genes, sel_R = genes_c[order], R_c[order]
    Xsel = XA[:, [posA[g] for g in sel_genes]]
    C = s2.redundancy(Xsel, f["redundancy_metric"], f["redundancy_threshold"])
    save_npz(f"{wd}/features.npz", R=sel_R, C=C, genes=sel_genes)
    nnz = int((C > 0).sum() // 2)
    print(f"saved features.npz: {len(sel_genes)} genes, C edges={nnz}, "
          f"R=[{sel_R.min():.3f},{sel_R.max():.3f}]")
    print("Top 10 meta candidates (by R):")
    for g, r in list(zip(sel_genes, sel_R))[:10]:
        print(f"  {g:14s} R_meta={r:.3f}")
    print("\n[warn] GSE56814 was used for selection, so step08 on GSE56814 is "
          "no longer an independent test. Use a THIRD cohort for real validation.")


if __name__ == "__main__":
    main()
