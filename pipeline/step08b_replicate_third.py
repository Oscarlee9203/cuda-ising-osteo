"""Step 08b — third-cohort replication on GSE7158 (monocytes, GPL570).
GSE7158 groups are in the sample TITLE ('Monocyte High/Low PBM'), not a
'bone mineral density' field, so it needs its own parser. case(y=1)=Low bone
mass, control(y=0)=High. This is an INDEPENDENT monocyte cohort (premenopausal
peak-bone-mass), so it is a valid external test of the meta candidates."""
from __future__ import annotations
import sys, csv, json
import numpy as np
from scipy import stats
from common import load_config, workdir, load_npz
import step01_download_geo as s1
import step08_replicate as s8


def load_gse7158(acc, platform, wd):
    import GEOparse
    gse = GEOparse.get_GEO(geo=acc, destdir=wd, silent=True)
    expr = gse.pivot_samples("VALUE")
    labels = []
    for g in list(expr.columns):
        txt = " ".join(gse.gsms[g].metadata.get("title", [])).lower()
        txt += " " + " ".join(gse.gsms[g].metadata.get("characteristics_ch1", [])).lower()
        labels.append(1 if "low" in txt else (0 if "high" in txt else -1))
    y = np.array(labels)
    X = expr.T.to_numpy(dtype=float)
    probes = np.array(expr.index.astype(str))
    keep = y >= 0
    X, y = X[keep], y[keep]
    good = ~np.all(np.isnan(X), axis=0)
    X, probes = X[:, good], probes[good]
    X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
    if np.nanmax(X) > 100:
        X = np.log2(X + 1.0)
    X, genes = s1._collapse_symbols(gse, X, probes, platform)
    print(f"[GSE7158] {acc}/{platform} X={X.shape} "
          f"low={int(y.sum())} high={int((y==0).sum())}")
    return X, y, np.array([str(g) for g in genes])


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    wd = workdir(cfg)
    k = cfg["qubo"]["k"]
    r3 = cfg.get("replication3", {"geo_accession": "GSE7158", "platform": "GPL570"})
    disc = load_npz(f"{wd}/expression.npz")
    Xd, yd = disc["X"], disc["y"]
    gd = np.array([str(g) for g in disc["genes"]])
    di = {g: i for i, g in enumerate(gd)}
    with open(f"{wd}/candidates.csv", newline="", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r["select_freq"]), reverse=True)
    panel = [r["gene"] for r in rows[:k]]
    Xe, ye, ge = load_gse7158(r3["geo_accession"], r3["platform"], wd)
    ei = {g: i for i, g in enumerate(ge)}
    matched = [g for g in panel if g in di and g in ei]
    print(f"panel {len(panel)}, matched in GSE7158 {len(matched)}")
    dcol = [di[g] for g in matched]
    ecol = [ei[g] for g in matched]
    auc_d2e = s8.transfer_auc(Xd[:, dcol], yd, Xe[:, ecol], ye)
    auc_e2d = s8.transfer_auc(Xe[:, ecol], ye, Xd[:, dcol], yd)
    ext_m, ext_s = s8.cv_auc(Xe[:, ecol], ye)
    effd = s8.signed_effect(Xd[:, dcol], yd)
    effe = s8.signed_effect(Xe[:, ecol], ye)
    conc = float(np.mean(np.sign(effd) == np.sign(effe)))
    rho, p = (stats.spearmanr(effd, effe) if len(matched) >= 3 else (float("nan"), float("nan")))
    print("\n===== 第三世代複製 (GSE7158, 停經前單核球, GPL570) =====")
    print(f"discovery GSE56815 (n={len(yd)}) vs external GSE7158 (n={len(ye)})")
    print(f"(A) transfer AUC  disc->ext={auc_d2e:.3f}  ext->disc={auc_e2d:.3f}")
    print(f"(C) external CV AUC = {ext_m:.3f} ± {ext_s:.3f}")
    print(f"(B) direction concordance = {conc:.1%}  |  Spearman rho={rho:.3f} (p={p:.3g})")
    with open(f"{wd}/replication3_panel.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["gene", "disc_effect", "ext_effect", "same_direction"])
        for g, a, b in zip(matched, effd, effe):
            w.writerow([g, f"{a:.4f}", f"{b:.4f}", bool(np.sign(a) == np.sign(b))])
    print(f"-> {wd}/replication3_panel.csv")


if __name__ == "__main__":
    main()
