"""
Step 01 — 取得表現量矩陣 + 表型標籤
==================================================================
輸出:outputs/expression.npz  內含
    X       : (n_samples, n_genes) 表現量矩陣
    y       : (n_samples,) 標籤,1=病例(low BMD),0=對照(high BMD)
    genes   : (n_genes,) 基因名稱(gene symbol,若有做 collapse)

data.loader 決定走哪條路:
  * "demo"     -> 合成資料,完全離線可跑。
  * "gse56815" -> 已對照 GEO 逐筆、寫死欄位解析的真實 series(需網路 + GEOparse)。
  * "generic"  -> 泛用解析(欄位命名不保證正確,通常需自行微調)。
"""
from __future__ import annotations
import sys
import numpy as np

from common import load_config, workdir, save_npz


# ===================== demo(離線合成資料)=====================
def make_demo(cfg):
    d = cfg["data"]
    rng = np.random.default_rng(d["demo_seed"])
    n_g, n_s, n_true = d["demo_n_genes"], d["demo_n_samples"], d["demo_n_true_genes"]
    y = np.array([0] * (n_s // 2) + [1] * (n_s - n_s // 2))
    X = rng.normal(0, 1, size=(n_s, n_g))
    for j in range(n_true):
        X[y == 1, j] += 1.5 * (1.0 - j / n_true)
    for j in range(n_true, n_true + 5):          # 冗餘基因群
        X[:, j] = X[:, 0] + rng.normal(0, 0.2, size=n_s)
    genes = np.array([f"GENE_{i:04d}" for i in range(n_g)])
    for j in range(n_true):
        genes[j] = f"TRUE_{j:02d}"
    return X, y, genes


# ===================== GSE56815(寫死欄位解析)=====================
def _parse_characteristics(gsm):
    """把 characteristics_ch1 的 'key: value' 清單轉成 dict(小寫 key)。"""
    out = {}
    for line in gsm.metadata.get("characteristics_ch1", []):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def load_series(accession, platform, destdir, restrict_state=None,
                collapse=True, tag=""):
    """
    通用「high/low BMD 單核球」GEO series 載入器。
    GSE56815(GPL96)與 GSE56814(GPL5175)欄位結構相同(已逐筆對照 GEO):
      characteristics_ch1:
        "gender: Female"
        "bone mineral density: high BMD" / "low BMD"   <- 分組依據
        "state: postmenopausal" / "premenopausal"
        "cell type: monocytes"
    病例(y=1)= low BMD;對照(y=0)= high BMD。回傳 (X, y, genes)。
    """
    import GEOparse
    gse = GEOparse.get_GEO(geo=accession, destdir=destdir, silent=True)

    expr = gse.pivot_samples("VALUE")            # index=probe, cols=GSM
    labels, states = [], []
    for g in list(expr.columns):
        ch = _parse_characteristics(gse.gsms[g])
        bmd = ch.get("bone mineral density", "").lower()
        states.append(ch.get("state", "").lower())
        labels.append(1 if "low" in bmd else (0 if "high" in bmd else -1))
    y = np.array(labels)
    states = np.array(states)

    X = expr.T.to_numpy(dtype=float)
    probes = np.array(expr.index.astype(str))

    keep = y >= 0
    if restrict_state:
        keep &= (states == restrict_state.lower())
    X, y = X[keep], y[keep]

    good = ~np.all(np.isnan(X), axis=0)
    X, probes = X[:, good], probes[good]
    X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
    if np.nanmax(X) > 100:                        # 線性尺度 -> log2
        X = np.log2(X + 1.0)

    genes = probes
    if collapse:
        X, genes = _collapse_symbols(gse, X, probes, platform)

    print(f"[{tag or accession}] {accession}/{platform} X={X.shape} "
          f"病例(low)={int(y.sum())} 對照(high)={int((y==0).sum())} "
          f"| state={restrict_state}")
    return X, y, genes


def load_gse56815(cfg):
    """GSE56815(GPL96)—— 主 discovery cohort。薄包裝 load_series。"""
    d = cfg["data"]
    return load_series(d["geo_accession"], d.get("platform", "GPL96"),
                       workdir(cfg), d.get("restrict_state"),
                       d.get("collapse_to_symbol", True), tag="gse56815")


def _probe_to_symbol_map(tbl, platform):
    """依平台把 probe->gene symbol。支援 GPL96('Gene Symbol')與
    GPL5175 / Exon 陣列('gene_assignment' 需解析)。"""
    id_col = "ID" if "ID" in tbl.columns else tbl.columns[0]

    # 情況 1:有現成的 'Gene Symbol' 欄(GPL96 等 3' 陣列)
    sym_col = next((c for c in tbl.columns if c.lower() in
                    ("gene symbol", "gene_symbol", "symbol")), None)
    if sym_col is not None:
        out = {}
        for pid, s in zip(tbl[id_col].astype(str), tbl[sym_col].astype(str)):
            if s and s not in ("---", "nan"):
                out[pid] = s.split("///")[0].strip()      # 多對應取第一
        return out

    # 情況 2:Exon 陣列的 'gene_assignment' 欄
    #   格式:"NM_xxx // SYMBOL // desc // loc // id /// NM_yyy // SYMB2 // ..."
    ga_col = next((c for c in tbl.columns if c.lower() == "gene_assignment"), None)
    if ga_col is not None:
        out = {}
        for pid, ga in zip(tbl[id_col].astype(str), tbl[ga_col].astype(str)):
            if not ga or ga in ("---", "nan"):
                continue
            first = ga.split("///")[0].split("//")
            if len(first) >= 2:
                sym = first[1].strip()
                if sym:
                    out[pid] = sym
        return out
    return None


def _collapse_symbols(gse, X, probes, platform):
    """用平台註解把 probe 併成 gene symbol,同 symbol 多探針取平均。"""
    if platform not in gse.gpls:
        print(f"  [警告] 找不到平台 {platform} 註解,保留 probe 名。")
        return X, probes
    p2s = _probe_to_symbol_map(gse.gpls[platform].table, platform)
    if not p2s:
        print(f"  [警告] {platform} 註解無法解析 symbol,保留 probe 名。")
        return X, probes

    from collections import defaultdict
    groups = defaultdict(list)
    for j, p in enumerate(probes):
        sym = p2s.get(p)
        if sym:
            groups[sym].append(j)
    syms = sorted(groups)
    Xc = np.empty((X.shape[0], len(syms)))
    for c, sym in enumerate(syms):
        Xc[:, c] = X[:, groups[sym]].mean(axis=1)
    print(f"  probe {X.shape[1]} -> gene symbol {len(syms)}")
    return Xc, np.array(syms)


# ===================== 泛用 GEO(fallback)=====================
def load_generic(cfg):
    import GEOparse
    d = cfg["data"]
    gse = GEOparse.get_GEO(geo=d["geo_accession"], destdir=workdir(cfg), silent=True)
    expr = gse.pivot_samples("VALUE")
    genes = np.array(expr.index.astype(str))
    X = expr.T.to_numpy(dtype=float)
    labels = []
    for gsm in expr.columns:
        meta = " ".join(map(str, gse.gsms[gsm].metadata.get(
            d.get("phenotype_column", "characteristics_ch1"), []))).lower()
        if any(k.lower() in meta for k in d.get("case_keywords", [])):
            labels.append(1)
        elif any(k.lower() in meta for k in d.get("control_keywords", [])):
            labels.append(0)
        else:
            labels.append(-1)
    y = np.array(labels)
    keep = y >= 0
    X, y = X[keep], y[keep]
    good = ~np.all(np.isnan(X), axis=0)
    X, genes = X[:, good], genes[good]
    X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
    if np.nanmax(X) > 100:
        X = np.log2(X + 1.0)
    return X, y, genes


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    loader = cfg["data"].get("loader", "demo")
    if loader == "demo":
        X, y, genes = make_demo(cfg)
        print(f"[demo] 合成 X={X.shape}, 病例={int(y.sum())}, 對照={int((y==0).sum())}")
    elif loader == "gse56815":
        X, y, genes = load_gse56815(cfg)
    elif loader == "generic":
        X, y, genes = load_generic(cfg)
    else:
        raise ValueError(f"未知 loader: {loader}")

    out = f"{workdir(cfg)}/expression.npz"
    save_npz(out, X=X, y=y, genes=genes)
    print(f"已存檔 -> {out}")


if __name__ == "__main__":
    main()
