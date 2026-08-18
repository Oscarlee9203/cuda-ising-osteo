"""
Step 06 — 通路富集 + Open Targets「已知 vs 新穎」標註
==================================================================
輸入:outputs/candidates.csv (step05 產出:gene, select_freq, R)
輸出:
    outputs/candidates_annotated.csv
        gene, select_freq, R, ot_score, status(known/novel), known_source
    outputs/enrichment.csv
        term, library, p_value, fdr, overlap, genes

兩件事:
  (1) 新穎性:用 Open Targets Platform GraphQL 查該疾病(預設 osteoporosis,
      EFO_0003882)的關聯 target 分數;分數 >= 門檻者視為「已知」,其餘「新穎」。
      連不到 API 時,退回 data/known_osteoporosis_genes.txt 這份策展清單。
  (2) 富集:對前 N 個候選基因做通路富集。
      method=auto 會先試 gseapy(Enrichr,需網路),失敗改用本地超幾何檢定
      (需你提供 GMT 檔;離線也能算)。

⚠ 這一步是「pipeline 之外的下游詮釋」:結果只是把候選排序、標出哪些已知,
   真正的新標的仍需濕實驗驗證。
"""
from __future__ import annotations
import sys
import os
import csv
import json
import urllib.request

import numpy as np
from scipy import stats

from common import load_config, workdir


# ------------------------------------------------------------------
def read_candidates(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"gene": r["gene"],
                         "select_freq": float(r["select_freq"]),
                         "R": float(r.get("R", "nan") or "nan")})
    return rows


# ================= (1) Open Targets 新穎性 =================
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_QUERY = """
query($efoId:String!,$size:Int!){
  disease(efoId:$efoId){
    id name
    associatedTargets(page:{index:0,size:$size}){
      count
      rows{ target{approvedSymbol} score }
    }
  }
}"""


def opentargets_known(efo_id, size, timeout=30):
    """回傳 {symbol: score};失敗回傳 None(交由呼叫端 fallback)。"""
    body = json.dumps({"query": OT_QUERY,
                       "variables": {"efoId": efo_id, "size": int(size)}}).encode()
    req = urllib.request.Request(OT_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        d = json.load(r)
        rows = d["data"]["disease"]["associatedTargets"]["rows"]
        return {row["target"]["approvedSymbol"]: float(row["score"]) for row in rows}
    except Exception as e:
        print(f"  [Open Targets 連線失敗:{type(e).__name__}] 改用離線策展清單。")
        return None


def load_fallback_known(path):
    genes = set()
    if not os.path.exists(path):
        print(f"  [警告] 找不到 fallback 清單 {path}")
        return genes
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                genes.add(line.upper())
    return genes


def annotate_novelty(cands, cfg):
    nv = cfg["novelty"]
    thr = nv["ot_score_threshold"]
    ot = opentargets_known(nv["efo_id"], nv.get("ot_page_size", 500))
    if ot is not None:
        known_scores = {g.upper(): s for g, s in ot.items() if s >= thr}
        source = f"opentargets>={thr}"
        print(f"  Open Targets:{nv['disease_name']} 關聯 target "
              f"{len(ot)} 個,分數≥{thr} 視為已知 {len(known_scores)} 個")
    else:
        fb = load_fallback_known(nv["known_genes_fallback"])
        known_scores = {g: float("nan") for g in fb}
        source = "curated_fallback"
        print(f"  離線清單:已知基因 {len(known_scores)} 個")

    for c in cands:
        g = c["gene"].upper()
        if g in known_scores:
            c["status"] = "known"
            c["ot_score"] = known_scores[g]
            c["known_source"] = source
        else:
            c["status"] = "novel"
            c["ot_score"] = float("nan")
            c["known_source"] = ""
    return cands


# ================= (2) 通路富集 =================
def read_gmt(path):
    sets = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term, _desc, *genes = parts
            sets[term] = {g.upper() for g in genes if g}
    return sets


def enrich_local(query_genes, gmt_sets, background, fdr_cut):
    """超幾何檢定(離線)。background = 背景基因宇宙(通常是量測到的所有基因)。"""
    Q = {g.upper() for g in query_genes} & background
    N = len(background)
    results = []
    for term, gs in gmt_sets.items():
        K = len(gs & background)
        if K == 0:
            continue
        overlap = Q & gs
        k = len(overlap)
        if k == 0:
            continue
        p = stats.hypergeom.sf(k - 1, N, K, len(Q))
        results.append({"term": term, "library": "local_gmt", "p_value": p,
                        "overlap": f"{k}/{K}", "genes": ";".join(sorted(overlap))})
    results.sort(key=lambda x: x["p_value"])
    # BH-FDR
    m = len(results)
    for i, r in enumerate(results, 1):
        r["fdr"] = min(1.0, r["p_value"] * m / i)
    for r in results:
        r["fdr"] = min(r["fdr"], 1.0)
    return [r for r in results if r["fdr"] <= fdr_cut] or results[:20]


def enrich_enrichr(query_genes, libs, fdr_cut):
    """gseapy Enrichr(需網路)。失敗回傳 None。"""
    try:
        import gseapy as gp
    except Exception:
        print("  [gseapy 未安裝] 略過 Enrichr。")
        return None
    try:
        enr = gp.enrichr(gene_list=list(query_genes), gene_sets=list(libs),
                         organism="human", outdir=None)
        df = enr.results
        out = []
        for _, row in df.iterrows():
            out.append({"term": row["Term"], "library": row["Gene_set"],
                        "p_value": float(row["P-value"]),
                        "fdr": float(row["Adjusted P-value"]),
                        "overlap": row["Overlap"], "genes": row["Genes"]})
        out.sort(key=lambda x: x["fdr"])
        return [r for r in out if r["fdr"] <= fdr_cut] or out[:20]
    except Exception as e:
        print(f"  [Enrichr 連線失敗:{type(e).__name__}] 改用本地 GMT(若有)。")
        return None


def run_enrichment(query_genes, cfg):
    en = cfg["enrichment"]
    method = en.get("method", "auto")
    fdr_cut = en.get("fdr", 0.05)

    if method in ("auto", "enrichr"):
        res = enrich_enrichr(query_genes, en["enrichr_libraries"], fdr_cut)
        if res is not None:
            return res
        if method == "enrichr":
            return []

    # local
    gmt = en.get("gmt_path")
    if gmt and os.path.exists(gmt):
        background = _load_background(cfg)
        return enrich_local(query_genes, read_gmt(gmt), background, fdr_cut)
    print("  [無 GMT] 略過富集(請設 enrichment.gmt_path 或用線上 Enrichr)。")
    return []


def _load_background(cfg):
    from common import load_npz
    p = f"{workdir(cfg)}/features.npz"
    if os.path.exists(p):
        return {str(g).upper() for g in load_npz(p)["genes"]}
    return set()


# ================= 主流程 =================
def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    wd = workdir(cfg)
    cands = read_candidates(f"{wd}/candidates.csv")

    print("[1/2] 標註新穎性 (Open Targets)...")
    cands = annotate_novelty(cands, cfg)

    print("[2/2] 通路富集...")
    top = cfg["enrichment"].get("top_candidates", 50)
    cands_sorted = sorted(cands, key=lambda c: c["select_freq"], reverse=True)
    query = [c["gene"] for c in cands_sorted[:top]]
    enr = run_enrichment(query, cfg)

    # ---- 輸出 ----
    ann = f"{wd}/candidates_annotated.csv"
    with open(ann, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "select_freq", "R", "ot_score", "status", "known_source"])
        for c in cands_sorted:
            w.writerow([c["gene"], f"{c['select_freq']:.4f}", f"{c['R']:.4f}",
                        "" if np.isnan(c["ot_score"]) else f"{c['ot_score']:.4f}",
                        c["status"], c["known_source"]])
    print(f"  -> {ann}")

    enp = f"{wd}/enrichment.csv"
    with open(enp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["term", "library", "p_value", "fdr", "overlap", "genes"])
        for r in enr:
            w.writerow([r["term"], r["library"], f"{r['p_value']:.3e}",
                        f"{r['fdr']:.3e}", r["overlap"], r["genes"]])
    print(f"  -> {enp}  (顯著條目 {len(enr)})")

    n_known = sum(c["status"] == "known" for c in cands)
    n_novel = len(cands) - n_known
    print(f"\n候選 {len(cands)} 個:已知 {n_known}、新穎 {n_novel}")
    print("新穎候選 Top 10(依 select_freq):")
    for c in [x for x in cands_sorted if x["status"] == "novel"][:10]:
        print(f"  {c['gene']:12s} freq={c['select_freq']:.3f} R={c['R']:.3f}")


if __name__ == "__main__":
    main()
