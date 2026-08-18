"""
Step 09 — 正交驗證(Orthogonal validation of novel candidates)
==================================================================
目的:把 Model B 網路模組挑出的「新穎候選基因」拿去跟「pipeline 從未用過」
      的兩類獨立骨骼證據對質,讓 Tier-2 提名(nomination)有機會升級為
      真正的優先發現(prioritised discovery)。

⚠ 科學誠信聲明:
  - 這一步只「重新排序既有提名」,不會拿來挑基因(不回饋進最佳化)。
  - 本腳本不會捏造任何結果。若 API 連不到或未提供本地資料檔,
    對應欄位一律標記為 "NOT_RETRIEVED",而不是填假值。
  - 兩類證據都與探索輸入(monocyte 表現量、STRING 拓樸、策展 seed 清單)
    正交,故可作為獨立佐證。

輸入(擇一):
  --candidates outputs/candidates_annotated.csv   (step06 產出;取 status==novel)
  若該檔不存在,退回內建的 WNT 模組優先候選清單(Table 3 的具名基因)。

兩類正交證據:
  (1) 小鼠單基因剔除的骨骼表型:
      International Mouse Phenotyping Consortium (IMPC) genotype-phenotype Solr
      + Origins of Bone and Cartilage Disease (OBCD) 快速骨表型計畫。
      一個基因若其剔除鼠出現顯著骨量/骨結構異常,記為 skeletal-phenotype-positive。
  (2) 人類遺傳學支持:
      最新 estimated-BMD (eBMD, UK Biobank heel QUS) GWAS 的基因層級訊號。
      預設查 GWAS Catalog REST API 的 BMD 相關 trait;
      也可用 --ebmd-genes 傳入 Morris et al. 2019 的基因清單做離線比對。

輸出:
  outputs/orthogonal_validation.csv
      gene, tier, mouse_species_symbol,
      impc_skeletal_hits, impc_bmd_hits, impc_mp_terms, impc_status,
      ebmd_gwas_hit, ebmd_traits, ebmd_status,
      corroborated                 (yes / no / unknown)
  outputs/orthogonal_validation_summary.md
      人可讀的摘要 + 每欄的資料來源與擷取日期(由 --run-date 提供)

用法範例:
  python step09_orthogonal_validation.py \
      --candidates outputs/candidates_annotated.csv \
      --run-date 2026-07-18
  # 離線 / 有本地 eBMD 清單時:
  python step09_orthogonal_validation.py --offline \
      --ebmd-genes data/ebmd_morris2019_genes.txt
"""
from __future__ import annotations
import sys
import os
import csv
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error

try:
    from common import load_config, workdir
except Exception:                     # 允許獨立執行,不依賴 config
    def workdir(_cfg=None):
        os.makedirs("outputs", exist_ok=True)
        return "outputs"
    def load_config(_p="config.yaml"):
        return {}


# ------------------------------------------------------------------
# 內建退回清單:Model B WNT 模組的具名優先候選(對應論文 Table 3)。
# 只有在讀不到 candidates_annotated.csv 時才使用。
# ------------------------------------------------------------------
FALLBACK_CANDIDATES = {
    "Tier 1 — druggable": ["NOTUM", "SFRP1", "DKK2", "WNT10B"],
    "Tier 2 — WNT machinery": ["RNF43", "KREMEN2", "SFRP2", "SFRP5",
                               "FZD1", "FZD2", "FZD8", "FZD10",
                               "PTK7", "GPC3", "DVL3",
                               "WNT7A", "WNT8A", "WNT8B"],
    "Likely artefact": ["HNF4A"],
}

# BMD 相關的 MP 術語關鍵字(用來從 IMPC 骨表型中再標出 BMD 專屬命中)
BMD_MP_KEYWORDS = ("bone mineral density", "bone mineral content")

# eBMD / BMD 相關的 GWAS trait 關鍵字
BMD_TRAIT_KEYWORDS = ("bone mineral density", "bone mineral content",
                      "estimated bone mineral density", "heel bone",
                      "bone density", "ebmd")

IMPC_SOLR = "https://www.ebi.ac.uk/mi/impc/solr/genotype-phenotype/select"
GWAS_GENE_ASSOC = "https://www.ebi.ac.uk/gwas/rest/api/genes/{gene}/associations"

NR = "NOT_RETRIEVED"      # 統一的「沒抓到 / 未執行」標記


# ------------------------------------------------------------------
def http_get_json(url, timeout=30, retries=2):
    """回傳 (obj, None) 或 (None, error_str)。絕不丟出例外中斷全流程。"""
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cuda-ising-osteo/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.loads(fh.read().decode("utf-8")), None
        except Exception as e:          # noqa: BLE001 — 網路層任何錯都要能優雅退回
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return None, last


def mouse_symbol(human_symbol: str) -> str:
    """人類基因符號 → 小鼠直系同源符號(近似:首字母大寫其餘小寫)。
    多數 WNT 家族成員此規則成立(NOTUM→Notum, WNT10B→Wnt10b, PTK7→Ptk7)。"""
    return human_symbol[:1].upper() + human_symbol[1:].lower()


# ================= (1) IMPC 小鼠骨表型 =================
def query_impc(human_symbol: str, offline: bool):
    """回傳 dict:skeletal_hits, bmd_hits, mp_terms, status。
    status ∈ {ok, NOT_RETRIEVED}。offline 或抓取失敗 → NOT_RETRIEVED。"""
    if offline:
        return dict(skeletal_hits=NR, bmd_hits=NR, mp_terms=NR, status=NR)
    sym = mouse_symbol(human_symbol)
    q = (f'?q=marker_symbol:{urllib.parse.quote(sym)}'
         f'&fq=top_level_mp_term_name:%22skeleton%20phenotype%22'
         f'&rows=200&wt=json'
         f'&fl=marker_symbol,mp_term_name,top_level_mp_term_name,p_value')
    obj, err = http_get_json(IMPC_SOLR + q)
    if obj is None:
        return dict(skeletal_hits=NR, bmd_hits=NR, mp_terms=NR, status=f"{NR} ({err})")
    docs = obj.get("response", {}).get("docs", [])
    mp_terms = sorted({d.get("mp_term_name", "") for d in docs if d.get("mp_term_name")})
    bmd_terms = [m for m in mp_terms if any(k in m.lower() for k in BMD_MP_KEYWORDS)]
    return dict(skeletal_hits=len(docs),
                bmd_hits=len(bmd_terms),
                mp_terms="; ".join(mp_terms) if mp_terms else "",
                status="ok")


# ================= (2) eBMD / BMD GWAS 支持 =================
def load_local_ebmd(path):
    if not path or not os.path.exists(path):
        return None
    genes = set()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            g = ln.strip().split("\t")[0].strip()
            if g and not g.startswith("#"):
                genes.add(g.upper())
    return genes


def query_gwas_catalog(human_symbol: str, offline: bool):
    """回傳 dict:hit(yes/no/NOT_RETRIEVED), traits, status。"""
    if offline:
        return dict(hit=NR, traits=NR, status=NR)
    url = GWAS_GENE_ASSOC.format(gene=urllib.parse.quote(human_symbol))
    url += "?projection=associationsByGene&size=200"
    obj, err = http_get_json(url)
    if obj is None:
        return dict(hit=NR, traits=NR, status=f"{NR} ({err})")
    assocs = obj.get("_embedded", {}).get("associations", [])
    bmd_traits = set()
    for a in assocs:
        for t in a.get("efoTraits", []):
            label = (t.get("trait") or "").lower()
            if any(k in label for k in BMD_TRAIT_KEYWORDS):
                bmd_traits.add(t.get("trait"))
    return dict(hit="yes" if bmd_traits else "no",
                traits="; ".join(sorted(bmd_traits)),
                status="ok")


# ------------------------------------------------------------------
def read_candidates(path):
    """讀 step06 的 candidates_annotated.csv,取 status==novel 者。
    回傳 [(gene, tier_guess)]。tier 這裡未知,先標 'novel (unranked)'。"""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            status = (r.get("status") or "").strip().lower()
            if status == "novel":
                out.append((r["gene"].strip().upper(), "novel (unranked)"))
    return out


def gather_candidates(args):
    if args.candidates and os.path.exists(args.candidates):
        rows = read_candidates(args.candidates)
        if rows:
            print(f"[step09] 由 {args.candidates} 讀入 {len(rows)} 個 novel 候選")
            return rows
        print(f"[step09] {args.candidates} 中沒有 status==novel 的列,改用內建清單")
    else:
        print("[step09] 找不到 candidates 檔,改用內建 Table 3 具名清單")
    rows = []
    for tier, genes in FALLBACK_CANDIDATES.items():
        for g in genes:
            rows.append((g, tier))
    return rows


# ------------------------------------------------------------------
def corroborated_flag(impc, ebmd):
    """任一正交證據為陽性 → yes;兩者皆確定為陰性 → no;有未抓到 → unknown。"""
    impc_pos = isinstance(impc["skeletal_hits"], int) and impc["skeletal_hits"] > 0
    ebmd_pos = ebmd["hit"] == "yes"
    if impc_pos or ebmd_pos:
        return "yes"
    impc_known_neg = impc["status"] == "ok" and impc["skeletal_hits"] == 0
    ebmd_known_neg = ebmd["status"] == "ok" and ebmd["hit"] == "no"
    if impc_known_neg and ebmd_known_neg:
        return "no"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Step 09 — orthogonal validation of novel candidates")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--candidates", default="outputs/candidates_annotated.csv")
    ap.add_argument("--ebmd-genes", default="", help="Morris 2019 eBMD 基因清單(離線比對用)")
    ap.add_argument("--offline", action="store_true", help="完全不連網;所有 API 欄位標 NOT_RETRIEVED")
    ap.add_argument("--run-date", default="", help="擷取日期(寫進摘要;不影響結果)")
    ap.add_argument("--sleep", type=float, default=0.5, help="每個基因之間的禮貌等待(秒)")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
        wd = workdir(cfg)
    except Exception:
        wd = workdir()

    candidates = gather_candidates(args)
    local_ebmd = load_local_ebmd(args.ebmd_genes)
    if local_ebmd is not None:
        print(f"[step09] 已載入本地 eBMD 清單:{len(local_ebmd)} 個基因")

    rows = []
    for gene, tier in candidates:
        impc = query_impc(gene, args.offline)

        if local_ebmd is not None:
            hit = "yes" if gene.upper() in local_ebmd else "no"
            ebmd = dict(hit=hit, traits="Morris 2019 eBMD gene list", status="ok(local)")
        else:
            ebmd = query_gwas_catalog(gene, args.offline)

        rows.append(dict(
            gene=gene, tier=tier, mouse_species_symbol=mouse_symbol(gene),
            impc_skeletal_hits=impc["skeletal_hits"], impc_bmd_hits=impc["bmd_hits"],
            impc_mp_terms=impc["mp_terms"], impc_status=impc["status"],
            ebmd_gwas_hit=ebmd["hit"], ebmd_traits=ebmd["traits"], ebmd_status=ebmd["status"],
            corroborated=corroborated_flag(impc, ebmd),
        ))
        print(f"  {gene:8s} tier={tier:24s} IMPC={impc['skeletal_hits']} "
              f"eBMD={ebmd['hit']} -> {rows[-1]['corroborated']}")
        if not args.offline:
            time.sleep(args.sleep)

    # ---- 寫 CSV ----
    cols = ["gene", "tier", "mouse_species_symbol",
            "impc_skeletal_hits", "impc_bmd_hits", "impc_mp_terms", "impc_status",
            "ebmd_gwas_hit", "ebmd_traits", "ebmd_status", "corroborated"]
    out_csv = os.path.join(wd, "orthogonal_validation.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- 寫 Markdown 摘要 ----
    n = len(rows)
    yes = sum(1 for r in rows if r["corroborated"] == "yes")
    no = sum(1 for r in rows if r["corroborated"] == "no")
    unk = sum(1 for r in rows if r["corroborated"] == "unknown")
    any_nr = any("NOT_RETRIEVED" in str(r["impc_status"]) or "NOT_RETRIEVED" in str(r["ebmd_status"])
                 for r in rows)
    out_md = os.path.join(wd, "orthogonal_validation_summary.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# Orthogonal validation of novel candidates\n\n")
        if args.run_date:
            fh.write(f"Retrieval date: {args.run_date}\n\n")
        fh.write("Evidence sources (both orthogonal to the discovery inputs):\n\n")
        fh.write("1. **IMPC** genotype-phenotype (skeleton phenotype) — single-gene mouse knockouts; "
                 "https://www.ebi.ac.uk/mi/impc/ . Complemented by the **OBCD** rapid-throughput "
                 "skeletal-phenotyping programme (https://boneandcartilage.com/), consulted manually.\n")
        fh.write("2. **eBMD GWAS** (UK Biobank heel QUS) via GWAS Catalog gene associations "
                 "(https://www.ebi.ac.uk/gwas/), or a local Morris et al. 2019 eBMD gene list.\n\n")
        fh.write(f"- Candidates evaluated: **{n}**\n")
        fh.write(f"- Independently corroborated (IMPC skeletal hit and/or eBMD GWAS hit): **{yes}**\n")
        fh.write(f"- Confirmed negative in both retrieved sources: **{no}**\n")
        fh.write(f"- Unresolved (at least one source NOT_RETRIEVED): **{unk}**\n\n")
        if any_nr:
            fh.write("> ⚠ **Some fields were NOT_RETRIEVED** (offline run or API unreachable). "
                     "Those genes are reported as `unknown`, never as validated. "
                     "Re-run online to complete.\n\n")
        fh.write("| Gene | Tier | IMPC skeletal | IMPC BMD | eBMD GWAS | Corroborated |\n")
        fh.write("|------|------|---------------|----------|-----------|--------------|\n")
        for r in rows:
            fh.write(f"| {r['gene']} | {r['tier']} | {r['impc_skeletal_hits']} | "
                     f"{r['impc_bmd_hits']} | {r['ebmd_gwas_hit']} | {r['corroborated']} |\n")

    print(f"\n[step09] wrote {out_csv}")
    print(f"[step09] wrote {out_md}")
    print(f"[step09] corroborated={yes}  negative={no}  unknown={unk}  (n={n})")
    if any_nr:
        print("[step09] ⚠ 有欄位為 NOT_RETRIEVED:離線或 API 連不到,請上線重跑以完成。")


if __name__ == "__main__":
    main()
