"""Model B — GWAS + STRING active-module detection -> ising.npz + features.npz.

Objective (minimize), x_i in {0,1}:
   E(x) = -alpha * sum_i g_i x_i               # reward high-GWAS genes (node score = h)
          - beta  * sum_{(i,j) in PPI} x_i x_j  # reward selecting connected pairs (STRING edge)
          + lam   * (sum_i x_i - k)^2           # size ~ k
=> a connected, high-GWAS module. Reuses step03.qubo_to_ising for the QUBO->Ising step,
   so solve_gpu.py + step05_postprocess.py consume the output unchanged.

Inputs (real):
  --scores  TSV 'gene<TAB>score' (gene-level GWAS, e.g. MAGMA -log10 p, or GWAS-Catalog).
  --string-links / --string-info  STRING human files (9606). combined_score >= --edge-threshold.
Or --demo for an offline self-test with a planted high-score connected module.
"""
from __future__ import annotations
import sys, argparse, gzip
import numpy as np
from collections import defaultdict
from common import load_config, workdir, save_npz
import step03_build_qubo as q3


def _open(f):
    return gzip.open(f, "rt") if f.endswith(".gz") else open(f, encoding="utf-8")


def read_scores(path):
    d = {}
    with _open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    d[p[0].strip().upper()] = float(p[1])
                except ValueError:
                    pass
    return d


def read_gwas_catalog(path, keywords=("bone mineral density", "bone density", "osteoporosis")):
    """Build gene-level scores from a GWAS Catalog full-associations TSV.
    Keep rows whose trait mentions BMD/osteoporosis; per gene score = max(PVALUE_MLOG)."""
    import csv, re
    scores = {}
    with _open(path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            trait = ((row.get("MAPPED_TRAIT", "") or "") + " " +
                     (row.get("DISEASE/TRAIT", "") or "")).lower()
            if not any(k in trait for k in keywords):
                continue
            try:
                mlog = float(row.get("PVALUE_MLOG", "") or "nan")
            except ValueError:
                continue
            if mlog != mlog:
                continue
            for g in re.split(r"[,\s;/-]+", row.get("MAPPED_GENE", "") or ""):
                g = g.strip().upper()
                if g and g not in ("NR", "INTERGENIC") and re.match(r"^[A-Z0-9][A-Z0-9.\-]*$", g):
                    scores[g] = max(scores.get(g, 0.0), mlog)
    return scores


def read_string(links, info, thr):
    id2sym = {}
    with _open(info) as fh:
        next(fh, None)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2:
                id2sym[c[0]] = c[1].strip().upper()
    edges = defaultdict(set)
    with _open(links) as fh:
        next(fh, None)
        for line in fh:
            c = line.split()
            if len(c) >= 3 and int(c[2]) >= thr:
                a, b = id2sym.get(c[0]), id2sym.get(c[1])
                if a and b and a != b:
                    edges[a].add(b); edges[b].add(a)
    return edges


def build_demo(n=300, seed=0, mod=15):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n)]
    scores = {g: float(rng.random()) for g in genes}
    for i in range(mod):
        scores[f"G{i}"] = 2.0 + rng.random()          # planted high-score module
    edges = defaultdict(set)
    for i in range(mod):                               # densely interconnect the module
        for j in range(i + 1, mod):
            if rng.random() < 0.6:
                edges[f"G{i}"].add(f"G{j}"); edges[f"G{j}"].add(f"G{i}")
    for _ in range(n):                                 # random background edges
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            edges[f"G{a}"].add(f"G{b}"); edges[f"G{b}"].add(f"G{a}")
    return scores, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--scores"); ap.add_argument("--gwas-catalog")
    ap.add_argument("--string-links"); ap.add_argument("--string-info")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--top-genes", type=int, default=300)
    ap.add_argument("--edge-threshold", type=int, default=700)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()
    cfg = load_config(a.config); wd = workdir(cfg)

    if a.demo:
        scores, edges = build_demo()
    else:
        scores = read_gwas_catalog(a.gwas_catalog) if a.gwas_catalog else read_scores(a.scores)
        print(f"gene scores: {len(scores)} genes")
        edges = read_string(a.string_links, a.string_info, a.edge_threshold)

    genes = sorted(scores, key=lambda x: scores[x], reverse=True)[:a.top_genes]
    idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    g = np.array([scores[x] for x in genes], dtype=float)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)     # node score -> [0,1]

    A = np.zeros((n, n))
    for u in genes:
        for v in edges.get(u, ()):
            if v in idx:
                A[idx[u], idx[v]] = 1.0
    A = np.triu(A, 1)
    n_edges = int(A.sum())

    # QUBO:  Qdiag from -alpha*g + cardinality;  Qoff = -beta*A (reward edges) + 2*lam (size)
    Qdiag = -a.alpha * g + a.lam * (1.0 - 2.0 * a.k)
    Qoff = -a.beta * A + 2.0 * a.lam * np.triu(np.ones((n, n)), 1)
    h, J, offset = q3.qubo_to_ising(Qdiag, Qoff)
    offset += a.lam * a.k * a.k

    save_npz(f"{wd}/ising.npz", h=h, J=J, genes=np.array(genes), offset=np.array([offset]))
    save_npz(f"{wd}/features.npz", R=g, C=A, genes=np.array(genes))
    print(f"nodes={n}, PPI edges among them={n_edges}, target k={a.k}, "
          f"spins={len(h)}, non-zero J={int((np.triu(J,1)!=0).sum())}")
    print(f"top-score genes: {', '.join(genes[:10])}")
    print(f"-> {wd}/ising.npz  +  {wd}/features.npz  (now run solve_gpu.py then step05)")


if __name__ == "__main__":
    main()
