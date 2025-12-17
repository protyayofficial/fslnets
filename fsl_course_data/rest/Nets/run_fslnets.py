#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import glob
import platform
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---- Headless matplotlib (MUST be before pyplot import) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fsl import nets


# -----------------------------
# Helpers
# -----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def require_path(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


def savefig(path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # tight_layout can warn in large figures; ok, but we try.
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close()


def vech_index_to_ij(k: int, n: int) -> Tuple[int, int]:
    """
    Map an index k in a 'vech' vector (lower-triangular, excluding diagonal)
    to matrix indices (i, j) with i > j.

    Assumes ordering:
      (1,0), (2,0),(2,1), (3,0),(3,1),(3,2), ...
    i.e., row-major over i, then j.
    """
    # Find smallest i such that k < i*(i)/2? Actually number of pairs up to row i is i*(i-1)/2.
    # We want row i contributes i entries (j=0..i-1) excluding diagonal => i entries? For i=1 contributes 1.
    # Cumulative pairs up to i inclusive: i*(i+1)/2? let's derive with i starting at 1:
    # pairs for row r is r (j=0..r-1). cumulative up to r is r*(r+1)/2.
    # Let r = i-1? We'll just loop; n is small (<=100).
    c = 0
    for i in range(1, n):
        if k < c + i:
            j = k - c
            return (i, j)
        c += i
    raise IndexError(f"k={k} out of range for n={n} (max {n*(n-1)//2 - 1})")


def print_top_edges(p1m: np.ndarray, tstats: np.ndarray, ts_nodes: List[int], topk: int = 20) -> None:
    """
    p1m: vector of 1-p values (corrected or uncorrected) for each edge
    tstats: vector of t statistics aligned with p1m (if available, else NaN)
    ts_nodes: mapping from post-clean index -> original ICA node id (per FSLNets practical note)
    """
    n_nodes = len(ts_nodes)
    n_edges_expected = n_nodes * (n_nodes - 1) // 2
    log(f"[CHECK] nodes={n_nodes}, expected edges={n_edges_expected}, pvec={p1m.size}")

    if p1m.size != n_edges_expected:
        log("[WARN] Edge vector length does not match n*(n-1)/2; edge->(i,j) mapping may be wrong.")
        log("       Still printing top edges by index (k), but i/j may not correspond.")
    idx = np.argsort(p1m)[::-1][:topk]

    log("")
    log("------------------------------------------------------------")
    log("Top edges by 1-p value (so higher = more significant).")
    log("Reminder: p = 1 - (1-p). Values > 0.95 correspond to p<0.05. ")
    log("------------------------------------------------------------")
    header = f"{'rank':>4} | {'k':>7} | {'i':>4} | {'j':>4} | {'orig_i':>6} | {'orig_j':>6} | {'t':>10} | {'1-p':>10} | {'p':>10}"
    log(header)
    log("-" * len(header))

    for r, k in enumerate(idx, 1):
        try:
            i, j = vech_index_to_ij(int(k), n_nodes)
            orig_i = ts_nodes[i]
            orig_j = ts_nodes[j]
        except Exception:
            i = j = -1
            orig_i = orig_j = -1

        one_minus_p = float(p1m[k])
        pval = 1.0 - one_minus_p
        tval = float(tstats[k]) if (tstats is not None and tstats.size == p1m.size) else float("nan")
        log(f"{r:4d} | {int(k):7d} | {i:4d} | {j:4d} | {orig_i:6d} | {orig_j:6d} | {tval:10.4f} | {one_minus_p:10.7f} | {pval:10.7f}")

    log("------------------------------------------------------------")
    log("")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    BASE = Path(__file__).resolve().parent
    os.chdir(BASE)

    outdir = BASE / "fslnets_headless_outputs_verify"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Environment prints ----
    log("[ENV] Starting FSLNets headless verification run")
    log(f"[ENV] cwd              = {Path.cwd()}")
    log(f"[ENV] python           = {sys.executable}")
    log(f"[ENV] python_version   = {sys.version.replace(os.linesep, ' ')}")
    log(f"[ENV] platform         = {platform.platform()}")
    log(f"[ENV] MPLBACKEND       = {os.environ.get('MPLBACKEND', '')}")
    log(f"[ENV] FSLDIR           = {os.environ.get('FSLDIR', '')}")
    log(f"[ENV] FSLOUTPUTTYPE    = {os.environ.get('FSLOUTPUTTYPE', '')}")

    # Confirm FSL commands likely available (randomise needed for nets.glm)
    # We don't call subprocess here; nets.glm will fail loudly if FSL is missing.
    log("[ENV] (Note) nets.glm calls FSL randomise under the hood; FSLDIR must be set.")

    # ---- Input existence checks ----
    dr_dir = BASE / "groupICA100.dr"
    sum_dir = BASE / "groupICA100.sum"
    design_mat = BASE / "design" / "unpaired_ttest_1con.mat"
    design_con = BASE / "design" / "unpaired_ttest_1con.con"

    require_path(dr_dir, "Dual regression timeseries directory (groupICA100.dr)")
    require_path(sum_dir, "Thumbnail directory (groupICA100.sum)")
    require_path(design_mat, "Design matrix (.mat)")
    require_path(design_con, "Design contrast (.con)")

    # Show what files we see in groupICA100.dr (helps diagnose load issues)
    sample_files = sorted([p for p in dr_dir.rglob("*") if p.is_file()])[:20]
    log(f"[INPUT] groupICA100.dr exists. Showing first {len(sample_files)} files:")
    for p in sample_files:
        log(f"        - {p.relative_to(BASE)}")

    # ---- Load timeseries (as in practical) ----
    TR = 0.72
    log(f"[STEP 1] Loading timeseries with nets.load(dr='{dr_dir.name}', TR={TR}, varnorm=0)")
    ts = nets.load(str(dr_dir), TR, varnorm=0, thumbnaildir=str(sum_dir))
    # The practical notes ts.ts contains the stage-1 dual regression timeseries. :contentReference[oaicite:2]{index=2}
    try:
        n_sub = len(ts.ts)
    except Exception:
        n_sub = None
    log(f"[OK] Loaded. n_subjects (len(ts.ts)) = {n_sub}")

    # ---- Plot spectra (saved) ----
    log("[STEP 2] Plotting spectra -> saving 01_spectra.png")
    nets.plot_spectra(ts)
    savefig(outdir / "01_spectra.png")

    # ---- Good nodes list (from course practical) ----
    goodnodes = [
        0,1,2,4,5,6,7,8,10,11,12,16,17,18,19,20,21,22,24,25,
        26,27,28,29,30,31,32,33,34,35,36,37,39,41,42,46,47,48,49,51,
        52,54,55,56,57,58,60,61,63,64,65,69,70,71,72,73,76,79,80,85,
        86,92,96
    ]
    log(f"[STEP 3] Cleaning timeseries (aggressive=True) with goodnodes count = {len(goodnodes)}")

    # ---- Clean ----
    nets.clean(ts, goodnodes, True)

    # After clean, indices are re-indexed; ts.nodes maps post-clean -> original ICA node id. :contentReference[oaicite:3]{index=3}
    try:
        nodes_map = list(ts.nodes)
        log(f"[OK] Cleaned. Post-clean node count = {len(nodes_map)}")
        log(f"[INFO] First 20 ts.nodes mappings (post-clean -> original): {nodes_map[:20]}")
    except Exception as e:
        nodes_map = []
        log(f"[WARN] Could not read ts.nodes mapping: {e}")

    # ---- Netmats ----
    log("[STEP 4] Computing netmats: full corr and ridge partial (alpha=0.1)")
    Fnetmats = nets.netmats(ts, "corr", True)
    Pnetmats = nets.netmats(ts, "ridgep", True, 0.1)

    log(f"[OK] Fnetmats shape = {getattr(Fnetmats, 'shape', None)}  size={Fnetmats.size}")
    log(f"[OK] Pnetmats shape = {getattr(Pnetmats, 'shape', None)}  size={Pnetmats.size}")

    # Save
    np.save(outdir / "Fnetmats_corr.npy", Fnetmats)
    np.save(outdir / "Pnetmats_ridgep.npy", Pnetmats)

    # Derived counts
    if Pnetmats.ndim == 2:
        n_subjects = Pnetmats.shape[0]
        n_edges = Pnetmats.shape[1]
        log(f"[CHECK] Derived n_subjects={n_subjects}, n_edges={n_edges}, product={n_subjects*n_edges}")
    else:
        log("[WARN] Pnetmats not 2D; unexpected. Check saved array.")

    # ---- Group means ----
    log("[STEP 5] Group means (Znet/Mnet for full and partial)")
    Znet_F, Mnet_F = nets.groupmean(ts, Fnetmats, False)
    Znet_P, Mnet_P = nets.groupmean(ts, Pnetmats, True, "Partial correlation")

    np.save(outdir / "Znet_F.npy", Znet_F)
    np.save(outdir / "Mnet_F.npy", Mnet_F)
    np.save(outdir / "Znet_P.npy", Znet_P)
    np.save(outdir / "Mnet_P.npy", Mnet_P)

    log(f"[OK] Mnet_F shape = {Mnet_F.shape}, Mnet_P shape = {Mnet_P.shape}")

    # Course spot-check: Mnet_P[2,26] ~ 6.6 in the practical dataset/settings. :contentReference[oaicite:4]{index=4}
    try:
        val = float(Mnet_P[2, 26])
        log(f"[CHECK] Course spot-check Mnet_P[2,26] = {val:.6f}  (expected ~6.6 in the course example)")
    except Exception as e:
        log(f"[WARN] Could not compute Mnet_P[2,26] spot-check: {e}")

    # ---- Hierarchy plot ----
    log("[STEP 6] Plotting hierarchy -> saving 02_hierarchy.png")
    nets.plot_hierarchy(ts, Znet_F, Znet_P, "Full correlations", "Partial correlations")
    savefig(outdir / "02_hierarchy.png")

    # ---- GLM (randomise) ----
    # randomise outputs are 1-p values; >0.95 corresponds to p<0.05. :contentReference[oaicite:5]{index=5}
    log("[STEP 7] GLM via nets.glm (calls FSL randomise). Outputs are 1-p values (randomise convention).")
    p_corr, p_uncorr = nets.glm(ts, Pnetmats, str(design_mat), str(design_con))

    # Save p outputs (object arrays)
    np.save(outdir / "p_corr.npy", np.array(p_corr, dtype=object))
    np.save(outdir / "p_uncorr.npy", np.array(p_uncorr, dtype=object))

    # ---- Summaries ----
    # p_corr / p_uncorr are typically lists (one per contrast); use contrast 0
    pc0 = np.asarray(p_corr[0]).astype(float)
    pu0 = np.asarray(p_uncorr[0]).astype(float)

    log(f"[OK] GLM done. Contrast count (len(p_corr)) = {len(p_corr)}")
    log(f"[OK] p_corr[0] shape={pc0.shape}, min={pc0.min():.6f}, max={pc0.max():.6f}")
    log(f"[OK] p_uncorr[0] shape={pu0.shape}, min={pu0.min():.6f}, max={pu0.max():.6f}")

    # Count significant edges using corrected threshold 0.95 on 1-p values. :contentReference[oaicite:6]{index=6}
    n_sig = int(np.sum(pc0 > 0.95))
    log(f"[CHECK] # corrected-significant edges (p_corr[0] > 0.95) = {n_sig}")

    # If nets.glm also produced t-stats aligned with edges, we can show them.
    # Some versions return t-stats separately; if not available, we print NaN.
    tstats = None
    # Common pattern: p_corr is list of vectors; t-stats not returned.
    # We'll attempt to compute a crude "t-like" ranking later only from p, so leave tstats None.

    # Print top edges (corrected)
    if nodes_map:
        print_top_edges(pc0, tstats=np.full_like(pc0, np.nan), ts_nodes=nodes_map, topk=20)
    else:
        log("[WARN] ts.nodes mapping unavailable; printing top edges without orig node mapping.")
        print_top_edges(pc0, tstats=np.full_like(pc0, np.nan), ts_nodes=list(range(int((1+np.sqrt(1+8*pc0.size))/2))), topk=20)

    # ---- Boxplots ----
    log("[STEP 8] Boxplots -> saving 03_boxplots.png")
    nets.boxplots(ts, Pnetmats, Znet_P, pc0, groups=(6, 6))
    savefig(outdir / "03_boxplots.png")

    # ---- Write summary JSON ----
    summary = {
        "cwd": str(Path.cwd()),
        "outdir": str(outdir),
        "TR": TR,
        "varnorm": 0,
        "goodnodes_count": len(goodnodes),
        "n_subjects": int(Pnetmats.shape[0]) if getattr(Pnetmats, "ndim", 0) == 2 else None,
        "n_edges": int(Pnetmats.shape[1]) if getattr(Pnetmats, "ndim", 0) == 2 else None,
        "glm": {
            "contrast_count": int(len(p_corr)),
            "p_corr0_min": float(pc0.min()),
            "p_corr0_max": float(pc0.max()),
            "n_sig_corr_edges_1mp_gt_0p95": int(n_sig),
        },
        "notes": [
            "randomise-style outputs are 1-p values; >0.95 corresponds to p<0.05",
            "post-clean node indices are re-mapped; ts.nodes maps post-clean index -> original ICA component index",
        ],
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"[DONE] All outputs saved to: {outdir}")
    log(f"[DONE] Summary written to: {outdir / 'run_summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("[FAIL] Run failed with exception:")
        log(str(e))
        raise
