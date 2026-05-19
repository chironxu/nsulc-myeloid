#!/usr/bin/env python3
"""
Patient-level pseudobulk sensitivity analysis — simplified version.
Input: myeloid_visualization/myeloid_annotated.h5ad
  - sampleID: patient ID
  - MPRtype: nPR / pCR / pPR / MPR_but_not_pCR
  - cell_type: 17 myeloid subtypes
Keeps only nPR and pCR for analysis. Drops pPR and pCR-like.

Usage:
  python pseudobulk_analysis.py --h5ad myeloid_visualization/myeloid_annotated.h5ad
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import argparse, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10,
                     'axes.labelsize': 12, 'axes.titlesize': 14, 'figure.dpi': 100})

OUTPUT_DIR = Path("pseudobulk_analysis")
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "data").mkdir(parents=True, exist_ok=True)

NPR_COLOR = "#F94144"
PCR_COLOR = "#0077B6"


def compare_groups(df, value_col, group_col, group_a, group_b):
    """Mann-Whitney U + Cohen's d with bootstrap 95% CI."""
    a = df[df[group_col] == group_a][value_col].dropna().values
    b = df[df[group_col] == group_b][value_col].dropna().values
    if len(a) < 3 or len(b) < 3:
        return {}
    stat, pval = mannwhitneyu(a, b, alternative="two-sided")
    pooled = np.sqrt(((len(a)-1)*np.var(a)+(len(b)-1)*np.var(b)) / (len(a)+len(b)-2))
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0
    # Bootstrap CI
    ds = []
    for _ in range(1000):
        ab = np.random.choice(a, len(a), replace=True)
        bb = np.random.choice(b, len(b), replace=True)
        ps = np.sqrt(((len(a)-1)*np.var(ab)+(len(b)-1)*np.var(bb)) / (len(a)+len(b)-2))
        ds.append((ab.mean()-bb.mean())/ps if ps > 0 else 0)
    ci = np.percentile(ds, [2.5, 97.5])
    return {"value": value_col, "U": stat, "P": pval, "Cohen_d": d,
            "CI_95_low": ci[0], "CI_95_high": ci[1], "n_a": len(a), "n_b": len(b)}


def pseudobulk(adata, genes):
    """Per-patient mean expression of given genes."""
    rows = []
    for pid in adata.obs["sampleID"].unique():
        pcells = adata[adata.obs["sampleID"] == pid]
        row = {"patient": pid, "n_cells": pcells.n_obs,
               "MPRtype": pcells.obs["MPRtype"].iloc[0]}
        for g in genes:
            if g in adata.var_names:
                expr = pcells[:, g].X
                row[g] = np.mean(expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr))
            else:
                row[g] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pseudobulk_pathway(adata, pw_genes, pw_name):
    """Per-patient mean pathway score."""
    genes_found = [g for g in pw_genes if g in adata.var_names]
    if not genes_found:
        print(f"    WARNING: no genes found for {pw_name}, skipping")
        return None
    rows = []
    for pid in adata.obs["sampleID"].unique():
        pcells = adata[adata.obs["sampleID"] == pid]
        if pcells.n_obs == 0:
            continue
        scores = []
        for g in genes_found:
            expr = pcells[:, g].X
            v = expr.toarray().flatten() if hasattr(expr, "toarray") else np.asarray(expr).flatten()
            scores.append(v)
        pscore = np.mean(np.column_stack(scores), axis=1).mean()
        rows.append({"patient": pid, "n_cells": pcells.n_obs,
                     "MPRtype": pcells.obs["MPRtype"].iloc[0],
                     f"{pw_name}": pscore})
    return pd.DataFrame(rows)


def run_all(adata):
    # === Filter: keep only nPR and pCR ===
    adata = adata[adata.obs["MPRtype"].isin(["nPR", "MPR_and_pCR"])].copy()
    # Standardize pCR label
    adata.obs["MPRtype"] = adata.obs["MPRtype"].replace({"MPR_and_pCR": "pCR"})
    print(f"Filtered (nPR + pCR): {adata.n_obs:,} cells, {adata.obs['sampleID'].nunique()} patients")
    print(f"  nPR: {(adata.obs['MPRtype']=='nPR').sum():,} cells")
    print(f"  pCR: {(adata.obs['MPRtype']=='pCR').sum():,} cells")

    # ==========================================
    # 1. DE genes — pseudobulk
    # ==========================================
    print("\n[1/5] Pseudobulk DE genes...")
    key_genes = [
        "ANGPTL4","OLFML3","FCGBP","CXCR2","ANXA1",
        "ITGB8","MKI67","DNASE1L3","FABP4","PHLDA3","STAC","S100A8",
        "S100A9","S100A12","CD177","PROK2","ALPL",
        "HIF1A","SPI1","STAT1","IRF1","PPARG","CEBPB",
    ]
    genes_avail = [g for g in key_genes if g in adata.var_names]
    print(f"  {len(genes_avail)}/{len(key_genes)} genes found in var_names")

    pb = pseudobulk(adata, genes_avail)
    de_results = []
    for g in genes_avail:
        r = compare_groups(pb, g, "MPRtype", "nPR", "pCR")
        if r:
            de_results.append(r)
    de_df = pd.DataFrame(de_results)
    _, fdr, _, _ = multipletests(de_df["P"].values, method="fdr_bh")
    de_df["FDR"] = fdr
    de_df = de_df.sort_values("FDR")
    de_df.to_csv(OUTPUT_DIR / "data" / "pseudobulk_DE_results.csv", index=False)
    n_sig = de_df["FDR"].lt(0.05).sum()
    print(f"  {n_sig}/{len(de_df)} genes FDR<0.05")

    # ==========================================
    # 2. Pathway scores — pseudobulk
    # ==========================================
    print("\n[2/5] Pseudobulk pathway scores...")
    pathways = {
        "Iron_Redox": ["HMOX1","FTL","FTH1","TF","TFRC","SLC40A1","CP","HEPH","STEAP3","NCOA4","ACO1","IREB2"],
        "Lipid_PPAR": ["FABP4","PPARG","CD36","LPL","FABP5","PLIN2","ADIPOQ","LIPE","ACSL1"],
        "Fatty_Acid": ["ACADM","CPT1A","ACADVL","HADHA","ECI1","EHHADH","ACAA2","ACOX1"],
    }
    pw_results = []
    pw_data = {}  # For figure generation
    for pw_name, pw_genes in pathways.items():
        pb_pw = pseudobulk_pathway(adata, pw_genes, pw_name)
        if pb_pw is None:
            continue
        pw_data[pw_name] = pb_pw
        r = compare_groups(pb_pw, pw_name, "MPRtype", "nPR", "pCR")
        if r:
            r["pathway"] = pw_name
            pw_results.append(r)
            print(f"  {pw_name}: d={r['Cohen_d']:.3f}, 95%CI=[{r['CI_95_low']:.3f},{r['CI_95_high']:.3f}], P={r['P']:.4f}")
    pw_df = pd.DataFrame(pw_results)
    pw_df.to_csv(OUTPUT_DIR / "data" / "pseudobulk_pathway_results.csv", index=False)

    # ==========================================
    # 3. Signature scores — pseudobulk
    # ==========================================
    print("\n[3/5] Pseudobulk macrophage signatures...")
    npr_sig = ["ANGPTL4","OLFML3","FCGBP","CXCR2","ANXA1"]
    pcr_sig = ["ITGB8","MKI67","DNASE1L3","FABP4","PHLDA3","STAC","S100A8"]
    pb_sig = pseudobulk(adata, npr_sig + pcr_sig)
    pb_sig["npr_score"] = pb_sig[[g for g in npr_sig if g in pb_sig.columns]].mean(axis=1)
    pb_sig["pcr_score"] = pb_sig[[g for g in pcr_sig if g in pb_sig.columns]].mean(axis=1)
    pb_sig["mac_ratio"] = pb_sig["npr_score"] - pb_sig["pcr_score"]
    for sc in ["npr_score","pcr_score","mac_ratio"]:
        r = compare_groups(pb_sig, sc, "MPRtype", "nPR", "pCR")
        if r:
            print(f"  {sc}: d={r['Cohen_d']:.3f}, P={r['P']:.4f}")
    pb_sig.to_csv(OUTPUT_DIR / "data" / "pseudobulk_signature_scores.csv", index=False)

    # ==========================================
    # 4. TF regulon (if present in .obs)
    # ==========================================
    print("\n[4/5] Pseudobulk TF regulon (if available)...")
    tf_cols = [c for c in adata.obs.columns if "regulon" in c.lower() or c.endswith("_TF")]
    if tf_cols:
        print(f"  Found {len(tf_cols)} TF columns: {tf_cols[:5]}...")
        pb_tf = pseudobulk(adata, [])
        for tf in tf_cols:
            means = adata.obs.groupby("sampleID")[tf].mean()
            pb_tf[tf] = pb_tf["patient"].map(means)
        tf_results = []
        for tf in tf_cols:
            r = compare_groups(pb_tf, tf, "MPRtype", "nPR", "pCR")
            if r:
                r["TF"] = tf
                tf_results.append(r)
        if tf_results:
            tf_df = pd.DataFrame(tf_results)
            tf_df.to_csv(OUTPUT_DIR / "data" / "pseudobulk_TF_results.csv", index=False)
            print(f"  {tf_df['P'].lt(0.05).sum()}/{len(tf_df)} TFs P<0.05")
    else:
        print("  No TF regulon columns in .obs — skipping")

    # ==========================================
    # 5. Generate figures
    # ==========================================
    print("\n[5/5] Generating figures...")

    # Fig SXa: DE volcano
    fig, ax = plt.subplots(figsize=(8, 6))
    de_plot = de_df.dropna(subset=["Cohen_d","FDR"]).copy()
    de_plot["nlogFDR"] = -np.log10(de_plot["FDR"].clip(lower=1e-50))
    ax.scatter(de_plot["Cohen_d"], de_plot["nlogFDR"], c="gray", alpha=0.5, s=40)
    sig = de_plot[de_plot["FDR"] < 0.05]
    colors = [NPR_COLOR if d > 0 else PCR_COLOR for d in sig["Cohen_d"]]
    ax.scatter(sig["Cohen_d"], sig["nlogFDR"], c=colors, s=80, alpha=0.9, edgecolors="black", lw=0.5)
    for _, row in sig.iterrows():
        ax.annotate(row["value"], (row["Cohen_d"], row["nlogFDR"]),
                    fontsize=7, xytext=(5, 4), textcoords="offset points", alpha=0.85)
    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=1)
    ax.set_xlabel("Cohen's d (nPR − pCR)", fontweight="bold")
    ax.set_ylabel("−log10(FDR)", fontweight="bold")
    ax.set_title("Patient-Level Pseudobulk DE (Myeloid)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "FigSXa_pseudobulk_DE_volcano.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figures" / "FigSXa_pseudobulk_DE_volcano.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: FigSXa")

    # Fig SXb: Signature boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sc_name in zip(axes, ["npr_score","pcr_score","mac_ratio"]):
        d_npr = pb_sig[pb_sig["MPRtype"]=="nPR"][sc_name].dropna()
        d_pcr = pb_sig[pb_sig["MPRtype"]=="pCR"][sc_name].dropna()
        bp = ax.boxplot([d_npr, d_pcr], labels=["nPR","pCR"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(NPR_COLOR)
        bp["boxes"][1].set_facecolor(PCR_COLOR)
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        for i, (d, c) in enumerate([(d_npr, NPR_COLOR), (d_pcr, PCR_COLOR)]):
            x = np.random.normal(i+1, 0.04, len(d))
            ax.scatter(x, d, alpha=0.5, s=40, color=c, edgecolors="black", lw=0.3)
        _, pv = mannwhitneyu(d_npr, d_pcr)
        sig_str = "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
        ax.set_title(f"{sc_name}  P={pv:.4f} {sig_str}", fontsize=10)
        ax.set_ylabel("Mean Score per Patient")
    fig.suptitle("Patient-Level Signature Scores", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "FigSXb_pseudobulk_signature_boxplots.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figures" / "FigSXb_pseudobulk_signature_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: FigSXb")

    # Fig SXc: Pathway boxplots
    n_pw = len(pw_data)
    if n_pw > 0:
        fig, axes = plt.subplots(1, n_pw, figsize=(5*n_pw, 5))
        if n_pw == 1:
            axes = [axes]
        for ax, (pw_name, pb_pw) in zip(axes, pw_data.items()):
            d_npr = pb_pw[pb_pw["MPRtype"]=="nPR"][pw_name].dropna()
            d_pcr = pb_pw[pb_pw["MPRtype"]=="pCR"][pw_name].dropna()
            bp = ax.boxplot([d_npr, d_pcr], labels=["nPR","pCR"],
                            patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor(NPR_COLOR)
            bp["boxes"][1].set_facecolor(PCR_COLOR)
            for box in bp["boxes"]:
                box.set_alpha(0.7)
            for i, (d, c) in enumerate([(d_npr, NPR_COLOR), (d_pcr, PCR_COLOR)]):
                x = np.random.normal(i+1, 0.04, len(d))
                ax.scatter(x, d, alpha=0.5, s=40, color=c, edgecolors="black", lw=0.3)
            r = pw_df[pw_df["pathway"]==pw_name].iloc[0]
            ax.set_title(f"{pw_name}\nP={r['P']:.4f}, d={r['Cohen_d']:.3f}", fontsize=10)
            ax.set_ylabel("Mean per Patient")
        fig.suptitle("Patient-Level Metabolic Pathway Scores", fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figures" / "FigSXc_pseudobulk_pathway_boxplots.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "figures" / "FigSXc_pseudobulk_pathway_boxplots.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: FigSXc")

    # Summary
    print(f"""
{'='*60}
PSEUDOBULK ANALYSIS DONE
  Patients: {pb["patient"].nunique()}
  DE genes FDR<0.05: {n_sig}/{len(de_df)}
  Output: {OUTPUT_DIR}/
{'='*60}
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--myeloid-key", type=str, default=None,
                       help="Value in cell_type / major_cell_type for myeloid cells "
                            "(e.g. 'Myeloid cell', 'myeloid'). If omitted, all cells used.")
    parser.add_argument("--myeloid-col", type=str, default=None,
                       help="Column name for cell type (auto-detect if omitted).")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # --- Filter to myeloid cells if needed ---
    if args.myeloid_key:
        col = args.myeloid_col
        if col is None:
            for c in ["major_cell_type", "cell_type", "celltype"]:
                if c in adata.obs.columns:
                    col = c
                    break
        if col and col in adata.obs.columns:
            mask = adata.obs[col].astype(str) == args.myeloid_key
            adata = adata[mask].copy()
            print(f"  Filtered to '{args.myeloid_key}' via {col}: {adata.n_obs:,} cells")
        else:
            print(f"  WARNING: column for myeloid filter not found, using all cells")

    print(f"  sampleID: {adata.obs['sampleID'].nunique()} patients")
    print(f"  MPRtype values: {sorted(adata.obs['MPRtype'].dropna().unique())}")
    run_all(adata)


if __name__ == "__main__":
    main()
