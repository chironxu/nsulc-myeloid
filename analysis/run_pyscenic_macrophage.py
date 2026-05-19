#!/usr/bin/env python3
"""
TF Regulon Analysis — Master Regulators of Macrophage Polarization
===================================================================
Self-contained: uses literature-curated TF regulons for macrophage polarization.
Computes per-cell TF activity as mean expression of target genes.

V2 fix: reads FULL h5ad (all genes) + myeloid h5ad (cell-type labels),
merges by barcode to get full-gene expression for macrophage cells.

Input:  processed_data/GSE243013_processed_FULL.h5ad (raw.X = all genes)
        myeloid_analysisnew0.5/myeloid_analyzed.h5ad (cell_type labels)
Output: Differential TF activity, figures, TF-target network

Dependencies: scanpy, pandas, numpy, matplotlib, seaborn, scipy
  pip install scanpy pandas numpy matplotlib seaborn scipy
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.sparse import issparse
import json, warnings
warnings.filterwarnings('ignore')

# ======================================
# Configuration
# ======================================
# Full h5ad — has ALL genes in .raw.X (log-normalized)
FULL_H5AD = "processed_data/GSE243013_processed_FULL.h5ad"
# Myeloid h5ad — has cell_type annotations
MYELOID_H5AD = "myeloid_visualization/myeloid_annotated.h5ad"
OUTPUT_DIR = Path("pyscenic_results")
OUTPUT_DIR.mkdir(exist_ok=True)

NPR_MACROPHAGES = ['ANGPTL4+ TAM']
PCR_MACROPHAGES = ['FABP4+ Macrophage', 'MKI67+ Macrophage']
ALL_MACROPHAGES = NPR_MACROPHAGES + PCR_MACROPHAGES + [
    'SELENOP+ Macrophage', 'CHI3L1+ TAM', 'LINC02432+ TAM',
    'MT1M+ Macrophage', 'SFTPB+ Macrophage'
]

# ======================================
# Literature-curated TF regulons for macrophage polarization
# Targets from: DoRothEA (A+B levels), TRRUST, MSigDB, macrophage ChIP-seq
# ======================================
TF_REGULONS = {
    # —— M1 / pro-inflammatory / anti-tumor TFs (expected ↑ in pCR) ——
    'STAT1': ['CXCL10', 'CXCL11', 'IRF1', 'GBP1', 'GBP2', 'TAP1', 'PSMB8', 'PSMB9',
              'CIITA', 'B2M', 'ICAM1', 'FCGR1A', 'SOCS1', 'ISG15', 'IFIT1', 'IFIT3',
              'IFIH1', 'OAS1', 'MX1', 'RSAD2'],
    'IRF1': ['STAT1', 'CXCL10', 'GBP2', 'PSMB8', 'TAP1', 'ISG15', 'IRF7', 'DDX58',
             'OAS1', 'CCL5', 'CXCL9', 'CASP1', 'TNFSF10', 'CIITA', 'HLA-A', 'HLA-B'],
    'IRF5': ['IL6', 'IL12B', 'TNF', 'CCL3', 'CCL4', 'CXCL10', 'IFNB1', 'IL23A'],
    'NFKB1': ['TNFA', 'IL1B', 'IL6', 'IL8', 'CCL2', 'CCL5', 'CXCL1', 'CXCL2',
              'ICAM1', 'VCAM1', 'SELE', 'MMP9', 'BCL2L1', 'XIAP', 'NFKBIA', 'BIRC3'],
    'RELA': ['TNFA', 'IL1B', 'IL6', 'IL8', 'CCL2', 'CCL5', 'ICAM1', 'VCAM1',
             'MMP9', 'BCL2L1', 'XIAP', 'NFKBIA', 'BIRC3', 'CXCL8', 'SOD2'],
    'IRF8': ['CSF1R', 'CD14', 'ITGAM', 'FCGR1A', 'TLR4', 'IL12B', 'NOS2', 'CIITA'],
    'HIF1A': ['VEGFA', 'LDHA', 'HK2', 'SLC2A1', 'CA9', 'BNIP3', 'PGK1', 'ENO1',
              'ADM', 'EDN1', 'TFRC', 'SERPINE1', 'ANGPT2'],
    'SPI1': ['CSF1R', 'CD14', 'CD68', 'FCGR1A', 'ITGAM', 'TLR4', 'CCR2', 'IRF8',
             'MAFB', 'TYROBP', 'FCER1G', 'C1QA', 'C1QB', 'CTSS', 'CD74'],

    # —— M2 / anti-inflammatory / pro-tumor TFs (expected ↑ in nPR) ——
    'STAT3': ['SOCS3', 'MYC', 'CCND1', 'VEGFA', 'IL10', 'TGFB1', 'HIF1A', 'MMP9',
              'BIRC5', 'FGF2', 'BCL2', 'CCL2', 'CXCR4', 'MCL1', 'JUNB'],
    'STAT6': ['CCL17', 'CCL22', 'CCL24', 'ALOX15', 'SOCS1', 'FCER2', 'IL13RA1',
              'ARG1', 'MRC1', 'IRF4', 'GATA3', 'IL4R'],
    'IRF4': ['BCL6', 'PRDM1', 'SDC1', 'AICDA', 'XBP1', 'IL4', 'MYB', 'MYC'],
    'PPARG': ['FABP4', 'CD36', 'LPL', 'FABP5', 'ACSL1', 'PPARA', 'CPT1A',
              'ADIPOQ', 'LEP', 'RXRA', 'ANGPTL4', 'FABP1', 'SCD'],
    'CEBPB': ['IL6', 'CCL2', 'CCL20', 'SAA1', 'HP', 'SERPINA1', 'LCN2',
              'S100A8', 'S100A9', 'CSF3', 'CXCL8', 'NFIL3'],
    'KLF4': ['CDH1', 'CLDN3', 'CLDN4', 'MUC1', 'KRT19', 'TJP1', 'P53', 'CCND1',
             'IL10', 'FGF2'],
    'MAF': ['IL4', 'IL10', 'CCL2', 'CD14', 'FABP4', 'TGFB1', 'CTLA4'],
    'JUN': ['MMP1', 'MMP3', 'MMP9', 'IL2', 'IL6', 'IFNG', 'CCL2', 'FOSL1',
            'VEGFA', 'CCND1', 'TNF', 'FOS', 'EGR1'],
    'FOS': ['JUN', 'JUNB', 'FOSL1', 'MMP1', 'MMP3', 'IL2', 'CCL2', 'EGR1', 'ATF3'],
    'EGR1': ['TNF', 'IL2', 'FOS', 'JUNB', 'PTGS2', 'CCL2', 'MMP9', 'PDGFA'],

    # —— Metabolism-linked TFs ——
    'MYC': ['CCND2', 'CDK4', 'E2F1', 'E2F2', 'NCL', 'ODC1', 'LDHA', 'CAD',
            'HK2', 'PKM', 'GLUT1', 'RPL3', 'RPS6', 'NPM1'],
    'NRF2': ['HMOX1', 'NQO1', 'GCLC', 'GCLM', 'TXNRD1', 'PRDX1', 'SOD1',
             'GPX2', 'FTL', 'FTH1', 'SRXN1', 'GSTM1'],
    'TFEB': ['ATP6V0D1', 'LAMP1', 'SQSTM1', 'CTSD', 'CTSB', 'PPARGC1A',
             'LC3B', 'BECN1', 'UVRAG', 'RAB7A'],
    'TFEC': ['CSF1R', 'F4/80', 'CD68', 'GPNMB', 'MITF', 'TFE3', 'TYROBP'],

    # —— Macrophage differentiation master regulators ——
    'RUNX1': ['CSF1R', 'CD14', 'ITGAM', 'MPO', 'CEBPA', 'SPI1', 'FCGR3A',
              'CD36', 'IL3RA', 'ANPEP'],
    'CEBPA': ['CSF3R', 'GCSFR', 'MPO', 'ELANE', 'CEBPE', 'SPI1', 'CD14',
              'ITGAM', 'CCL2', 'IL6'],
}

print("=" * 70)
print("TF Regulon Analysis — Macrophage Master Regulators (V2)")
print("=" * 70)
print(f"  Pre-loaded regulons: {len(TF_REGULONS)} TFs")

# ======================================
# Step 1: Load data
# ======================================
print("\n[Step 1] Loading data...")

# Load myeloid h5ad to get cell-type labels
print("  Loading myeloid annotations...")
adata_mye = sc.read(MYELOID_H5AD)
print(f"  Myeloid cells: {adata_mye.n_obs:,}")

# Get macrophage barcodes
macro_mask = adata_mye.obs['cell_type'].isin(ALL_MACROPHAGES)
macro_barcodes = set(adata_mye[macro_mask].obs_names)
print(f"  Macrophage barcodes: {len(macro_barcodes)}")

# Load full h5ad for ALL-gene expression
print("  Loading full expression data (this may take a minute)...")
adata_full = sc.read(FULL_H5AD)
print(f"  Full cells: {adata_full.n_obs:,}, genes: {adata_full.n_vars:,}")

# Use raw.X (log-normalized, all genes — ~31k)
if adata_full.raw is not None:
    n_raw_genes = adata_full.raw.n_vars
    print(f"  raw.X genes: {n_raw_genes:,}")
    expr_full = adata_full.raw.X
    gene_names_full = adata_full.raw.var_names
else:
    expr_full = adata_full.X
    gene_names_full = adata_full.var_names
    print("  WARNING: no .raw found, using .X")

# Match macrophage cells
full_barcodes = set(adata_full.obs_names)
matched_barcodes = list(macro_barcodes & full_barcodes)
print(f"  Matched macrophage cells: {len(matched_barcodes):,}")

if len(matched_barcodes) < 100:
    print("  ERROR: Too few matched cells! Check barcode format.")
    print(f"  Sample myeloid barcode: {list(macro_barcodes)[:3]}")
    print(f"  Sample full barcode: {list(full_barcodes)[:3]}")
    import sys
    sys.exit(1)

# Extract macrophage expression
macro_idx = [list(adata_full.obs_names).index(b) for b in matched_barcodes]
if issparse(expr_full):
    expr_macro = expr_full[macro_idx, :].toarray()
else:
    expr_macro = expr_full[macro_idx, :]

# Build obs dataframe with myeloid annotations
mye_barcode_to_ctype = dict(zip(adata_mye.obs_names, adata_mye.obs['cell_type']))
cell_types = [mye_barcode_to_ctype.get(b, 'Unknown') for b in matched_barcodes]
obs_df = pd.DataFrame({'cell_type': cell_types}, index=matched_barcodes)

# Transfer UMAP if available
if 'X_umap' in adata_mye.obsm:
    mye_umap = dict(zip(adata_mye.obs_names, adata_mye.obsm['X_umap']))
    umap_coords = np.array([mye_umap.get(b, [np.nan, np.nan]) for b in matched_barcodes])
    has_umap = not np.isnan(umap_coords).any()
    if has_umap:
        umap_arr = umap_coords
else:
    has_umap = False

print(f"  Final matrix: {expr_macro.shape[0]:,} cells x {expr_macro.shape[1]:,} genes")

# ======================================
# Step 2: Compute TF activity scores
# ======================================
print("\n[Step 2] Computing TF regulon activity scores...")

# z-score expression per gene for robust scoring
expr_mean = expr_macro.mean(axis=0)
expr_std = expr_macro.std(axis=0)
expr_std[expr_std == 0] = 1.0
expr_z = (expr_macro - expr_mean) / expr_std

gene_list = gene_names_full.tolist()

tf_scores = {}
tf_valid = {}
for tf_name, target_genes in TF_REGULONS.items():
    available = [g for g in target_genes if g in gene_list]
    if len(available) >= 4:
        idx = [gene_list.index(g) for g in available]
        tf_score = expr_z[:, idx].mean(axis=1)
        tf_scores[tf_name] = tf_score
        tf_valid[tf_name] = available
        obs_df[f'TF_{tf_name}'] = tf_score
        print(f"  {tf_name}: {len(available)}/{len(target_genes)} targets")
    else:
        print(f"  {tf_name}: SKIP — only {len(available)}/{len(target_genes)} targets found")

print(f"  Total TFs scored: {len(tf_scores)}")

# ======================================
# Step 3: Differential TF activity — nPR vs pCR
# ======================================
print("\n[Step 3] Differential TF activity analysis...")

obs_df['macro_group'] = 'Other'
obs_df.loc[obs_df['cell_type'].isin(NPR_MACROPHAGES), 'macro_group'] = 'nPR'
obs_df.loc[obs_df['cell_type'].isin(PCR_MACROPHAGES), 'macro_group'] = 'pCR'

npr_mask = obs_df['macro_group'] == 'nPR'
pcr_mask = obs_df['macro_group'] == 'pCR'
print(f"  nPR cells: {npr_mask.sum()}, pCR cells: {pcr_mask.sum()}")

tf_cols = [c for c in obs_df.columns if c.startswith('TF_')]
print(f"  Testing {len(tf_cols)} TFs")

tf_stats = []
for tf_col in tf_cols:
    tf_name = tf_col.replace('TF_', '')
    npr_vals = obs_df.loc[npr_mask, tf_col].dropna().values
    pcr_vals = obs_df.loc[pcr_mask, tf_col].dropna().values
    if len(npr_vals) > 3 and len(pcr_vals) > 3:
        stat, pval = mannwhitneyu(npr_vals, pcr_vals, alternative='two-sided')
        delta = np.mean(npr_vals) - np.mean(pcr_vals)
        tf_stats.append({
            'TF': tf_name,
            'mean_nPR': np.mean(npr_vals),
            'mean_pCR': np.mean(pcr_vals),
            'delta': delta,
            'p_value': pval
        })

tf_stats_df = pd.DataFrame(tf_stats).sort_values('p_value')
tf_stats_df['neg_log10_p'] = -np.log10(tf_stats_df['p_value'].clip(lower=1e-300))
tf_stats_df.to_csv(OUTPUT_DIR / 'tf_differential_activity.csv', index=False)

print(f"  Top TFs (higher in nPR — immunosuppressive):")
print(tf_stats_df.sort_values('delta', ascending=False).head(10)[['TF', 'delta', 'p_value']].to_string(index=False))
print(f"\n  Top TFs (higher in pCR — immune-supportive):")
print(tf_stats_df.sort_values('delta', ascending=True).head(10)[['TF', 'delta', 'p_value']].to_string(index=False))

# Per-cell-type TF activity
cell_type_tf = obs_df.groupby('cell_type')[tf_cols].mean()
cell_type_tf.to_csv(OUTPUT_DIR / 'tf_activity_by_celltype.csv')
print(f"\n  Saved: tf_activity_by_celltype.csv")

# ======================================
# Step 4: Generate figures
# ======================================
print("\n[Step 4] Generating figures...")

# ---- Fig TF-1: Top differential TFs (bar chart) ----
top_up = tf_stats_df.nsmallest(15, 'p_value').nlargest(6, 'delta')
top_down = tf_stats_df.nsmallest(15, 'p_value').nsmallest(6, 'delta')
top_tfs = pd.concat([top_up, top_down]).drop_duplicates()

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#3498db' if d < 0 else '#e74c3c' for d in top_tfs['delta']]
ax.barh(range(len(top_tfs)), top_tfs['delta'].values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top_tfs)))
labels = [f"{t['TF']} (P={t['p_value']:.1e})" for _, t in top_tfs.iterrows()]
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('TF Activity Difference (nPR - pCR)', fontsize=13)
ax.set_title('Differential TF Regulon Activity\nnPR vs pCR Macrophages (full gene set)', fontsize=14)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#e74c3c', label='Higher in nPR (Immunosuppressive)'),
    Patch(color='#3498db', label='Higher in pCR (Immune-supportive)')
], fontsize=10, loc='lower right')
ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_TF_differential_activity.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig_TF_differential_activity.png")

# ---- Fig TF-2: Volcano plot ----
fig, ax = plt.subplots(figsize=(10, 8))
colors_v = ['#e74c3c' if d > 0 else '#3498db' for d in tf_stats_df['delta']]
ax.scatter(tf_stats_df['delta'], tf_stats_df['neg_log10_p'],
           c=colors_v, alpha=0.7, s=120, edgecolors='black', linewidth=0.2)
for _, row in tf_stats_df.nsmallest(10, 'p_value').iterrows():
    ax.annotate(row['TF'], (row['delta'], row['neg_log10_p']),
                fontsize=9, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points')
ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='P=0.05')
ax.axvline(x=0, color='black', linestyle='-', alpha=0.2)
ax.set_xlabel('TF Activity Difference (nPR - pCR)', fontsize=13)
ax.set_ylabel('-log10(P-value)', fontsize=13)
ax.set_title('TF Regulon Volcano Plot\nnPR vs pCR Macrophages (full gene set)', fontsize=14)
ax.legend(handles=[
    Patch(color='#e74c3c', label='nPR-enriched'),
    Patch(color='#3498db', label='pCR-enriched')
], fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_TF_volcano.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig_TF_volcano.png")

# ---- Fig TF-3: UMAP (top 4 TFs) ----
top4_tfs = tf_stats_df.nsmallest(4, 'p_value')['TF'].tolist()
if has_umap:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for i, (tf, ax) in enumerate(zip(top4_tfs, axes.flatten())):
        sc_vals = obs_df[f'TF_{tf}'].values
        vmin, vmax = sc_vals.min(), sc_vals.max()
        vabs = max(abs(vmin), abs(vmax))
        scat = ax.scatter(umap_arr[:, 0], umap_arr[:, 1],
                         c=sc_vals, cmap='RdBu_r', s=2, alpha=0.6,
                         vmin=-vabs, vmax=vabs)
        plt.colorbar(scat, ax=ax, label='TF Activity')
        pv = tf_stats_df[tf_stats_df['TF'] == tf]['p_value'].values[0]
        ax.set_title(f'{tf} Activity\n(P={pv:.2e})', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Master Regulator TF Activities in Macrophages',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig_TF_umap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_TF_umap.png")
else:
    print("  WARNING: No UMAP — skipping UMAP figure")

# ---- Fig TF-4: Heatmap across macrophage subtypes ----
macro_types = obs_df['cell_type'].unique()
if len(macro_types) >= 3 and len(tf_cols) >= 5:
    tf_mean = obs_df.groupby('cell_type')[tf_cols].mean()
    tf_var = tf_mean.var().sort_values(ascending=False)
    top_var_tfs = tf_var.head(15).index.tolist()
    tf_z = (tf_mean[top_var_tfs] - tf_mean[top_var_tfs].mean()) / tf_mean[top_var_tfs].std()
    tf_z.columns = [c.replace('TF_', '') for c in tf_z.columns]

    fig, ax = plt.subplots(figsize=(14, max(4, len(macro_types) * 0.7 + 2)))
    sns.heatmap(tf_z, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Z-score'}, ax=ax, linewidths=0.5)
    ax.set_title('TF Regulon Activity Across Macrophage Subsets\n(Z-scored across cell types, full gene set)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Transcription Factor', fontsize=12)
    ax.set_ylabel('Macrophage Subset', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig_TF_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_TF_heatmap.png")

# ======================================
# Step 5: TF-target network
# ======================================
print("\n[Step 5] Building TF-target regulatory network...")

top5_tfs = tf_stats_df.nsmallest(5, 'p_value')['TF'].tolist()
edges = []
for tf in top5_tfs:
    if tf in tf_valid:
        for target in tf_valid[tf]:
            edges.append({'TF': tf, 'Target': target, 'Regulon': 'activation'})

if edges:
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(OUTPUT_DIR / 'tf_target_network.csv', index=False)
    print(f"  Network: {len(edges_df)} edges for top 5 TFs")
    print("  Saved: tf_target_network.csv (importable into Cytoscape/igraph)")

# Save ALL regulons
full_regulons_file = OUTPUT_DIR / 'full_tf_regulons.csv'
with open(full_regulons_file, 'w') as f:
    f.write("TF,Target\n")
    for tf, targets in tf_valid.items():
        for t in targets:
            f.write(f"{tf},{t}\n")
print(f"  Saved: full_tf_regulons.csv")

# ======================================
# Step 6: Save results
# ======================================
print("\n[Step 6] Saving results...")

obs_df.to_csv(OUTPUT_DIR / 'macrophage_tf_activity.csv')
print(f"  Saved: macrophage_tf_activity.csv ({len(obs_df)} cells)")

n_sig = len(tf_stats_df[tf_stats_df['p_value'] < 0.05])
summary = {
    'n_macrophages': int(len(obs_df)),
    'n_cell_types': int(obs_df['cell_type'].nunique()),
    'npr_macrophages': NPR_MACROPHAGES,
    'pcr_macrophages': PCR_MACROPHAGES,
    'n_TFs_tested': len(tf_cols),
    'n_TFs_significant': int(n_sig),
    'top_npr_TFs': tf_stats_df.nlargest(5, 'delta')['TF'].tolist(),
    'top_pcr_TFs': tf_stats_df.nsmallest(5, 'delta')['TF'].tolist(),
}
with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print("TF REGULON ANALYSIS COMPLETE (V2 — full gene set)")
print(f"{'='*70}")
print(f"Output directory: {OUTPUT_DIR}/")
print(f"Summary:")
print(f"  Significant differential TFs (P<0.05): {n_sig}/{len(tf_cols)}")
print(f"  Top nPR-driving TFs: {summary['top_npr_TFs']}")
print(f"  Top pCR-driving TFs: {summary['top_pcr_TFs']}")
print(f"\n  Output files:")
for fname in sorted(OUTPUT_DIR.glob('*')):
    if fname.is_file():
        print(f"    {fname.name}")
