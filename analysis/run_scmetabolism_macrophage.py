#!/usr/bin/env python3
"""
Metabolic Reprogramming Analysis — nPR vs pCR Macrophage Metabolism
====================================================================
Self-contained: uses pre-defined KEGG metabolic gene sets.
Computes per-cell pathway activity scores from scRNA-seq expression.

V2 fix: reads FULL h5ad (all genes) + myeloid h5ad (cell-type labels),
merges by barcode to get full-gene expression for macrophage cells.

Input:  processed_data/GSE243013_processed_FULL.h5ad (raw.X = all genes)
        myeloid_analysisnew0.5/myeloid_analyzed.h5ad (cell_type labels)
Output: Differential metabolic pathway scores, figures

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
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

# ======================================
# Configuration
# ======================================
# Full h5ad — has ALL genes in .raw.X (log-normalized)
FULL_H5AD = "processed_data/GSE243013_processed_FULL.h5ad"
# Myeloid h5ad — has cell_type annotations
MYELOID_H5AD = "myeloid_visualization/myeloid_annotated.h5ad"
OUTPUT_DIR = Path("scmetabolism_results")
OUTPUT_DIR.mkdir(exist_ok=True)

NPR_MACROPHAGES = ['ANGPTL4+ TAM']
PCR_MACROPHAGES = ['FABP4+ Macrophage', 'MKI67+ Macrophage']
ALL_MACROPHAGES = NPR_MACROPHAGES + PCR_MACROPHAGES + [
    'SELENOP+ Macrophage', 'CHI3L1+ TAM', 'LINC02432+ TAM',
    'MT1M+ Macrophage', 'SFTPB+ Macrophage'
]

# Key metabolic pathways (KEGG)
METABOLIC_PATHWAYS = {
    'Glycolysis / Gluconeogenesis': ['HK1', 'HK2', 'HK3', 'GPI', 'PFKL', 'PFKM', 'PFKP',
                                      'ALDOA', 'GAPDH', 'PGK1', 'PKM', 'LDHA', 'LDHB',
                                      'ENO1', 'ENO2', 'TPI1'],
    'Oxidative Phosphorylation': ['NDUFV1', 'NDUFS1', 'SDHA', 'SDHB', 'UQCRC1', 'UQCRC2',
                                   'COX5A', 'COX6B1', 'ATP5A1', 'ATP5B', 'ATP5F1'],
    'Fatty Acid Metabolism': ['CPT1A', 'CPT2', 'ACADM', 'ACADVL', 'HADHA', 'HADHB',
                               'ACSL1', 'ACSL4', 'FABP1', 'FABP4', 'FABP5', 'CD36'],
    'TCA Cycle': ['CS', 'ACO1', 'ACO2', 'IDH1', 'IDH2', 'IDH3A', 'OGDH', 'SUCLA2',
                   'SDHA', 'SDHB', 'FH', 'MDH1', 'MDH2'],
    'Glutamine Metabolism': ['GLS', 'GLS2', 'GLUD1', 'GLUD2', 'GPT', 'GPT2',
                              'GOT1', 'GOT2', 'ASNS', 'CAD', 'CPS1'],
    'Arginine Metabolism': ['ARG1', 'ARG2', 'NOS1', 'NOS2', 'NOS3', 'ASS1',
                             'ASL', 'ODC1', 'SRM', 'SMS'],
    'Lipid Metabolism (PPAR)': ['PPARG', 'PPARA', 'PPARD', 'LPL', 'FABP4',
                                 'FABP5', 'CD36', 'SCD', 'FASN', 'ACACA'],
    'One Carbon / Folate': ['MTHFR', 'MTR', 'SHMT1', 'SHMT2', 'MTHFD1',
                             'TYMS', 'DHFR', 'ATIC', 'GART'],
    'Cholesterol Biosynthesis': ['HMGCR', 'HMGCS1', 'SQLE', 'FDFT1', 'LSS',
                                  'CYP51A1', 'MSMO1', 'NSDHL', 'DHCR7', 'DHCR24'],
    'Iron / Redox Metabolism': ['FTH1', 'FTL', 'TFRC', 'SLC40A1', 'HMOX1',
                                 'SOD1', 'SOD2', 'CAT', 'GPX1', 'GPX4', 'PRDX1',
                                 'TXN', 'TXNRD1', 'MT1M', 'MT2A', 'MT1A'],
}

print("=" * 70)
print("Metabolic Reprogramming Analysis — Macrophage States (V2)")
print("=" * 70)

# ======================================
# Step 1: Load data
# ======================================
print("\n[Step 1] Loading data...")

# Load myeloid h5ad to get cell-type labels
print("  Loading myeloid annotations...")
adata_mye = sc.read(MYELOID_H5AD)
print(f"  Myeloid cells: {adata_mye.n_obs:,}")
print(f"  Cell types: {adata_mye.obs['cell_type'].value_counts().to_dict()}")

# Get macrophage barcodes
macro_mask = adata_mye.obs['cell_type'].isin(ALL_MACROPHAGES)
macro_barcodes = set(adata_mye[macro_mask].obs_names)
print(f"  Macrophage barcodes: {len(macro_barcodes)}")

# Load full h5ad for ALL-gene expression
print("  Loading full expression data (this may take a minute)...")
adata_full = sc.read(FULL_H5AD)
print(f"  Full cells: {adata_full.n_obs:,}, genes: {adata_full.n_vars:,}")

# Use raw.X (log-normalized, all genes)
if adata_full.raw is not None:
    n_raw_genes = adata_full.raw.n_vars
    print(f"  raw.X genes: {n_raw_genes:,}")
    # Build expression from raw
    expr_full = adata_full.raw.X
    gene_names_full = adata_full.raw.var_names
else:
    expr_full = adata_full.X
    gene_names_full = adata_full.var_names
    print("  WARNING: no .raw found, using .X")

# Match macrophage cells in full data
full_barcodes = set(adata_full.obs_names)
matched_barcodes = list(macro_barcodes & full_barcodes)
print(f"  Matched macrophage cells: {len(matched_barcodes):,}")

if len(matched_barcodes) < 100:
    print("  ERROR: Too few matched cells! Check barcode format compatibility.")
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

# Build adata_macro with full genes + myeloid annotations
# Match annotation order to expression order
mye_barcode_to_ctype = dict(zip(adata_mye.obs_names, adata_mye.obs['cell_type']))
mye_barcode_to_leiden = {}
for col in adata_mye.obs.columns:
    if col != 'cell_type':
        try:
            mye_barcode_to_leiden[col] = dict(zip(adata_mye.obs_names, adata_mye.obs[col]))
        except:
            pass

cell_types = [mye_barcode_to_ctype.get(b, 'Unknown') for b in matched_barcodes]
obs_df = pd.DataFrame({'cell_type': cell_types}, index=matched_barcodes)

# Add UMAP if available
if 'X_umap' in adata_mye.obsm:
    mye_umap = dict(zip(adata_mye.obs_names, adata_mye.obsm['X_umap']))
    umap_coords = np.array([mye_umap.get(b, [np.nan, np.nan]) for b in matched_barcodes])
    has_umap = not np.isnan(umap_coords).any()
else:
    has_umap = False
    umap_coords = np.zeros((len(matched_barcodes), 2))

print(f"  Final macrophage matrix: {expr_macro.shape[0]:,} cells x {expr_macro.shape[1]:,} genes")

# Check if expression is log-normalized
sample_vals = expr_macro[0, :100]
print(f"  Expression range (sample): [{sample_vals.min():.2f}, {sample_vals.max():.2f}]")

# ======================================
# Step 2: Compute metabolic pathway scores
# ======================================
print("\n[Step 2] Computing metabolic pathway activity scores...")

gene_list = gene_names_full.tolist()

pathway_scores = {}
for pathway_name, genes in METABOLIC_PATHWAYS.items():
    available = [g for g in genes if g in gene_list]
    if len(available) >= 3:
        gene_idx = [gene_list.index(g) for g in available]
        pathway_expr = expr_macro[:, gene_idx].mean(axis=1)
        pathway_scores[pathway_name] = pathway_expr
        safe_name = pathway_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        obs_df[f'Metab_{safe_name}'] = pathway_expr
        print(f"  {pathway_name}: {len(available)}/{len(genes)} genes")
    else:
        print(f"  {pathway_name}: SKIP (only {len(available)}/{len(genes)} genes)")

print(f"  Total pathways scored: {len(pathway_scores)}")

# ======================================
# Step 3: Differential metabolic analysis — nPR vs pCR
# ======================================
print("\n[Step 3] Differential metabolic pathway analysis...")

obs_df['macro_group'] = 'Other'
obs_df.loc[obs_df['cell_type'].isin(NPR_MACROPHAGES), 'macro_group'] = 'nPR'
obs_df.loc[obs_df['cell_type'].isin(PCR_MACROPHAGES), 'macro_group'] = 'pCR'

npr_mask = obs_df['macro_group'] == 'nPR'
pcr_mask = obs_df['macro_group'] == 'pCR'
print(f"  nPR cells: {npr_mask.sum()}, pCR cells: {pcr_mask.sum()}")

metab_cols = [c for c in obs_df.columns if c.startswith('Metab_')]
print(f"  Comparing {len(metab_cols)} metabolic features")

metab_stats = []
for col in metab_cols:
    npr_vals = obs_df.loc[npr_mask, col].dropna()
    pcr_vals = obs_df.loc[pcr_mask, col].dropna()
    if len(npr_vals) > 3 and len(pcr_vals) > 3:
        stat, pval = stats.mannwhitneyu(npr_vals, pcr_vals, alternative='two-sided')
        fc = np.mean(npr_vals) - np.mean(pcr_vals)
        pathway_name = col.replace('Metab_', '').replace('_', ' ')
        metab_stats.append({
            'Pathway': pathway_name,
            'mean_nPR': np.mean(npr_vals),
            'mean_pCR': np.mean(pcr_vals),
            'delta': fc,
            'p_value': pval,
            'neg_log10_p': -np.log10(max(pval, 1e-300))
        })

metab_df = pd.DataFrame(metab_stats).sort_values('p_value')
metab_df.to_csv(OUTPUT_DIR / 'differential_metabolism.csv', index=False)

print(f"\n  Top pathways enriched in nPR (immunosuppressive):")
print(metab_df.nlargest(8, 'delta')[['Pathway', 'delta', 'p_value']].to_string(index=False))
print(f"\n  Top pathways enriched in pCR (immune-supportive):")
print(metab_df.nsmallest(8, 'delta')[['Pathway', 'delta', 'p_value']].to_string(index=False))

# ======================================
# Step 4: Generate figures
# ======================================
print("\n[Step 4] Generating figures...")

# ---- Fig MET-1: Metabolic pathway differential heatmap ----
sig_pathways = metab_df[metab_df['p_value'] < 0.1].head(20)
if len(sig_pathways) < 5:
    sig_pathways = metab_df.head(15)

pathway_cols_orig = [
    f"Metab_{p.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"
    for p in sig_pathways['Pathway']
]
pathway_cols_orig = [c for c in pathway_cols_orig if c in obs_df.columns]

if pathway_cols_orig and obs_df['cell_type'].nunique() >= 2:
    obs_df_copy = obs_df.copy()
    cell_type_means = obs_df_copy.groupby('cell_type')[pathway_cols_orig].mean()
    rename_map = {}
    for c in pathway_cols_orig:
        pretty = c.replace('Metab_', '').replace('_', ' ')
        rename_map[c] = pretty
    cell_type_means = cell_type_means.rename(columns=rename_map)

    cell_type_z = (cell_type_means - cell_type_means.mean()) / cell_type_means.std()

    fig, ax = plt.subplots(figsize=(max(10, len(pathway_cols_orig) * 1.2),
                                    max(5, len(cell_type_z) * 0.6)))
    sns.heatmap(
        cell_type_z, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
        xticklabels=True, yticklabels=True,
        cbar_kws={'label': 'Z-score'}, ax=ax, linewidths=0.5
    )
    ax.set_title('Metabolic Pathway Activity Across Macrophage Subsets\n(Z-scored across cell types, full gene set)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Metabolic Pathway', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig_metabolism_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_metabolism_heatmap.png")

# ---- Fig MET-2: Volcano plot ----
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    metab_df['delta'], metab_df['neg_log10_p'],
    c=['#e74c3c' if d > 0 else '#3498db' for d in metab_df['delta']],
    alpha=0.7, s=120, edgecolors='black', linewidth=0.3
)
for _, row in metab_df.nsmallest(8, 'p_value').iterrows():
    ax.annotate(row['Pathway'], (row['delta'], row['neg_log10_p']),
                fontsize=9, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points')

ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='P=0.05')
ax.axvline(x=0, color='black', linestyle='-', alpha=0.2)
ax.set_xlabel('Metabolic Activity Difference (nPR - pCR)', fontsize=13)
ax.set_ylabel('-log10(P-value)', fontsize=13)
ax.set_title('Differential Metabolic Programs\nnPR vs pCR Macrophages (full gene set)', fontsize=14)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#e74c3c', label='Higher in nPR (Immunosuppressive)'),
    Patch(color='#3498db', label='Higher in pCR (Immune-supportive)')
], fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_metabolism_volcano.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig_metabolism_volcano.png")

# ---- Fig MET-3: Boxplot comparison ----
top_pathways = pd.concat([
    metab_df.nlargest(4, 'delta'),
    metab_df.nsmallest(4, 'delta')
]).drop_duplicates()

if len(top_pathways) > 0:
    n_pathways = len(top_pathways)
    ncols = min(4, n_pathways)
    nrows = (n_pathways + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if n_pathways == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(top_pathways.iterrows()):
        ax = axes[i]
        pathway = row['Pathway']
        col_name = f"Metab_{pathway.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"
        if col_name not in obs_df.columns:
            continue

        groups = ['nPR', 'pCR']
        group_data = {}
        for g in groups:
            mask_g = obs_df['macro_group'] == g
            vals = obs_df.loc[mask_g, col_name].dropna()
            group_data[g] = vals

        positions = [0, 1]
        bp = ax.boxplot([group_data['nPR'], group_data['pCR']],
                       positions=positions, widths=0.4,
                       patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#3498db')

        for pos, data in zip(positions, [group_data['nPR'], group_data['pCR']]):
            if len(data) > 0:
                jitter = np.random.normal(0, 0.03, len(data))
                ax.scatter(np.full(len(data), pos) + jitter, data,
                          alpha=0.3, s=15, c='#2c3e50', edgecolors='none')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['nPR', 'pCR'], fontsize=11)
        ax.set_ylabel('Pathway Score', fontsize=11)
        ax.set_title(f'{pathway}\nP={row["p_value"]:.2e}', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    for i in range(n_pathways, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Metabolic Pathway Scores — nPR vs pCR Macrophages',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig_metabolism_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_metabolism_boxplots.png")

# ---- Fig MET-4: Per-macrophage metabolic profile ----
macro_types = obs_df['cell_type'].unique()
if len(metab_cols) >= 3 and len(macro_types) >= 3:
    type_means = obs_df.groupby('cell_type')[metab_cols].mean()
    rename = {c: c.replace('Metab_', '').replace('_', ' ') for c in metab_cols}
    type_means = type_means.rename(columns=rename)
    type_z = (type_means - type_means.mean()) / type_means.std()

    fig, ax = plt.subplots(figsize=(12, max(5, len(macro_types) * 0.6 + 1)))
    sns.heatmap(
        type_z, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
        ax=ax, linewidths=0.5, cbar_kws={'label': 'Z-score'}
    )
    ax.set_title('Metabolic Profiles of Macrophage Subsets\n(Z-scored across cell types, full gene set)',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Macrophage Subset', fontsize=12)
    ax.set_xlabel('Metabolic Program', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig_metabolism_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_metabolism_profiles.png")

# ======================================
# Step 5: Gene-level metabolic analysis
# ======================================
print("\n[Step 5] Gene-level metabolic markers...")

key_metab_genes = [
    'HK2', 'PKM', 'LDHA',          # Glycolysis
    'CPT1A', 'FABP4', 'FABP5',     # Fatty acid
    'PPARG', 'CD36',                # Lipid/PPAR
    'GLS', 'GLUD1',                 # Glutamine
    'ARG1', 'NOS2',                 # Arginine (M1/M2 marker!)
    'HMOX1', 'FTL', 'MT2A',        # Iron/redox
    'SQLE', 'HMGCR',                # Cholesterol
    'SDHA', 'ATP5B',                # OXPHOS
]

available_metab = [g for g in key_metab_genes if g in gene_list]
print(f"  Available marker genes: {len(available_metab)}/{len(key_metab_genes)}")

if available_metab:
    gene_idx = [gene_list.index(g) for g in available_metab]
    expr_gene = expr_macro[:, gene_idx]
    expr_df = pd.DataFrame(expr_gene, index=matched_barcodes, columns=available_metab)
    expr_df['cell_type'] = obs_df['cell_type'].values
    cell_gene_means = expr_df.groupby('cell_type').mean()
    cell_gene_z = (cell_gene_means - cell_gene_means.mean()) / cell_gene_means.std()

    fig, ax = plt.subplots(figsize=(max(12, len(available_metab) * 0.8),
                                    max(5, len(cell_gene_z) * 0.6)))
    sns.heatmap(
        cell_gene_z, cmap='RdBu_r', center=0,
        xticklabels=True, yticklabels=True,
        cbar_kws={'label': 'Z-score'}, ax=ax, linewidths=0.5
    )
    ax.set_title('Key Metabolic Gene Expression Across Macrophage Subsets',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Gene', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig_metabolism_genes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_metabolism_genes.png")

# ======================================
# Step 6: Save results
# ======================================
print("\n[Step 6] Saving results...")

# Save as simple dataframe CSV (avoid h5ad dependency)
obs_df.to_csv(OUTPUT_DIR / 'macrophage_metabolism_scores.csv')
print(f"  Saved: macrophage_metabolism_scores.csv ({len(obs_df)} cells)")

n_sig = len(metab_df[metab_df['p_value'] < 0.05])
top_npr = metab_df.nlargest(3, 'delta')['Pathway'].tolist()
top_pcr = metab_df.nsmallest(3, 'delta')['Pathway'].tolist()

print(f"\n{'='*70}")
print("METABOLIC ANALYSIS COMPLETE (V2 — full gene set)")
print(f"{'='*70}")
print(f"Output: {OUTPUT_DIR}/")
print(f"Total metabolic pathways scored: {len(metab_df)}")
print(f"Significant pathways (P<0.05): {n_sig}/{len(metab_df)}")
print(f"nPR-enriched metabolism: {top_npr}")
print(f"pCR-enriched metabolism: {top_pcr}")
print(f"\nKey biological interpretation:")
if any('Glycolysis' in p for p in top_npr):
    print("  -> nPR macrophages: Glycolysis-dominant (Warburg-like, pro-tumor)")
if any('Iron' in p or 'Redox' in p for p in top_npr):
    print("  -> nPR macrophages: Iron/Redox metabolism = ferroptosis resistance")
if any('Arginine' in p for p in top_npr):
    print("  -> ARG1+ signature = M2-like immunosuppression")
if any('Lipid' in p for p in top_pcr) or any('Fatty' in p for p in top_pcr):
    print("  -> pCR macrophages: Lipid/FAO-dominant (anti-tumor phenotype)")
if any('Oxidative' in p for p in top_pcr):
    print("  -> pCR macrophages: OXPHOS-dominant (M1-like, anti-tumor)")
