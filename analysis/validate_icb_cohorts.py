#!/usr/bin/env python3
"""
ICB Cohort Validation — Complete Script
=======================================
Validates macrophage nPR/pCR scores in NSCLC anti-PD-1 cohorts.
Uses pre-computed ENSG→Symbol mapping for GSE135222.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from pathlib import Path
import gzip, json, re
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("C:/Users/13202/Desktop/figures")
GEO_DIR = Path("C:/Users/13202/Desktop/geo_cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NPR_GENES = ['ANGPTL4', 'OLFML3', 'FCGBP', 'CXCR2', 'ANXA1']
PCR_GENES = ['ITGB8', 'MKI67', 'DNASE1L3', 'FABP4', 'PHLDA3', 'STAC', 'S100A8']

# ENSG→Symbol mapping for GSE135222 (pre-computed)
ENSG_MAP = {
    'ENSG00000167772': 'ANGPTL4', 'ENSG00000116774': 'OLFML3',
    'ENSG00000275395': 'FCGBP', 'ENSG00000180871': 'CXCR2',
    'ENSG00000135046': 'ANXA1', 'ENSG00000105855': 'ITGB8',
    'ENSG00000148773': 'MKI67', 'ENSG00000163687': 'DNASE1L3',
    'ENSG00000170323': 'FABP4', 'ENSG00000174307': 'PHLDA3',
    'ENSG00000144681': 'STAC', 'ENSG00000143546': 'S100A8'
}

print("=" * 60)
print("ICB Cohort Validation — Macrophage Signatures")
print("=" * 60)

results_all = []

# ======================================
# 1. GSE135222 (n=27, TPM, anti-PD-1, PFS endpoint)
# ======================================
print("\n[1] GSE135222 (n=27 NSCLC anti-PD-1)...")

# Load expression
expr_raw = pd.read_csv(GEO_DIR / 'GSE135222_exp.tsv.gz', sep='\t', index_col=0)
print(f"  Expression: {expr_raw.shape}")

# Map ENSG to symbol using our pre-computed map
expr_raw['symbol'] = [ENSG_MAP.get(e.split('.')[0], None) for e in expr_raw.index]
expr_sym = expr_raw.dropna(subset=['symbol'])
expr_sym = expr_sym.set_index('symbol')
# Keep only numeric columns (samples)
sample_cols = [c for c in expr_sym.columns if c.startswith('NSCLC')]
expr_mat = expr_sym[sample_cols].astype(float)

# TPM → log2(TPM+1)
expr_log2 = np.log2(expr_mat + 0.01)
# z-score per gene
gene_means = expr_log2.mean(axis=1)
gene_stds = expr_log2.std(axis=1)
gene_stds[gene_stds == 0] = 1.0
expr_z = expr_log2.subtract(gene_means, axis=0).divide(gene_stds, axis=0)

available_npr = [g for g in NPR_GENES if g in expr_z.index]
available_pcr = [g for g in PCR_GENES if g in expr_z.index]
print(f"  nPR genes: {available_npr}")
print(f"  pCR genes: {available_pcr}")

# Compute scores
scores135222 = pd.DataFrame(index=expr_z.columns)
scores135222['mac_npr_score'] = expr_z.loc[available_npr].mean()
scores135222['mac_pcr_score'] = expr_z.loc[available_pcr].mean()
scores135222['mac_ratio'] = scores135222['mac_npr_score'] - scores135222['mac_pcr_score']

# Parse clinical data
with gzip.open(GEO_DIR / 'GSE135222_series_matrix.txt.gz', 'rt', encoding='latin-1') as f:
    meta = f.read()

sample_ids = []
pfs_map = {}
for line in meta.split('\n'):
    if line.startswith('!Sample_title'):
        vals = line.strip().split('\t')
        sample_ids = [v.strip('"') for v in vals[1:]]
    if 'progression-free survival (pfs):' in line:
        vals = line.strip().split('\t')
        for i, v in enumerate(vals[1:]):
            if i < len(sample_ids):
                pfs_map[sample_ids[i]] = int(v.strip('"').split(':')[1].strip())

# Match samples
matched = {}
for ec in scores135222.index:
    m = re.match(r'NSCLC(\d+)', ec)
    if m:
        for st in sample_ids:
            if m.group(1) in st.replace(' ', ''):
                matched[ec] = st
                break

valid_expr = list(matched.keys())
scores135222 = scores135222.loc[valid_expr]
scores135222['sample_name'] = [matched[e] for e in valid_expr]
scores135222['pfs'] = [pfs_map.get(s, np.nan) for s in scores135222['sample_name']]
# PFS=0 = no progression = responder; PFS=1 = progression = non-responder
scores135222['response'] = 1 - scores135222['pfs']
scores135222['dataset'] = 'GSE135222'
scores135222 = scores135222.dropna(subset=['response', 'mac_npr_score'])
print(f"  Valid: {len(scores135222)} patients, {int(scores135222['response'].sum())} responders (PFS=0)")

results_all.append(scores135222[['mac_npr_score', 'mac_pcr_score', 'mac_ratio', 'response', 'dataset']])

# ======================================
# 2. GSE126044 (n=16, counts, anti-PD-1, RECIST)
# ======================================
print("\n[2] GSE126044 (n=16 NSCLC anti-PD-1)...")

counts_raw = pd.read_csv(GEO_DIR / 'GSE126044_counts.txt.gz', sep='\t', index_col=0)
print(f"  Counts: {counts_raw.shape}")

# log2(CPM+1)
sample_cols2 = [c for c in counts_raw.columns if c.startswith('Dis_')]
lib_sizes = counts_raw[sample_cols2].sum(axis=0)
cpm = counts_raw[sample_cols2].divide(lib_sizes, axis=1) * 1e6
expr_norm = np.log2(cpm + 1)

# z-score
means2 = expr_norm.mean(axis=1)
stds2 = expr_norm.std(axis=1)
stds2[stds2 == 0] = 1.0
expr_z2 = expr_norm.subtract(means2, axis=0).divide(stds2, axis=0)

available_npr2 = [g for g in NPR_GENES if g in expr_z2.index]
available_pcr2 = [g for g in PCR_GENES if g in expr_z2.index]
print(f"  nPR genes: {available_npr2}")
print(f"  pCR genes: {available_pcr2}")

scores126044 = pd.DataFrame(index=sample_cols2)
scores126044['mac_npr_score'] = expr_z2.loc[available_npr2].mean()
scores126044['mac_pcr_score'] = expr_z2.loc[available_pcr2].mean()
scores126044['mac_ratio'] = scores126044['mac_npr_score'] - scores126044['mac_pcr_score']

# Parse response data
with gzip.open(GEO_DIR / 'GSE126044_series_matrix.txt.gz', 'rt', encoding='latin-1') as f:
    meta2 = f.read()

sample_ids2 = []
resp_map = {}
for line in meta2.split('\n'):
    if line.startswith('!Sample_title'):
        vals = line.strip().split('\t')
        sample_ids2 = [v.strip('"') for v in vals[1:]]
    if 'patient response:' in line:
        vals = line.strip().split('\t')
        for i, v in enumerate(vals[1:]):
            if i < len(sample_ids2):
                resp_str = v.strip('"').split(':')[1].strip().lower()
                resp_map[sample_ids2[i]] = 1 if 'responder' in resp_str and 'non' not in resp_str else 0

matched2 = {}
for ec in sample_cols2:
    for st in sample_ids2:
        if ec in st or st in ec:
            matched2[ec] = st
            break

scores126044['sample_name'] = [matched2.get(e, e) for e in sample_cols2]
scores126044['response'] = [resp_map.get(s, np.nan) for s in scores126044['sample_name']]
scores126044['dataset'] = 'GSE126044'
scores126044 = scores126044.dropna(subset=['response', 'mac_npr_score'])
print(f"  Valid: {len(scores126044)} patients, {int(scores126044['response'].sum())} responders")

results_all.append(scores126044[['mac_npr_score', 'mac_pcr_score', 'mac_ratio', 'response', 'dataset']])

# ======================================
# 3. Combined analysis
# ======================================
combined = pd.concat(results_all, axis=0)
print(f"\n[3] Combined: {len(combined)} patients, {int(combined['response'].sum())} responders")

# Summary stats per dataset
for ds in combined['dataset'].unique():
    dd = combined[combined['dataset'] == ds]
    print(f"  {ds}: n={len(dd)}, responders={int(dd['response'].sum())}")

# ---- Fig ICB-1: Score boxplots ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (score, title, color) in enumerate([
    ('mac_npr_score', 'Macrophage nPR Score', '#e74c3c'),
    ('mac_pcr_score', 'Macrophage pCR Score', '#3498db'),
    ('mac_ratio', 'nPR/pCR Difference', '#2ecc71')]):
    ax = axes[i]
    resp = combined[combined['response'] == 1][score]
    nonresp = combined[combined['response'] == 0][score]

    bp = ax.boxplot([nonresp, resp], positions=[0, 1], widths=0.4,
                   patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')

    for pos, data in zip([0, 1], [nonresp, resp]):
        jitter = np.random.normal(0, 0.03, len(data))
        ax.scatter(np.full(len(data), pos) + jitter, data,
                  alpha=0.6, s=40, c='#2c3e50', edgecolors='none')

    u, p = stats.mannwhitneyu(nonresp, resp, alternative='two-sided')
    auc_v = roc_auc_score(combined['response'], combined[score])
    if auc_v < 0.5:
        auc_v = 1 - auc_v

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Non-Responder\n(n={len(nonresp)})',
                       f'Responder\n(n={len(resp)})'], fontsize=11)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f'P={p:.4f} | AUC={auc_v:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Macrophage Scores by ICB Response — NSCLC Cohorts (n={len(combined)})',
            fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_ICB_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig_ICB_boxplots.png")

# ---- Fig ICB-2: ROC curves ----
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ['#e74c3c', '#3498db', '#2ecc71']
for score, label, color in zip(['mac_npr_score', 'mac_pcr_score', 'mac_ratio'],
                                ['nPR Score', 'pCR Score', 'nPR/pCR Ratio'],
                                colors_roc):
    fpr, tpr, _ = roc_curve(combined['response'], combined[score])
    auc_val = roc_auc_score(combined['response'], combined[score])
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{label} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title(f'ROC — ICB Response Prediction\nNSCLC Cohorts GSE135222+GSE126044 (n={len(combined)})',
            fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_ICB_ROC.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig_ICB_ROC.png")

# ---- Fig ICB-3: Per-dataset forest plot ----
fig, ax = plt.subplots(figsize=(10, 5))
y_pos = 0
all_labels = []
for ds_name in combined['dataset'].unique():
    dd = combined[combined['dataset'] == ds_name]
    for score, color in zip(['mac_npr_score', 'mac_pcr_score', 'mac_ratio'], colors_roc):
        if len(np.unique(dd['response'])) >= 2:
            auc_v = roc_auc_score(dd['response'], dd[score])
            boot_aucs = []
            for _ in range(500):
                idx = np.random.choice(len(dd), len(dd), replace=True)
                if len(np.unique(dd['response'].values[idx])) < 2:
                    continue
                boot_aucs.append(roc_auc_score(dd['response'].values[idx], dd[score].values[idx]))
            if boot_aucs:
                ci_low = np.percentile(boot_aucs, 2.5)
                ci_high = np.percentile(boot_aucs, 97.5)
            else:
                ci_low, ci_high = auc_v, auc_v

            ax.barh(y_pos, auc_v - 0.5, left=0.5, height=0.7, color=color, alpha=0.85)
            ax.errorbar(auc_v, y_pos, xerr=[[auc_v - ci_low], [ci_high - auc_v]],
                       fmt='none', color='black', capsize=2)
            short = {'mac_npr_score': 'nPR', 'mac_pcr_score': 'pCR', 'mac_ratio': 'Ratio'}[score]
            all_labels.append(f'{ds_name} | {short}')
            y_pos -= 1

ax.set_yticks(range(y_pos + 1, 1))
ax.set_yticklabels(all_labels, fontsize=10)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, lw=1)
ax.set_xlabel('AUC', fontsize=13)
ax.set_title('ICB Response Prediction — Per-Dataset AUC', fontsize=14)
ax.set_xlim(0.1, 1.0)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_ICB_forest.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig_ICB_forest.png")

# ---- Save results ----
summary_rows = []
for ds_name in combined['dataset'].unique():
    dd = combined[combined['dataset'] == ds_name]
    for score in ['mac_npr_score', 'mac_pcr_score', 'mac_ratio']:
        try:
            auc_v = roc_auc_score(dd['response'], dd[score])
        except:
            auc_v = np.nan
        try:
            u, p = stats.mannwhitneyu(dd[dd['response']==0][score],
                                       dd[dd['response']==1][score])
        except:
            p = np.nan
        summary_rows.append({
            'Dataset': ds_name, 'Score': score,
            'N': len(dd), 'Responders': int(dd['response'].sum()),
            'AUC': round(auc_v, 3), 'MWU_P': round(p, 4)
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / 'icb_validation_results.csv', index=False)
combined.to_csv(OUTPUT_DIR / 'icb_combined_scores.csv', index=True)
print(f"\n  Results saved: icb_validation_results.csv")
print(summary_df.to_string(index=False))

print(f"\n{'='*60}")
print("ICB VALIDATION COMPLETE")
print(f"{'='*60}")
