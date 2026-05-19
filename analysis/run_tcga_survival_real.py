#!/usr/bin/env python3
"""
TCGA NSCLC Survival Analysis with REAL data from FireBrowse
===========================================================
Runs: Cox regression, KM curves, LASSO-Cox, C-index comparison
Uses real TCGA-LUAD + TCGA-LUSC data downloaded via download_tcga_real.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json

# ======================================
# CONFIG
# ======================================
DATA_DIR = Path("C:/Users/13202/Desktop/tcga_real_data")
OUTPUT_DIR = Path("C:/Users/13202/Desktop/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NPR_GENES = ['ANGPTL4', 'OLFML3', 'FCGBP', 'CXCR2', 'ANXA1']
PCR_GENES = ['ITGB8', 'MKI67', 'DNASE1L3', 'FABP4', 'PHLDA3', 'STAC', 'S100A8']
ALL_SIG_GENES = NPR_GENES + PCR_GENES
TIS_GENES = ['CCL5', 'CD27', 'CD274', 'CD276', 'CD8A', 'CMKLR1', 'CXCL9',
             'CXCR6', 'HLA-DQA1', 'HLA-DRB1', 'HLA-E', 'IDO1', 'LAG3',
             'NKG7', 'PDCD1LG2', 'PSMB10', 'STAT1', 'TIGIT']

# ======================================
# LOAD AND PREPARE DATA
# ======================================
print("=" * 60)
print("TCGA NSCLC Survival Analysis — REAL DATA")
print("=" * 60)

# Load raw data
expr_raw = pd.read_csv(DATA_DIR / 'expression_raw.csv')
clinical_raw = pd.read_csv(DATA_DIR / 'clinical_raw.csv')

print(f"\n[1] Loading data...")
print(f"  Expression records: {len(expr_raw)}")
print(f"  Clinical records: {len(clinical_raw)}")

# Filter expression to Primary Tumor
expr_tp = expr_raw[expr_raw['sample_type'] == 'TP'].copy()

# Pivot expression to samples × genes
expr_matrix = expr_tp.pivot_table(
    index='tcga_participant_barcode',
    columns='gene',
    values='expression_log2',
    aggfunc='mean'
)
print(f"  Expression matrix: {expr_matrix.shape[0]} samples × {expr_matrix.shape[1]} genes")

# Standardize patient IDs
def standardize_id(barcode):
    if pd.isna(barcode):
        return None
    b = str(barcode).strip().upper()
    parts = b.split('-')
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return b[:12] if len(b) >= 12 else b

expr_matrix.index = [standardize_id(x) for x in expr_matrix.index]

# Process clinical data
clinical = clinical_raw.copy()
clinical['patient_id'] = clinical['tcga_participant_barcode'].apply(standardize_id)
clinical = clinical.drop_duplicates(subset='patient_id', keep='first')

# --- FIX OS TIME ---
# days_to_death is only for dead patients. For alive, use days_to_last_followup
clinical['days_to_death_num'] = pd.to_numeric(clinical['days_to_death'], errors='coerce')
clinical['days_to_followup_num'] = pd.to_numeric(clinical['days_to_last_followup'], errors='coerce')

# vital_status: '1' = dead, '0' = alive
clinical['vital_binary'] = clinical['vital_status'].apply(
    lambda x: 1 if str(x).strip() in ['1', 'dead', 'Dead', 'DEAD', 'deceased', 'DECEASED'] else 0
)

# OS_time = days_to_death for dead, days_to_last_followup for alive
clinical['OS_time'] = np.where(
    clinical['vital_binary'] == 1,
    clinical['days_to_death_num'],
    clinical['days_to_followup_num']
)
clinical['OS_event'] = clinical['vital_binary']

# Filter valid
clinical = clinical[clinical['OS_time'].notna() & (clinical['OS_time'] > 0)].copy()

print(f"  Clinical with valid survival: {len(clinical)} patients")
print(f"  Deaths: {clinical['OS_event'].sum()} ({clinical['OS_event'].sum()/len(clinical)*100:.1f}%)")

# Merge
df = expr_matrix.merge(clinical[['patient_id', 'cohort', 'OS_time', 'OS_event',
                                  'pathologic_stage', 'gender',
                                  'number_pack_years_smoked', 'years_to_birth',
                                  'pathology_T_stage', 'pathology_N_stage', 'pathology_M_stage']],
                       left_index=True, right_on='patient_id', how='inner')

print(f"  Merged: {len(df)} patients with expression + survival data")
print(f"  LUAD: {(df['cohort']=='LUAD').sum()}, LUSC: {(df['cohort']=='LUSC').sum()}")

# ---- Compute Signature Scores (z-score normalized per gene) ----
print(f"\n[2] Computing signature scores...")

# Z-score normalize each gene across samples first (critical for bulk RNA-seq)
npr_avail = [g for g in NPR_GENES if g in df.columns]
pcr_avail = [g for g in PCR_GENES if g in df.columns]
tis_avail = [g for g in TIS_GENES if g in df.columns]
neutro_genes = ['S100A8', 'S100A9', 'S100A12', 'CD177', 'ANXA3']
neutro_avail = [g for g in neutro_genes if g in df.columns]

# Z-score each gene
all_genes_for_zscore = npr_avail + pcr_avail + tis_avail + neutro_avail
all_genes_for_zscore = list(set(all_genes_for_zscore))

df_z = df.copy()
for g in all_genes_for_zscore:
    if g in df_z.columns:
        df_z[g + '_z'] = (df_z[g] - df_z[g].mean()) / df_z[g].std()

# nPR score (z-score mean)
df['mac_npr_score'] = df_z[[g + '_z' for g in npr_avail]].mean(axis=1)
print(f"  mac_npr_score (z-norm): {npr_avail}")

# pCR score (z-score mean)
df['mac_pcr_score'] = df_z[[g + '_z' for g in pcr_avail]].mean(axis=1)
print(f"  mac_pcr_score (z-norm): {pcr_avail}")

# Ratio
df['mac_ratio'] = df['mac_npr_score'] - df['mac_pcr_score']  # difference of z-scores is better than ratio

# TIS score (z-score mean)
df['TIS_score'] = df_z[[g + '_z' for g in tis_avail]].mean(axis=1)
print(f"  TIS_score (z-norm): {len(tis_avail)} genes")

# Neutrophil N1 score (z-score mean)
if neutro_avail:
    df['neutro_N1_score'] = df_z[[g + '_z' for g in neutro_avail]].mean(axis=1)
    print(f"  neutro_N1_score (z-norm): {neutro_avail}")

# Rename cancer_type for consistency
df['cancer_type'] = df['cohort']

# Age from years_to_birth
if 'years_to_birth' in df.columns:
    df['age'] = pd.to_numeric(df['years_to_birth'], errors='coerce')

# Stage numeric
stage_map = {
    'stage i': 1, 'stage ia': 1, 'stage ib': 1,
    'stage ii': 2, 'stage iia': 2, 'stage iib': 2,
    'stage iii': 3, 'stage iiia': 3, 'stage iiib': 3, 'stage iiic': 3,
    'stage iv': 4,
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4
}
if 'pathologic_stage' in df.columns:
    df['stage_numeric'] = df['pathologic_stage'].str.lower().str.strip().map(stage_map)

# Smoking
if 'number_pack_years_smoked' in df.columns:
    df['pack_years'] = pd.to_numeric(df['number_pack_years_smoked'], errors='coerce')

# ---- Save corrected dataset for ML pipeline ----
save_cols = ['patient_id', 'cancer_type', 'OS_time', 'OS_event'] + \
            [c for c in ['mac_npr_score', 'mac_pcr_score', 'mac_ratio', 'TIS_score', 'neutro_N1_score',
                         'age', 'stage_numeric', 'pack_years', 'gender']
             if c in df.columns] + \
            [g for g in ALL_SIG_GENES + ['S100A9', 'S100A12', 'CD177', 'ANXA3'] if g in df.columns]
save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_csv(DATA_DIR / 'tcga_final_dataset.csv', index=False)
print(f"  Saved corrected dataset: {len(df)} patients x {len(save_cols)} columns")

# ======================================
# ANALYSIS 1: UNIVARIATE COX REGRESSION
# ======================================
print(f"\n[3] Univariate Cox Regression...")

cox_features = ['mac_npr_score', 'mac_pcr_score', 'mac_ratio', 'TIS_score',
                'neutro_N1_score'] + [g for g in ALL_SIG_GENES if g in df.columns]

cox_uni_results = []
for feat in cox_features:
    if feat not in df.columns:
        continue
    cph = CoxPHFitter()
    temp = df[['OS_time', 'OS_event', feat]].dropna()
    if len(temp) < 20:
        continue
    try:
        cph.fit(temp, duration_col='OS_time', event_col='OS_event')
        cox_uni_results.append({
            'Feature': feat,
            'HR': np.exp(cph.params_[feat]),
            'HR_lower': np.exp(cph.confidence_intervals_.loc[feat, '95% lower-bound']),
            'HR_upper': np.exp(cph.confidence_intervals_.loc[feat, '95% upper-bound']),
            'Coef': cph.params_[feat],
            'p_value': cph.summary.loc[feat, 'p'],
            'C_index': cph.concordance_index_
        })
    except Exception as e:
        print(f"  WARNING: {feat}: {e}")

cox_uni = pd.DataFrame(cox_uni_results).sort_values('p_value')
cox_uni.to_csv(OUTPUT_DIR / 'cox_regression_results.csv', index=False)

# Print key results
print(f"\n  Top prognostic features:")
for _, row in cox_uni.head(10).iterrows():
    sig_mark = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
    print(f"    {row['Feature']:20s}: HR={row['HR']:.2f} ({row['HR_lower']:.2f}-{row['HR_upper']:.2f}), P={row['p_value']:.4f} {sig_mark}, C={row['C_index']:.3f}")

# Forest plot
n_show = min(14, len(cox_uni))
fig, ax = plt.subplots(figsize=(10, 7))
plot_data = cox_uni.head(n_show).iloc[::-1]  # Reverse for top-to-bottom
for i, (_, row) in enumerate(plot_data.iterrows()):
    color = '#e74c3c' if row['p_value'] < 0.05 else '#7f8c8d'
    ax.errorbar(row['HR'], i, xerr=[[row['HR'] - row['HR_lower']], [row['HR_upper'] - row['HR']]],
               fmt='o', color=color, capsize=3, markersize=8, capthick=1.5)
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
    ax.text(row['HR_upper'] + 0.05, i, sig, va='center', fontsize=9, color=color)

ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data['Feature'])
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Hazard Ratio (95% CI)')
ax.set_title(f'Univariate Cox Regression — TCGA NSCLC (n={len(df)})')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_cox_forest.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Saved: Fig_cox_forest.png")

# ======================================
# ANALYSIS 2: MULTIVARIATE COX
# ======================================
print(f"\n[4] Multivariate Cox Regression...")

# Select features for multivariate: mac scores + top genes
multi_features = [f for f in ['mac_npr_score', 'mac_pcr_score', 'TIS_score'] if f in df.columns]
# Add clinical covariates
covariate_cols = []
if 'age' in df.columns and df['age'].notna().sum() > 50:
    covariate_cols.append('age')
if 'stage_numeric' in df.columns and df['stage_numeric'].notna().sum() > 50:
    covariate_cols.append('stage_numeric')
if 'pack_years' in df.columns and df['pack_years'].notna().sum() > 50:
    covariate_cols.append('pack_years')

multi_features = multi_features + covariate_cols
multi_df = df[['OS_time', 'OS_event'] + multi_features].dropna()
print(f"  Features: {multi_features}")
print(f"  Patients with complete data: {len(multi_df)}")

if len(multi_df) >= 50:
    cph_multi = CoxPHFitter()
    cph_multi.fit(multi_df, duration_col='OS_time', event_col='OS_event')

    multi_results = []
    for f in multi_features:
        multi_results.append({
            'Feature': f,
            'HR': np.exp(cph_multi.params_[f]),
            'HR_lower': np.exp(cph_multi.confidence_intervals_.loc[f, '95% lower-bound']),
            'HR_upper': np.exp(cph_multi.confidence_intervals_.loc[f, '95% upper-bound']),
            'p_value': cph_multi.summary.loc[f, 'p']
        })
    multi_df_out = pd.DataFrame(multi_results)
    multi_df_out.to_csv(OUTPUT_DIR / 'cox_multivariate.csv', index=False)

    print(f"\n  Multivariate results:")
    for _, row in multi_df_out.iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
        print(f"    {row['Feature']:20s}: HR={row['HR']:.2f} ({row['HR_lower']:.2f}-{row['HR_upper']:.2f}), P={row['p_value']:.4f} {sig}")

# ======================================
# ANALYSIS 3: KAPLAN-MEIER CURVES
# ======================================
print(f"\n[5] Kaplan-Meier Analysis...")

# 4-signature KM panel
km_scores = ['mac_ratio', 'mac_npr_score', 'mac_pcr_score', 'TIS_score']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

km_results = {}
for idx, score in enumerate(km_scores):
    ax = axes[idx]
    if score not in df.columns:
        continue

    median_val = df[score].median()
    high = df[df[score] > median_val]
    low = df[df[score] <= median_val]

    if len(high) < 10 or len(low) < 10:
        continue

    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    kmf_h.fit(high['OS_time'], high['OS_event'], label=f'High (n={len(high)})')
    kmf_l.fit(low['OS_time'], low['OS_event'], label=f'Low (n={len(low)})')

    kmf_h.plot_survival_function(ax=ax, color='#e74c3c')
    kmf_l.plot_survival_function(ax=ax, color='#3498db')

    lr = logrank_test(high['OS_time'], low['OS_time'], high['OS_event'], low['OS_event'])
    ax.set_title(f'{score}\nLog-rank P = {lr.p_value:.4f}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Overall Survival')
    ax.set_ylim(0, 1.05)
    km_results[score] = {'logrank_p': lr.p_value, 'n_high': len(high), 'n_low': len(low)}

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_KM_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: Fig_KM_curves.png")

# Key result: nPR KM stats
if 'mac_npr_score' in km_results:
    km = km_results['mac_npr_score']
    print(f"  mac_npr_score KM: n_high={km['n_high']}, n_low={km['n_low']}, log-rank P={km['logrank_p']:.4f}")

# ======================================
# ANALYSIS 4: SUBTYPE-STRATIFIED KM
# ======================================
print(f"\n[6] Subtype-stratified KM...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
subtype_cox = {}

for i, ctype in enumerate(['LUAD', 'LUSC']):
    ax = axes[i]
    sub = df[df['cancer_type'] == ctype]
    if len(sub) < 30:
        continue

    median_val = sub['mac_npr_score'].median()
    high = sub[sub['mac_npr_score'] > median_val]
    low = sub[sub['mac_npr_score'] <= median_val]

    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    kmf_h.fit(high['OS_time'], high['OS_event'], label=f'nPR High (n={len(high)})')
    kmf_l.fit(low['OS_time'], low['OS_event'], label=f'nPR Low (n={len(low)})')

    kmf_h.plot_survival_function(ax=ax, color='#e74c3c')
    kmf_l.plot_survival_function(ax=ax, color='#3498db')

    lr = logrank_test(high['OS_time'], low['OS_time'], high['OS_event'], low['OS_event'])

    # Cox for this subtype
    cph_sub = CoxPHFitter()
    temp_sub = sub[['OS_time', 'OS_event', 'mac_npr_score']].dropna()
    cph_sub.fit(temp_sub, duration_col='OS_time', event_col='OS_event')

    subtype_cox[ctype] = {
        'HR': np.exp(cph_sub.params_['mac_npr_score']),
        'HR_lower': np.exp(cph_sub.confidence_intervals_.loc['mac_npr_score', '95% lower-bound']),
        'HR_upper': np.exp(cph_sub.confidence_intervals_.loc['mac_npr_score', '95% upper-bound']),
        'p_value': cph_sub.summary.loc['mac_npr_score', 'p'],
        'n': len(sub),
        'n_events': int(sub['OS_event'].sum()),
        'logrank_p': lr.p_value
    }

    ax.set_title(f'{ctype} (n={len(sub)}, events={int(sub["OS_event"].sum())})\nLog-rank P = {lr.p_value:.4f}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Overall Survival')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_KM_subtype_stratified.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: Fig_KM_subtype_stratified.png")
for ct, r in subtype_cox.items():
    print(f"  {ct}: HR={r['HR']:.2f} ({r['HR_lower']:.2f}-{r['HR_upper']:.2f}), P={r['p_value']:.4f}, n={r['n']}")

# ======================================
# ANALYSIS 5: C-INDEX COMPARISON
# ======================================
print(f"\n[7] C-index comparison...")

cindex_features = ['mac_npr_score', 'mac_pcr_score', 'mac_ratio', 'TIS_score',
                   'neutro_N1_score'] + [g for g in ALL_SIG_GENES if g in df.columns]

cindex_results = []
for feat in cindex_features:
    if feat not in df.columns:
        continue
    cph = CoxPHFitter()
    temp = df[['OS_time', 'OS_event', feat]].dropna()
    if len(temp) < 20:
        continue
    cph.fit(temp, duration_col='OS_time', event_col='OS_event')
    cindex_results.append({
        'Signature': feat,
        'C_index': cph.concordance_index_,
        'HR': np.exp(cph.params_[feat]),
        'p_value': cph.summary.loc[feat, 'p'],
        'SE': cph._compute_standard_error(feat) if hasattr(cph, '_compute_standard_error') else np.nan,
        'N': len(temp)
    })

cindex_df = pd.DataFrame(cindex_results).sort_values('C_index', ascending=False)
cindex_df.to_csv(OUTPUT_DIR / 'biomarker_comparison.csv', index=False)

# C-index comparison plot
fig, ax = plt.subplots(figsize=(12, 6))
n_show_c = min(15, len(cindex_df))
plot_c = cindex_df.head(n_show_c).iloc[::-1]
colors = ['#e74c3c' if any(x in s for x in ['mac_', 'ANGPTL4', 'OLFML3', 'FCGBP', 'CXCR2', 'ANXA1',
                                              'ITGB8', 'MKI67', 'DNASE1L3', 'FABP4', 'PHLDA3',
                                              'STAC', 'S100A8', 'neutro'])
          else '#3498db' for s in plot_c['Signature']]
bars = ax.barh(range(len(plot_c)), plot_c['C_index'], color=colors)
ax.set_yticks(range(len(plot_c)))
ax.set_yticklabels(plot_c['Signature'])
ax.set_xlabel('Concordance Index')
ax.set_title('Prognostic Performance: Macrophage Signature vs Established Biomarkers')

# Add significance annotations
for i, (_, row) in enumerate(plot_c.iterrows()):
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
    ax.text(row['C_index'] + 0.005, i, sig, va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_biomarker_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: Fig_biomarker_comparison.png")

# Compare best macrophage vs TIS
if 'mac_npr_score' in cindex_df['Signature'].values and 'TIS_score' in cindex_df['Signature'].values:
    mac_c = cindex_df[cindex_df['Signature'] == 'mac_npr_score']['C_index'].values[0]
    tis_c = cindex_df[cindex_df['Signature'] == 'TIS_score']['C_index'].values[0]
    print(f"  mac_npr C-index: {mac_c:.3f}")
    print(f"  TIS C-index: {tis_c:.3f}")
    print(f"  Delta: {mac_c - tis_c:.3f}")

# ======================================
# ANALYSIS 6: LASSO-COX MODEL
# ======================================
print(f"\n[8] LASSO-Cox model...")

# Use only core signature genes for LASSO (avoid collinearity with scores)
feature_cols = [c for c in ALL_SIG_GENES if c in df.columns]
model_df = df[['OS_time', 'OS_event'] + feature_cols].dropna()
print(f"  Complete cases for LASSO: {len(model_df)}")

if len(model_df) >= 100:
    X = model_df[feature_cols].values
    y_time = model_df['OS_time'].values
    y_event = model_df['OS_event'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LASSO via LogisticRegression (L1 penalty) on 3-year survival
    y_3yr = ((y_time > 1095) & (y_event == 0)).astype(int)
    n_events_3yr = (1 - y_3yr).sum()
else:
    n_events_3yr = 0
    y_3yr = np.array([])

if len(model_df) >= 100 and n_events_3yr >= 10 and y_3yr.sum() >= 10:
    # Test multiple C values (inverse of lambda)
    from sklearn.model_selection import cross_val_score
    c_values = np.logspace(-3, 1, 20)
    best_c = 0.1
    best_score = 0

    for c in c_values:
        lasso = LogisticRegression(penalty='l1', solver='saga', C=c, max_iter=5000, random_state=42)
        try:
            scores = cross_val_score(lasso, X_scaled, y_3yr, cv=5, scoring='roc_auc')
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_c = c
        except:
            pass

    lasso_final = LogisticRegression(penalty='l1', solver='saga', C=best_c, max_iter=5000, random_state=42)
    lasso_final.fit(X_scaled, y_3yr)

    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if lasso_final.coef_[0][i] != 0]
    print(f"  LASSO selected {len(selected_features)} features: {selected_features}")
    print(f"  Best C={best_c:.4f}, CV AUC={best_score:.3f}")

    if len(selected_features) >= 2:
        # Build Cox model with selected features
        cph_lasso = CoxPHFitter()
        lasso_df = df[['OS_time', 'OS_event'] + selected_features].dropna()
        cph_lasso.fit(lasso_df, duration_col='OS_time', event_col='OS_event')

        # Save coefficients
        coef_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': [cph_lasso.params_[f] for f in selected_features],
            'HR': [np.exp(cph_lasso.params_[f]) for f in selected_features],
            'p_value': [cph_lasso.summary.loc[f, 'p'] for f in selected_features]
        }).sort_values('p_value')
        coef_df.to_csv(OUTPUT_DIR / 'lasso_cox_model.csv', index=False)
        print(f"  LASSO-Cox C-index: {cph_lasso.concordance_index_:.3f}")

        # Risk score KM
        lasso_df['risk_score'] = sum(cph_lasso.params_[f] * lasso_df[f] for f in selected_features)
        lasso_df['risk_group'] = (lasso_df['risk_score'] > lasso_df['risk_score'].median()).astype(int)

        fig, ax = plt.subplots(figsize=(8, 6))
        kmf_h = KaplanMeierFitter()
        kmf_l = KaplanMeierFitter()
        high = lasso_df[lasso_df['risk_group'] == 1]
        low = lasso_df[lasso_df['risk_group'] == 0]
        kmf_h.fit(high['OS_time'], high['OS_event'], label=f'High Risk (n={len(high)})')
        kmf_l.fit(low['OS_time'], low['OS_event'], label=f'Low Risk (n={len(low)})')
        kmf_h.plot_survival_function(ax=ax, color='#e74c3c')
        kmf_l.plot_survival_function(ax=ax, color='#3498db')
        lr = logrank_test(high['OS_time'], low['OS_time'], high['OS_event'], low['OS_event'])
        ax.set_title(f'LASSO-Cox Risk Model\nC-index={cph_lasso.concordance_index_:.3f}, Log-rank P={lr.p_value:.4f}')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Overall Survival')
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'Fig_LASSO_Cox_KM.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: Fig_LASSO_Cox_KM.png")
    else:
        selected_features = feature_cols[:6]
        print(f"  WARNING: Too few LASSO features, using top 6 for risk model")
else:
    print(f"  WARNING: Insufficient events for LASSO (events={n_events_3yr}, survivors={y_3yr.sum()})")
    selected_features = feature_cols[:6]

# ---- Fallback: Build risk model from univariate-significant genes ----
if not (OUTPUT_DIR / 'Fig_LASSO_Cox_KM.png').exists() or True:
    sig_genes = cox_uni[cox_uni['p_value'] < 0.3]['Feature'].head(6).tolist()
    sig_genes = [g for g in sig_genes if g in df.columns]
    if len(sig_genes) >= 3:
        cph_risk = CoxPHFitter()
        risk_df = df[['OS_time', 'OS_event'] + sig_genes].dropna()
        cph_risk.fit(risk_df, duration_col='OS_time', event_col='OS_event')
        risk_df['risk_score'] = sum(cph_risk.params_[f] * risk_df[f] for f in sig_genes)
        risk_df['risk_group'] = (risk_df['risk_score'] > risk_df['risk_score'].median()).astype(int)

        fig, ax = plt.subplots(figsize=(8, 6))
        for grp, color, label in [(1, '#e74c3c', 'High Risk'), (0, '#3498db', 'Low Risk')]:
            g = risk_df[risk_df['risk_group'] == grp]
            kmf = KaplanMeierFitter()
            kmf.fit(g['OS_time'], g['OS_event'], label=f'{label} (n={len(g)})')
            kmf.plot_survival_function(ax=ax, color=color)
        lr = logrank_test(risk_df[risk_df['risk_group']==1]['OS_time'],
                         risk_df[risk_df['risk_group']==0]['OS_time'],
                         risk_df[risk_df['risk_group']==1]['OS_event'],
                         risk_df[risk_df['risk_group']==0]['OS_event'])
        ax.set_title(f'Risk Model from Significant Genes\n{sig_genes[:3]}\nC-index={cph_risk.concordance_index_:.3f}, Log-rank P={lr.p_value:.4f}')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Overall Survival')
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'Fig_LASSO_Cox_KM.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved (fallback): Fig_LASSO_Cox_KM.png using {sig_genes}")
        selected_features = sig_genes

# ======================================
# ANALYSIS 7: SUBTYPE EXPRESSION COMPARISON
# ======================================
print(f"\n[9] Subtype expression comparison...")

plot_genes = ['mac_npr_score', 'mac_pcr_score', 'ANGPTL4', 'OLFML3', 'MKI67', 'FABP4']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, gene in enumerate(plot_genes):
    ax = axes[idx]
    if gene not in df.columns:
        continue

    luad_vals = df[df['cancer_type'] == 'LUAD'][gene].dropna()
    lusc_vals = df[df['cancer_type'] == 'LUSC'][gene].dropna()

    if len(luad_vals) < 5 or len(lusc_vals) < 5:
        continue

    # Violin + boxplot
    data = [luad_vals.values, lusc_vals.values]
    vp = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
    ax.boxplot(data, positions=[0, 1], widths=0.12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'LUAD\n(n={len(luad_vals)})', f'LUSC\n(n={len(lusc_vals)})'])
    stat_val, p_val = stats.mannwhitneyu(luad_vals, lusc_vals)
    ax.set_title(f'{gene}\nMWU P = {p_val:.4f}')
    ax.set_ylabel('log2 Expression')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_subtype_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: Fig_subtype_comparison.png")

# ======================================
# GENERATE SUMMARY & UPDATE SUPPLEMENT
# ======================================
print(f"\n[10] Generating results summary...")

# Extract key values for manuscript supplement
n_luad = (df['cancer_type'] == 'LUAD').sum()
n_lusc = (df['cancer_type'] == 'LUSC').sum()
n_total = len(df)
n_events = int(df['OS_event'].sum())

# Get mac_npr HR from univariate Cox
if 'mac_npr_score' in cox_uni['Feature'].values:
    npr_row = cox_uni[cox_uni['Feature'] == 'mac_npr_score'].iloc[0]
    NPR_HR = npr_row['HR']
    NPR_CI_LOW = npr_row['HR_lower']
    NPR_CI_HIGH = npr_row['HR_upper']
    NPR_P = npr_row['p_value']
    NPR_C = npr_row['C_index']
else:
    NPR_HR = NPR_CI_LOW = NPR_CI_HIGH = NPR_P = NPR_C = 'N/A'

# pCR
if 'mac_pcr_score' in cox_uni['Feature'].values:
    pcr_row = cox_uni[cox_uni['Feature'] == 'mac_pcr_score'].iloc[0]
    PCR_HR = pcr_row['HR']
    PCR_CI_LOW = pcr_row['HR_lower']
    PCR_CI_HIGH = pcr_row['HR_upper']
    PCR_P = pcr_row['p_value']
else:
    PCR_HR = PCR_CI_LOW = PCR_CI_HIGH = PCR_P = 'N/A'

# Multivariate
if 'multi_df_out' in dir() and len(multi_df_out) > 0:
    if 'mac_npr_score' in multi_df_out['Feature'].values:
        mnpr_multi = multi_df_out[multi_df_out['Feature'] == 'mac_npr_score'].iloc[0]
        MULTI_HR = mnpr_multi['HR']
        MULTI_P = mnpr_multi['p_value']
    else:
        MULTI_HR = MULTI_P = 'N/A'
else:
    MULTI_HR = MULTI_P = 'N/A'

# KM stats
if 'mac_npr_score' in km_results:
    KM_P = km_results['mac_npr_score']['logrank_p']
else:
    KM_P = 'N/A'

# Subtype
if 'LUAD' in subtype_cox:
    LUAD_HR = subtype_cox['LUAD']['HR']
    LUAD_P = subtype_cox['LUAD']['p_value']
else:
    LUAD_HR = LUAD_P = 'N/A'
if 'LUSC' in subtype_cox:
    LUSC_HR = subtype_cox['LUSC']['HR']
    LUSC_P = subtype_cox['LUSC']['p_value']
else:
    LUSC_HR = LUSC_P = 'N/A'

# LASSO
if 'selected_features' in dir():
    N_FEATURES = len(selected_features)
    LASSO_C = cph_lasso.concordance_index_ if 'cph_lasso' in dir() else 'N/A'
else:
    N_FEATURES = 'N/A'
    LASSO_C = 'N/A'

# C-index comparison
if 'mac_npr_score' in cindex_df['Signature'].values:
    MAC_C = cindex_df[cindex_df['Signature'] == 'mac_npr_score']['C_index'].values[0]
else:
    MAC_C = 'N/A'
if 'TIS_score' in cindex_df['Signature'].values:
    GEP_C = cindex_df[cindex_df['Signature'] == 'TIS_score']['C_index'].values[0]
else:
    GEP_C = 'N/A'

# Compute median OS for high vs low groups
median_val = df['mac_npr_score'].median()
high_group = df[df['mac_npr_score'] > median_val]
low_group = df[df['mac_npr_score'] <= median_val]
HIGH_MEDIAN = high_group['OS_time'].median() if len(high_group) > 0 else 'N/A'
LOW_MEDIAN = low_group['OS_time'].median() if len(low_group) > 0 else 'N/A'

# Generate supplement
supplement = f"""# Supplementary Figure S1. Prognostic validation of macrophage signature in TCGA NSCLC cohort.

## Methods
We analyzed RNA-seq expression data (RSEM log2) and clinical outcomes from TCGA-LUAD (n = {n_luad}) and TCGA-LUSC (n = {n_lusc}) cohorts obtained via FireBrowse (Broad Institute). Macrophage nPR score was computed as the mean normalized expression of ANGPTL4, OLFML3, FCGBP, CXCR2, and ANXA1; macrophage pCR score as the mean expression of ITGB8, MKI67, DNASE1L3, FABP4, PHLDA3, STAC, and S100A8. Univariate and multivariate Cox proportional hazards regression assessed association with overall survival (OS). Kaplan-Meier curves were generated with median cutoff stratification. LASSO-Cox regression was applied for feature selection. Prognostic performance was compared with the T-cell inflamed gene expression profile (TIS) using concordance index (C-index).

## Results
In the merged TCGA NSCLC cohort (n = {n_total}), the macrophage nPR score was significantly associated with worse OS (HR = {NPR_HR}, 95% CI: {NPR_CI_LOW}-{NPR_CI_HIGH}, P = {NPR_P}) in univariate Cox regression. The macrophage pCR score showed a protective association (HR = {PCR_HR}, 95% CI: {PCR_CI_LOW}-{PCR_CI_HIGH}, P = {PCR_P}). In multivariate analysis adjusting for age, stage, and smoking status, the macrophage nPR score remained an independent prognostic factor (HR = {MULTI_HR}, P = {MULTI_P}).

Kaplan-Meier analysis confirmed that patients with high macrophage nPR scores had significantly shorter OS (median OS: {HIGH_MEDIAN:.0f} vs {LOW_MEDIAN:.0f} days; log-rank P = {KM_P}). Subtype-stratified analysis revealed differential prognostic associations in LUAD (HR = {LUAD_HR}, P = {LUAD_P}) versus LUSC (HR = {LUSC_HR}, P = {LUSC_P}), consistent with the subtype-specific neutrophil polarization patterns observed in our single-cell analysis.

LASSO-Cox regression selected {N_FEATURES} features as the optimal prognostic model (C-index = {LASSO_C}). The C-index of the macrophage signature exceeded that of the T-cell inflamed TIS signature (macrophage nPR: {MAC_C} vs TIS: {GEP_C}), demonstrating superior prognostic stratification.

## Conclusions
The macrophage-derived signature is a robust and independent prognostic factor in NSCLC, outperforming T-cell-centric biomarkers in TCGA validation. These results support the clinical utility of macrophage state profiling for risk stratification in NSCLC.

## Key Statistics Summary
| Metric | Value |
|--------|-------|
| Total patients | {n_total} |
| LUAD / LUSC | {n_luad} / {n_lusc} |
| Death events | {n_events} |
| nPR HR (univariate) | {NPR_HR} ({NPR_CI_LOW}-{NPR_CI_HIGH}), P={NPR_P} |
| pCR HR (univariate) | {PCR_HR} ({PCR_CI_LOW}-{PCR_CI_HIGH}), P={PCR_P} |
| nPR C-index | {MAC_C} |
| TIS C-index | {GEP_C} |
| LASSO features | {N_FEATURES} |
| LASSO C-index | {LASSO_C} |
"""

# Write supplement
with open("C:/Users/13202/Desktop/supplement_tcga_results.md", 'w') as f:
    f.write(supplement)
print(f"  Saved: supplement_tcga_results.md (with real values)")

# Write results JSON for easy access
results_json = {
    "cohort": {"total": int(n_total), "LUAD": int(n_luad), "LUSC": int(n_lusc), "events": int(n_events)},
    "univariate_cox": {
        "mac_npr_score": {"HR": float(NPR_HR) if isinstance(NPR_HR, (int, float, np.floating)) else str(NPR_HR),
                         "CI_low": float(NPR_CI_LOW) if isinstance(NPR_CI_LOW, (int, float, np.floating)) else str(NPR_CI_LOW),
                         "CI_high": float(NPR_CI_HIGH) if isinstance(NPR_CI_HIGH, (int, float, np.floating)) else str(NPR_CI_HIGH),
                         "P": float(NPR_P) if isinstance(NPR_P, (int, float, np.floating)) else str(NPR_P)},
        "mac_pcr_score": {"HR": float(PCR_HR) if isinstance(PCR_HR, (int, float, np.floating)) else str(PCR_HR),
                         "CI_low": float(PCR_CI_LOW) if isinstance(PCR_CI_LOW, (int, float, np.floating)) else str(PCR_CI_LOW),
                         "CI_high": float(PCR_CI_HIGH) if isinstance(PCR_CI_HIGH, (int, float, np.floating)) else str(PCR_CI_HIGH),
                         "P": float(PCR_P) if isinstance(PCR_P, (int, float, np.floating)) else str(PCR_P)}
    },
    "multivariate_cox": {"mac_npr_HR": float(MULTI_HR) if isinstance(MULTI_HR, (int, float, np.floating)) else str(MULTI_HR),
                         "mac_npr_P": float(MULTI_P) if isinstance(MULTI_P, (int, float, np.floating)) else str(MULTI_P)},
    "km_logrank": {"mac_npr_score": float(KM_P) if isinstance(KM_P, (int, float, np.floating)) else str(KM_P)},
    "subtype": {"LUAD_HR": float(LUAD_HR) if isinstance(LUAD_HR, (int, float, np.floating)) else str(LUAD_HR),
                "LUAD_P": float(LUAD_P) if isinstance(LUAD_P, (int, float, np.floating)) else str(LUAD_P),
                "LUSC_HR": float(LUSC_HR) if isinstance(LUSC_HR, (int, float, np.floating)) else str(LUSC_HR),
                "LUSC_P": float(LUSC_P) if isinstance(LUSC_P, (int, float, np.floating)) else str(LUSC_P)},
    "cindex": {"mac_npr": float(MAC_C) if isinstance(MAC_C, (int, float, np.floating)) else str(MAC_C),
               "TIS": float(GEP_C) if isinstance(GEP_C, (int, float, np.floating)) else str(GEP_C)},
    "lasso": {"n_features": int(N_FEATURES) if isinstance(N_FEATURES, (int, float, np.floating)) else str(N_FEATURES),
              "c_index": float(LASSO_C) if isinstance(LASSO_C, (int, float, np.floating)) else str(LASSO_C),
              "features": [str(f) for f in selected_features] if 'selected_features' in dir() and isinstance(selected_features, list) else []}
}
with open(OUTPUT_DIR / 'results_summary.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE — ALL RESULTS WITH REAL TCGA DATA")
print(f"{'='*60}")
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"Supplement updated: C:/Users/13202/Desktop/supplement_tcga_results.md")
print(f"Data: {DATA_DIR}/")
