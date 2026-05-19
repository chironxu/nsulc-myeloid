# NSCLC Myeloid Pseudobulk Analysis

Patient-level pseudobulk sensitivity analysis for "Macrophage Remodeling Is Associated with Immunotherapy Response and Enables Interpretable Machine-Learning Prediction in NSCLC."

## Structure

```
├── analysis/                          # Core analysis scripts
│   ├── pseudobulk_analysis.py         # Pseudobulk DE + pathway + signature scores
│   ├── run_scmetabolism_macrophage.py # KEGG metabolic pathway scoring
│   ├── run_pyscenic_macrophage.py     # TF regulon analysis (pySCENIC)
│   ├── AI_response_prediction_model.py # XGBoost/RF ML prediction
│   ├── celldb.py                      # CellPhoneDB ligand-receptor analysis
│   ├── run_tcga_survival_real.py      # TCGA survival analysis
│   ├── validate_icb_cohorts.py        # ICB cohort validation (GSE135222, GSE126044)
│   ├── generate_patient_table.py      # Patient characteristics table
│   └── generate_cell_count_table.py   # Cell count summary table
├── manuscript/                        # Manuscript revision scripts
│   ├── final_revision.py              # Initial revision (17 changes, color-coded)
│   └── update_pseudobulk_numbers.py   # Apply real pseudobulk numbers to manuscript
└── output/                            # Analysis outputs
    ├── data/
    │   ├── pseudobulk_DE_results.csv           # 20 genes, 6 FDR<0.05
    │   ├── pseudobulk_pathway_results.csv      # Iron/Redox, Lipid/PPAR (not sig)
    │   └── pseudobulk_signature_scores.csv     # 154 patients, nPR/pCR/ratio scores
    └── figures/
        ├── FigSXa_pseudobulk_DE_volcano.pdf
        ├── FigSXb_pseudobulk_signature_boxplots.pdf
        └── FigSXc_pseudobulk_pathway_boxplots.pdf
```

## Key Results (n = 154 patients, nPR = 62, pCR = 92)

### Differential Expression (6/20 genes FDR < 0.05)
| Gene | Cohen's d | FDR | Direction |
|------|-----------|-----|-----------|
| OLFML3 | 0.65 | 1.3×10⁻⁴ | nPR |
| MKI67 | 0.72 | 7.9×10⁻⁴ | nPR |
| ANGPTL4 | 0.44 | 6.9×10⁻⁴ | nPR |
| ANXA1 | -0.49 | 0.010 | pCR |
| DNASE1L3 | -0.34 | 0.030 | pCR |
| FABP4 | -0.28 | 0.031 | pCR |

### Signature Scores
- nPR composite score: d = 0.74, P < 0.0001
- pCR composite score: d = -0.10, P = 0.28 (not significant)
- Macrophage ratio (nPR - pCR): d = 0.58, P = 0.004

### Metabolic Pathways (not significant at patient level)
- Iron/Redox: d = -0.03, P = 0.76
- Lipid/PPAR: d = -0.11, P = 0.73

## Usage

```bash
# Pseudobulk analysis
python analysis/pseudobulk_analysis.py \
  --h5ad processed_data/GSE243013_processed_FULL.h5ad \
  --myeloid-key "Myeloid cell"

# Update manuscript with real pseudobulk numbers
python manuscript/update_pseudobulk_numbers.py
```

## Data Availability

Input: GSE243013 (GEO), processed AnnData object with 31,831 genes × 225 patients.

CellPhoneDB: v5.0.0, threshold=0.1, HVG=3000, 17 myeloid subtypes.

ICB validation: GSE135222 (n=27, TPM, PFS), GSE126044 (n=16, counts, RECIST).

TCGA: FireBrowse API, TCGA-LUAD + TCGA-LUSC, n=996.
