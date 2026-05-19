# NSCLC Myeloid Pseudobulk Analysis

Patient-level pseudobulk sensitivity analysis for "Macrophage Remodeling Is Associated with Immunotherapy Response and Enables Interpretable Machine-Learning Prediction in NSCLC."


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



## Data Availability

Input: GSE243013 (GEO), processed AnnData object with 31,831 genes × 225 patients.

CellPhoneDB: v5.0.0, threshold=0.1, HVG=3000, 17 myeloid subtypes.

ICB validation: GSE135222 (n=27, TPM, PFS), GSE126044 (n=16, counts, RECIST).

TCGA: FireBrowse API, TCGA-LUAD + TCGA-LUSC, n=996.
