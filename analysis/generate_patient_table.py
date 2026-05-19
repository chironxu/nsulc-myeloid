#!/usr/bin/env python3
"""
Generate per-patient cell count table for manuscript revision.
Rows = patients (sampleID), Columns = clinical info + cell-type counts.

Usage:
  python generate_patient_table.py --h5ad processed_data/GSE243013_processed_FULL.h5ad
  python generate_patient_table.py --metadata raw/metadata_with_celltype.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Clinical columns to include per patient (from metadata)
CLINICAL_COLS = [
    "cancer_type",              # LUAD / LUSC
    "majorMPRtype",             # pCR / MPR / pPR / nPR (or non-MPR)
    "MPRtype",                  # finer response
    "pathological_response",    # MPR / non-MPR
    "pathological_response_rate",  # numeric %
    "anti-PD1_therapy",         # anti-PD1 drug
    "chemotherapy",             # chemo regimen
    "targeted_therapy",         # targeted therapy
    "cycles",                   # treatment cycles
    "radiological_response",    # SD / PR / PD
    "gender",                   # M / F
    "age",                      # years
    "smoking_history",          # Y / N
    "pre_treatment_staging",    # TNM stage
]


def load_from_h5ad(h5ad_path):
    """Load per-cell obs from processed .h5ad."""
    import scanpy as sc
    print(f"Reading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    df = adata.obs.copy()
    print(f"  {len(df):,} cells, {df['sampleID'].nunique():,} samples")
    return df


def load_from_metadata(metadata_csv):
    """Load directly from metadata CSV (no QC, assumes pre-filtered)."""
    print(f"Reading: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    print(f"  {len(df):,} cells, {df['sampleID'].nunique():,} samples")
    return df


def build_patient_table(df):
    """Build a per-patient table with clinical info + cell-type counts."""

    # Detect column names (handle variations)
    col_sample = None
    for c in ["sampleID", "sample_id", "Sample"]:
        if c in df.columns:
            col_sample = c
            break

    col_celltype = None
    for c in ["celltype", "sub_cell_type", "major_cell_type"]:
        if c in df.columns:
            col_celltype = c
            break

    if not col_sample or not col_celltype:
        print("ERROR: need sampleID and major_cell_type columns")
        return None

    # --- Step 1: Count cells per patient × cell type ---
    print(f"\nCounting cells per patient × cell type...")
    cell_counts = (
        df.groupby([col_sample, col_celltype], observed=False)
          .size()
          .unstack(fill_value=0)
    )
    print(f"  Cell types found: {list(cell_counts.columns)}")

    # Add total cells per patient
    cell_counts.insert(0, "Total_cells", cell_counts.sum(axis=1))

    # Rename columns: "NK cell" → "NK_cells"
    cell_counts.columns = [
        str(c).replace(" ", "_").replace("/", "_") + "_cells"
        if c != "Total_cells" else c
        for c in cell_counts.columns
    ]

    # --- Step 2: Extract per-patient clinical info (one row per sample) ---
    print(f"Extracting clinical info per patient...")

    # Find which clinical columns actually exist
    clinical_cols_found = []
    for c in CLINICAL_COLS:
        if c in df.columns:
            clinical_cols_found.append(c)
        else:
            # Try alternate names
            for alt in [c.replace("-", "_"), c.replace("_", "-")]:
                if alt in df.columns:
                    clinical_cols_found.append(alt)
                    break

    # Get one row per patient for clinical columns
    # Use the first occurrence (all cells from same patient should have same clinical values)
    clinical = df.groupby(col_sample)[clinical_cols_found].first()

    available_clinical = list(clinical.columns)
    print(f"  Clinical columns found: {available_clinical}")

    # --- Step 3: Merge cell counts + clinical info ---
    patient_table = clinical.join(cell_counts, how="left")

    # --- Step 4: Add percentage columns for each cell type ---
    cell_type_cols = [c for c in patient_table.columns if c.endswith("_cells") and c != "Total_cells"]
    for ct in cell_type_cols:
        pct_col = ct.replace("_cells", "_pct")
        patient_table[pct_col] = (
            patient_table[ct] / patient_table["Total_cells"] * 100
        ).round(1)

    # Reorder: clinical columns → Total_cells → cell counts → percentages
    ordered_cols = (
        available_clinical +
        ["Total_cells"] +
        sorted(cell_type_cols) +
        sorted([c for c in patient_table.columns if c.endswith("_pct")])
    )
    patient_table = patient_table[ordered_cols]

    # Sort by cancer_type, then by Total_cells descending
    if "cancer_type" in patient_table.columns:
        patient_table = patient_table.sort_values(["cancer_type", "Total_cells"], ascending=[True, False])
    else:
        patient_table = patient_table.sort_values("Total_cells", ascending=False)

    return patient_table


def save_table(patient_table, output_dir):
    """Save patient table to CSV and Excel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "per_patient_cell_counts.csv"
    patient_table.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    xlsx_path = output_dir / "per_patient_cell_counts.xlsx"
    patient_table.to_excel(xlsx_path, sheet_name="Patient_Cell_Counts")
    print(f"  Saved: {xlsx_path}")

    # Print summary
    print(f"\n=== Patient Table Summary ===")
    print(f"  Patients: {len(patient_table)}")
    print(f"  Columns: {len(patient_table.columns)}")
    print(f"  Median cells/patient: {patient_table['Total_cells'].median():.0f}")
    print(f"  Range: {patient_table['Total_cells'].min()} – {patient_table['Total_cells'].max()}")
    print(f"\n  Clinical columns:")
    for c in patient_table.columns:
        if not c.endswith("_cells") and not c.endswith("_pct") and c != "Total_cells":
            print(f"    {c}: {patient_table[c].nunique()} unique values")
    print(f"\n  Cell type columns:")
    for c in patient_table.columns:
        if c.endswith("_cells") and c != "Total_cells":
            total = patient_table[c].sum()
            print(f"    {c}: {total:,} total ({patient_table[c].mean():.0f} mean/patient)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-patient cell count table"
    )
    parser.add_argument("--h5ad", type=str, help="Path to processed .h5ad file")
    parser.add_argument("--metadata", type=str, help="Path to metadata_with_celltype.csv (for raw metadata only)")
    parser.add_argument("--output", type=str, default="patient_table",
                       help="Output directory (default: patient_table)")
    args = parser.parse_args()

    if args.h5ad:
        df = load_from_h5ad(args.h5ad)
    elif args.metadata:
        df = load_from_metadata(args.metadata)
    else:
        # Auto-detect
        h5ad_candidates = (
            list(Path(".").glob("**/*processed*.h5ad")) +
            list(Path(".").glob("**/*FULL*.h5ad"))
        )
        meta_candidate = Path("raw/metadata_with_celltype.csv")
        if h5ad_candidates:
            print(f"Auto-detected .h5ad: {h5ad_candidates[0]}")
            df = load_from_h5ad(str(h5ad_candidates[0]))
        elif meta_candidate.exists():
            print(f"Auto-detected metadata: {meta_candidate}")
            df = load_from_metadata(str(meta_candidate))
        else:
            parser.print_help()
            print("\nNo data found.")
            return

    patient_table = build_patient_table(df)
    if patient_table is not None:
        save_table(patient_table, args.output)
        print("\nDONE.")

    # Print first few rows as preview
    print("\n--- Preview (first 5 patients) ---")
    print(patient_table.head(5).to_string())


if __name__ == "__main__":
    main()
