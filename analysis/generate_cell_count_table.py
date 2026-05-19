#!/usr/bin/env python3
"""
Generate cell count tables for manuscript revision.
Reviewer request: exact cell counts after QC, stratified by
  cohort / histological type / response group / cell type.

Works with:
  - A processed .h5ad file (preferred, after QC)
  - OR the raw metadata CSV + QC parameters (pre-QC, applies filters)

Usage:
  python generate_cell_count_table.py --h5ad processed_data/GSE243013_processed_FULL.h5ad
  python generate_cell_count_table.py --metadata raw/metadata_with_celltype.csv --counts raw/counts.mtx.gz --features raw/features.csv.gz --barcodes raw/barcodes.csv.gz
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# ============================================
# Configuration — QC thresholds (matching recover.py)
# ============================================
QC_CONFIG = {
    "min_genes": 600,
    "max_genes": 6000,
    "min_umi": 1200,
    "max_umi": 40000,
    "max_mt_pct": 15,
    "min_cells_per_gene": 200,
}

# ============================================
# Column mapping (from recover.py metadata columns)
# ============================================
COL_MAP = {
    "cell_id": "cellID",
    "major_cell_type": "major_cell_type",
    "sub_cell_type": "sub_cell_type",
    "sample_id": "sampleID",
    "cancer_type": "cancer_type",          # LUAD / LUSC
    "response_group": "majorMPRtype",       # pCR / MPR / pPR / nPR (or MPRtype for fine)
    "fine_response": "MPRtype",             # finer-grained response
    "therapy": "anti-PD1_therapy",
    "chemo": "chemotherapy",
    "radio_response": "radiological_response",
}


def load_from_h5ad(h5ad_path):
    """Load cell metadata from a processed AnnData .h5ad file."""
    import scanpy as sc
    print(f"Reading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    df = adata.obs.copy()
    print(f"  {len(df):,} cells after QC")
    # If raw counts layer exists, report QC stats
    if "counts" in adata.layers:
        print(f"  Raw counts shape: {adata.layers['counts'].shape}")
    return df


def load_from_raw(metadata_csv, counts_mtx, features_csv, barcodes_csv):
    """Load cell metadata from raw files + apply QC filtering."""
    import scanpy as sc
    from scipy.sparse import csr_matrix

    print("Reading raw data...")
    adata = sc.read_mtx(counts_mtx)
    adata.var_names = pd.read_csv(features_csv, header=0)["geneSymbol"].values
    adata.obs_names = pd.read_csv(barcodes_csv, header=0)["barcode"].values

    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    n_cells_raw = adata.n_obs
    print(f"  Raw: {n_cells_raw:,} cells × {adata.n_vars:,} genes")

    # Merge metadata
    metadata = pd.read_csv(metadata_csv)
    obsnames = pd.DataFrame({"cellID": adata.obs_names})
    newobs = pd.merge(obsnames, metadata, how="left", on="cellID")

    # Map metadata columns
    meta_cols = list(COL_MAP.values())
    for col in meta_cols:
        if col in newobs.columns:
            adata.obs[col] = pd.Categorical(newobs[col])

    # --- QC metrics ---
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"],
                               inplace=True, log1p=True, percent_top=None)

    # --- Filter ---
    before = adata.n_obs
    adata = adata[adata.obs.n_genes_by_counts > QC_CONFIG["min_genes"]]
    adata = adata[adata.obs.n_genes_by_counts < QC_CONFIG["max_genes"]]
    adata = adata[adata.obs.total_counts > QC_CONFIG["min_umi"]]
    adata = adata[adata.obs.total_counts < QC_CONFIG["max_umi"]]
    adata = adata[adata.obs.pct_counts_mt < QC_CONFIG["max_mt_pct"]]
    after = adata.n_obs
    removed = before - after
    print(f"  QC before: {before:,}  →  after: {after:,}  (removed {removed:,}, {removed/before*100:.1f}%)")

    sc.pp.filter_genes(adata, min_cells=QC_CONFIG["min_cells_per_gene"])

    df = adata.obs.copy()
    df["n_genes_by_counts"] = adata.obs["n_genes_by_counts"]
    df["total_counts"] = adata.obs["total_counts"]
    df["pct_counts_mt"] = adata.obs["pct_counts_mt"]
    return df


def build_count_tables(df, output_dir):
    """Build all count tables from the obs dataframe."""

    # Detect available columns
    col_cancer = None
    for c in ["cancer_type", "histology", "Histology"]:
        if c in df.columns:
            col_cancer = c
            break

    col_response = None
    for c in ["majorMPRtype", "MPRtype", "response", "Response"]:
        if c in df.columns:
            col_response = c
            break

    col_celltype = None
    for c in ["major_cell_type", "celltype", "cell_type"]:
        if c in df.columns:
            col_celltype = c
            break

    col_sample = None
    for c in ["sampleID", "sample_id", "Sample"]:
        if c in df.columns:
            col_sample = c
            break

    print(f"\nDetected columns:")
    print(f"  Histology (cancer_type): {col_cancer}")
    print(f"  Response group:          {col_response}")
    print(f"  Major cell type:         {col_celltype}")
    print(f"  Sample ID:               {col_sample}")

    tables = {}

    # ───────────────────────────────────────────────
    # Table 1: QC Summary
    # ───────────────────────────────────────────────
    print("\n=== Table 1: QC Summary ===")
    t1_data = {
        "Parameter": [
            "Min genes per cell",
            "Max genes per cell",
            "Min UMI per cell",
            "Max UMI per cell",
            "Max % mitochondrial",
            "Min cells per gene",
        ],
        "Threshold": [
            QC_CONFIG["min_genes"],
            QC_CONFIG["max_genes"],
            QC_CONFIG["min_umi"],
            QC_CONFIG["max_umi"],
            f"{QC_CONFIG['max_mt_pct']}%",
            QC_CONFIG["min_cells_per_gene"],
        ],
    }
    if "n_genes_by_counts" in df.columns:
        t1_data["Parameter"].extend([
            "Median genes/cell (post-QC)",
            "Median UMI/cell (post-QC)",
            "Total cells retained",
        ])
        t1_data["Threshold"].extend([
            f"{df['n_genes_by_counts'].median():.0f}",
            f"{df['total_counts'].median():.0f}",
            f"{len(df):,}",
        ])

    t1 = pd.DataFrame(t1_data)
    tables["Table1_QC_summary"] = t1
    print(t1.to_string(index=False))

    # ───────────────────────────────────────────────
    # Table 2: Cells per sample (optional, if sampleID present)
    # ───────────────────────────────────────────────
    if col_sample:
        print(f"\n=== Table 2: Cells per sample ===")
        sample_counts = df[col_sample].value_counts().reset_index()
        sample_counts.columns = ["SampleID", "n_cells"]
        sample_counts = sample_counts.sort_values("n_cells", ascending=False)
        tables["Table2_cells_per_sample"] = sample_counts
        print(f"  {len(sample_counts)} samples")
        print(f"  Median cells/sample: {sample_counts['n_cells'].median():.0f}")
        print(f"  Range: {sample_counts['n_cells'].min()} – {sample_counts['n_cells'].max()}")

    # ───────────────────────────────────────────────
    # Table 3: Cells by histological type × response group × cell type
    # ───────────────────────────────────────────────
    group_cols = []
    if col_cancer:
        group_cols.append(col_cancer)
    if col_response:
        group_cols.append(col_response)
    if col_celltype:
        group_cols.append(col_celltype)

    if group_cols:
        print(f"\n=== Table 3: Cell counts by {' × '.join(group_cols)} ===")
        t3 = df.groupby(group_cols, observed=False).size().reset_index(name="n_cells")
        t3 = t3.sort_values("n_cells", ascending=False)
        tables["Table3_cells_by_histology_response_celltype"] = t3
        print(t3.to_string(index=False, max_rows=40))

    # ───────────────────────────────────────────────
    # Table 4: Pivot table — Cell type (rows) × Response (cols) for each Histology
    # ───────────────────────────────────────────────
    if col_cancer and col_response and col_celltype:
        print(f"\n=== Table 4: Pivot — Cell type vs Response, by Histology ===")
        for hist in sorted(df[col_cancer].dropna().unique()):
            print(f"\n  --- {hist} ---")
            subset = df[df[col_cancer] == hist]
            t4 = subset.pivot_table(
                index=col_celltype,
                columns=col_response,
                aggfunc="size",
                fill_value=0,
                observed=False,
            )
            # Add total column
            t4["Total"] = t4.sum(axis=1)
            t4.loc["Total"] = t4.sum(axis=0)
            tables[f"Table4_pivot_{hist}"] = t4
            print(t4.to_string())

    # ───────────────────────────────────────────────
    # Table 5: Total summary (manuscript-ready)
    # ───────────────────────────────────────────────
    print(f"\n=== Table 5: Total Summary ===")
    t5_rows = []
    t5_rows.append({"Category": "Total cells after QC", "Count": f"{len(df):,}"})

    if col_sample:
        t5_rows.append({"Category": "Total samples", "Count": f"{df[col_sample].nunique()}"})

    if col_cancer:
        for v in sorted(df[col_cancer].dropna().unique()):
            cnt = (df[col_cancer] == v).sum()
            t5_rows.append({"Category": f"  ├─ {v}", "Count": f"{cnt:,} ({cnt/len(df)*100:.1f}%)"})

    if col_response:
        for v in sorted(df[col_response].dropna().unique()):
            cnt = (df[col_response] == v).sum()
            t5_rows.append({"Category": f"  Response: {v}", "Count": f"{cnt:,} ({cnt/len(df)*100:.1f}%)"})

    if col_celltype:
        for v in sorted(df[col_celltype].dropna().unique()):
            cnt = (df[col_celltype] == v).sum()
            t5_rows.append({"Category": f"  Cell type: {v}", "Count": f"{cnt:,} ({cnt/len(df)*100:.1f}%)"})

    t5 = pd.DataFrame(t5_rows)
    tables["Table5_summary"] = t5
    print(t5.to_string(index=False))

    return tables


def save_tables(tables, output_dir):
    """Save all tables to CSV and a combined Excel file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Individual CSVs
    for name, df in tables.items():
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Combined Excel with separate sheets
    xlsx_path = output_dir / "cell_count_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet_name = name[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\n  Combined Excel: {xlsx_path}")


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate cell count tables for manuscript")
    parser.add_argument("--h5ad", type=str, help="Path to processed .h5ad file (post-QC)")
    parser.add_argument("--metadata", type=str, help="Path to raw metadata_with_celltype.csv")
    parser.add_argument("--counts", type=str, help="Path to raw counts.mtx.gz")
    parser.add_argument("--features", type=str, help="Path to raw features.csv.gz")
    parser.add_argument("--barcodes", type=str, help="Path to raw barcodes.csv.gz")
    parser.add_argument("--output", type=str, default="cell_count_tables",
                       help="Output directory for tables (default: cell_count_tables)")
    args = parser.parse_args()

    if args.h5ad:
        df = load_from_h5ad(args.h5ad)
    elif args.metadata and args.counts and args.features and args.barcodes:
        df = load_from_raw(args.metadata, args.counts, args.features, args.barcodes)
    else:
        # Try to auto-detect
        h5ad_candidates = list(Path(".").glob("**/*processed*.h5ad")) + list(Path(".").glob("**/*FULL*.h5ad"))
        meta_candidate = Path("raw/metadata_with_celltype.csv")
        if h5ad_candidates:
            print(f"Auto-detected .h5ad: {h5ad_candidates[0]}")
            df = load_from_h5ad(str(h5ad_candidates[0]))
        elif meta_candidate.exists():
            print("Auto-detected raw data, but need --counts, --features, --barcodes too.")
            print("Please specify all four: --metadata raw/metadata_with_celltype.csv --counts raw/counts.mtx.gz --features raw/features.csv.gz --barcodes raw/barcodes.csv.gz")
            return
        else:
            parser.print_help()
            print("\nNo data found. Use --h5ad for processed data, or --metadata + raw files.")
            return

    tables = build_count_tables(df, args.output)
    save_tables(tables, args.output)

    print("\n" + "=" * 60)
    print("DONE. All tables saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
