import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from cellphonedb.src.core.methods import cpdb_analysis_method



adata = sc.read('myeloid_visualization/myeloid_annotated.h5ad')
print(adata)
adata.obs.columns




np.random.seed(0)
CELLTYPES = ['Classical Myeloid', 'Mast Cell','cDC2','Monocyte','MKI67+ Macrophage',
'Neutrophil','SELENOP+ Macrophage','CHI3L1+ TAM',
'IGKV+ Cell','ANGPTL4+ TAM','LINC02432+ TAM','FABP4+ Macrophage','MT1M+ Macrophage',
'pDC','cDC1','cDC3','SFTPB+ Macrophage']
PHENOTYPE_COL = 'MPRtype'     # 病理进展
CELLTYPE_COL = 'cell_type'
adata = adata[adata.obs[CELLTYPE_COL].isin(CELLTYPES)].copy()
print("剩余细胞数:", adata.n_obs)
print(adata.obs[CELLTYPE_COL].value_counts())
print(adata.obs[PHENOTYPE_COL].value_counts())


sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    flavor='seurat'
)

adata = adata[:, adata.var['highly_variable']].copy()

BASE_OUTDIR = Path("cellphone_by_MPRtype")
BASE_OUTDIR.mkdir(exist_ok=True)

for phenotype in adata.obs[PHENOTYPE_COL].unique():

    print(f"\n===== Running CellPhoneDB for phenotype: {phenotype} =====")

    adata_p = adata[adata.obs[PHENOTYPE_COL] == phenotype].copy()

    # ---------- 分层抽样 ----------
    target_per_type = 30000   # 每个 cell type 最多 3 万
    selected_cells = []

    for ct in CELLTYPES:
        idx = adata_p.obs.index[adata_p.obs[CELLTYPE_COL] == ct]
        if len(idx) > target_per_type:
            idx = np.random.choice(idx, target_per_type, replace=False)
        selected_cells.extend(idx)

    adata_p = adata_p[selected_cells].copy()

    print("抽样后细胞数:")
    print(adata_p.obs[CELLTYPE_COL].value_counts())

    # ---------- 输出目录 ----------
    outdir = BASE_OUTDIR / phenotype
    outdir.mkdir(exist_ok=True)

    # ---------- metadata.tsv ----------
    metadata = pd.DataFrame({
        'Cell': adata_p.obs_names,
        'cell_type': adata_p.obs[CELLTYPE_COL].values
    })
    metadata.to_csv(outdir / 'metadata.tsv', sep='\t', index=False)

    # ---------- counts.tsv ----------
    X = adata_p.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    counts = pd.DataFrame(
        X.T,
        index=adata_p.var_names,
        columns=adata_p.obs_names
    )
    counts.to_csv(outdir / 'counts.tsv', sep='\t')

    # ---------- run CellPhoneDB ----------
    cpdb_analysis_method.call(
        cpdb_file_path='/home/malixin/wangyang/xkl/meta/cellphone/test/cellphonedb-data-5.0.0/data/cellphonedb.zip',
        meta_file_path=str(outdir / 'metadata.tsv'),
        counts_file_path=str(outdir / 'counts.tsv'),
        counts_data='hgnc_symbol',
        score_interactions=False,
        output_path=str(outdir),
        threads=6,
        threshold=0.1,
        result_precision=3,
        debug=False
    )
