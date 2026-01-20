import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    'top_k_genes': 4000,
    'housekeeping_genes': [
        "C1orf43", "CHMP2A", "GPI", "PSMB2", "PSMB4", 
        "RAB7A", "REEP5", "SNRPD3", "VCP", "VPS29"
    ]
}

def load_and_preprocess_tcga(expression_path, clinical_path):
    """
    Loads TCGA data, extracts PFI, and filters top 4000 genes by variance.
    """
    print("Loading TCGA data...")
    # Load Expression Data (Rows=Genes, Cols=Patients)
    # Assumes index is gene symbols
    expr_df = pd.read_csv(expression_path, index_col=0)
    
    # Load Clinical Data for PFI (Progression Free Interval) and Event
    clinical_df = pd.read_csv(clinical_path)
    
    # Filter for valid clinical endpoints (PFI)
    # Assuming columns 'PFI' (event) and 'PFI.time' (duration) exist
    if not {'bcr_patient_barcode', 'PFI', 'PFI.time'}.issubset(set(clinical_df.columns)):
        raise ValueError("Clinical CSV must contain 'bcr_patient_barcode', 'PFI', and 'PFI.time' columns.")
    clinical_df = clinical_df[['bcr_patient_barcode', 'PFI', 'PFI.time']].dropna()
    
    # Match samples between expression and clinical data
    # (Simplified matching logic - adjust based on actual barcode formats)
    common_samples = [c for c in expr_df.columns if c[:12] in clinical_df['bcr_patient_barcode'].values]
    expr_df = expr_df[common_samples]
    
    # 1. Discard genes with missing values
    expr_df = expr_df.dropna(axis=0)
    
    # 2. Calculate Variance for each gene 
    variances = expr_df.var(axis=1)
    
    # 3. Select Top 4000 Genes
    top_genes_idx = variances.nlargest(CONFIG['top_k_genes']).index
    expr_df_selected = expr_df.loc[top_genes_idx]
    
    # 4. Log Transformation log2(1+x)
    expr_df_final = np.log2(expr_df_selected + 1)
    
    print(f"TCGA Processed: {expr_df_final.shape}")
    return expr_df_final, clinical_df, top_genes_idx

def normalize_geo_to_tcga(geo_path, tcga_df, top_genes_idx):
    """
    Normalizes GEO data to TCGA using housekeeping genes.
    """
    print("Processing GEO data...")
    geo_df = pd.read_csv(geo_path, index_col=0)
    
    # Ensure GEO has the same 4000 genes as TCGA
    # Fill missing with 0 or drop (Paper implies matching features)
    geo_df = geo_df.reindex(top_genes_idx).fillna(0)
    
    # Housekeeping Gene Normalization
    # Identify intersection of housekeeping genes present in data
    hk_genes = [g for g in CONFIG['housekeeping_genes'] if g in tcga_df.index and g in geo_df.index]
    
    if not hk_genes:
        print("Warning: No housekeeping genes found for normalization.")
        return np.log2(geo_df + 1)

    # Compute Average Expression (E_t and E_g)
    E_t = tcga_df.loc[hk_genes].mean(axis=0).mean() # Mean of HK genes in TCGA
    E_g = geo_df.loc[hk_genes].mean(axis=0).mean()  # Mean of HK genes in GEO
    
    normalization_factor = E_t / E_g
    print(f"Normalization Factor (Et/Eg): {normalization_factor:.4f}")
    
    # Apply Normalization
    geo_df_normalized = geo_df * normalization_factor
    
    # Log Transformation
    geo_df_final = np.log2(geo_df_normalized + 1)
    
    return geo_df_final

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess TCGA/GEO RNA expression data.")
    parser.add_argument("--tcga-expression", required=True, help="Path to TCGA RNA expression CSV.")
    parser.add_argument("--tcga-clinical", required=True, help="Path to TCGA clinical CSV.")
    parser.add_argument("--geo-expression", default=None, help="Optional GEO RNA expression CSV.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tcga_expr, tcga_clin, selected_genes = load_and_preprocess_tcga(
        args.tcga_expression,
        args.tcga_clinical
    )

    tcga_expr.to_csv(output_dir / "tcga_rna_features.csv")
    tcga_clin.to_csv(output_dir / "tcga_clinical_labels.csv")

    if args.geo_expression:
        geo_expr = normalize_geo_to_tcga(
            args.geo_expression,
            tcga_expr,
            selected_genes
        )
        geo_expr.to_csv(output_dir / "geo_rna_unlabeled.csv")
        print("Saved GEO outputs.")
    else:
        print("Skipping GEO preprocessing (no GEO file provided).")


if __name__ == "__main__":
    main()
