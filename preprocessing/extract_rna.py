import pandas as pd
import numpy as np
import os

# Configuration based on the paper
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
    
    # Filter for valid clinical endpoints (PFI) [cite: 142]
    # Assuming columns 'PFI' (event) and 'PFI.time' (duration) exist
    clinical_df = clinical_df[['bcr_patient_barcode', 'PFI', 'PFI.time']].dropna()
    
    # Match samples between expression and clinical data
    # (Simplified matching logic - adjust based on actual barcode formats)
    common_samples = [c for c in expr_df.columns if c[:12] in clinical_df['bcr_patient_barcode'].values]
    expr_df = expr_df[common_samples]
    
    # 1. Discard genes with missing values [cite: 144]
    expr_df = expr_df.dropna(axis=0)
    
    # 2. Calculate Variance for each gene 
    variances = expr_df.var(axis=1)
    
    # 3. Select Top 4000 Genes [cite: 58, 144]
    top_genes_idx = variances.nlargest(CONFIG['top_k_genes']).index
    expr_df_selected = expr_df.loc[top_genes_idx]
    
    # 4. Log Transformation log2(1+x) [cite: 145]
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
    
    # Housekeeping Gene Normalization [cite: 149-152]
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
    
    # Apply Normalization [cite: 152]
    geo_df_normalized = geo_df * normalization_factor
    
    # Log Transformation [cite: 145]
    geo_df_final = np.log2(geo_df_normalized + 1)
    
    return geo_df_final

if __name__ == "__main__":
    # Example Usage
    tcga_expr, tcga_clin, selected_genes = load_and_preprocess_tcga(
        'data/TCGA_BRCA_FPKM.csv', 
        'data/TCGA_CDR.csv'
    )
    
    geo_expr = normalize_geo_to_tcga(
        'data/GSE96058_FPKM.csv', 
        tcga_expr, 
        selected_genes
    )
    
    # Save processed files
    tcga_expr.to_csv('data/processed/tcga_rna_features.csv')
    tcga_clin.to_csv('data/processed/tcga_clinical_labels.csv')
    geo_expr.to_csv('data/processed/geo_rna_unlabeled.csv')
