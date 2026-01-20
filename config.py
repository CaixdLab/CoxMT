import os

CONFIG = {
    # --- Paths ---
    'data_root': 'data/processed',
    'tcga_rna_path': 'data/processed/tcga_rna_features.csv',
    'tcga_clinical_path': 'data/processed/tcga_clinical_labels.csv',
    'geo_rna_path': 'data/processed/geo_rna_unlabeled.csv', # Optional
    'wsi_dir': 'data/processed/wsi_features',               # Optional
    'save_dir': 'checkpoints',

    # --- Model Architecture ---
    'model_type': 'single_modal', # 'single_modal' or 'multi_modal'
    [cite_start]'input_dim_rna': 4000,        # Top 4000 genes [cite: 58]
    [cite_start]'input_dim_wsi': 1024,        # DINOv2 feature dim [cite: 198]
    [cite_start]'hidden_dims': [1000, 200],   # Hidden layers for BRCA [cite: 62]
    [cite_start]'dropout_rate': 0.2,          # Optimal dropout [cite: 103]
    
    # --- Training Hyperparameters ---
    'batch_size': 32,
    'epochs': 100,
    [cite_start]'lr': 0.002,                  # Optimal LR range 0.001-0.005 [cite: 109]
    'weight_decay': 1e-4,         # Regularization (standard assumption)
    
    # --- Mean Teacher / Semi-Supervised Params ---
    [cite_start]'ema_alpha': 0.99,            # EMA decay constant [cite: 103]
    [cite_start]'consistency_w': 1.0,         # Weight for unsupervised loss [cite: 103]
    [cite_start]'noise_sigma': 0.1,           # Gaussian noise std dev [cite: 103]
    
    # --- Hardware ---
    'device': 'cuda',             # 'cuda' or 'cpu'
    'seed': 669
}
