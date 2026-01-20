import os

CONFIG = {
    # --- Paths ---
    'data_root': 'data/processed',
    'tcga_rna_path': 'data/processed/tcga_rna_features.csv',
    'tcga_clinical_path': 'data/processed/tcga_clinical_labels.csv',
    'geo_rna_path': 'data/processed/geo_rna_unlabeled.csv', # Optional
    'wsi_dir': 'data/processed/wsi_features',               # Optional
    'geo_wsi_dir': None,                                    # Optional
    'save_dir': 'checkpoints',

    # --- Model Architecture ---
    'model_type': 'single_modal',  # 'single_modal' or 'multi_modal'
    'input_dim_rna': 4000,
    'input_dim_wsi': 1024,
    'hidden_dims': [1000, 200],
    'dropout_rate': 0.2,
    'wsi_feature_type': 'avg',     # 'avg' for 1x1024, 'seq' for Nx1024
    
    # --- Training Hyperparameters ---
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.002,
    'weight_decay': 1e-4,
    
    # --- Mean Teacher / Semi-Supervised Params ---
    'ema_alpha': 0.99,
    'consistency_w': 1.0,
    'noise_sigma': 0.1,
    
    # --- Hardware ---
    'device': 'cuda',             # 'cuda' or 'cpu'
    'seed': 669
}
