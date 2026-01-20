import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SurvivalDataset(Dataset):
    """
    Standard PyTorch Dataset for Survival Analysis.
    Returns: (features, time, event_indicator)
    """
    def __init__(self, rna_file, clinical_file=None, wsi_dir=None, mode='labeled'):
        """
        Args:
            rna_file (str): Path to processed RNA-seq CSV.
            clinical_file (str): Path to clinical CSV (None for unlabeled data).
            wsi_dir (str): Directory containing .npy WSI features (optional).
            mode (str): 'labeled' (TCGA) or 'unlabeled' (GEO).
        """
        self.rna_data = pd.read_csv(rna_file, index_col=0)
        self.mode = mode
        self.wsi_dir = wsi_dir
        
        if self.mode == 'labeled':
            self.clinical = pd.read_csv(clinical_file, index_col=0)
            # Ensure alignment between RNA and Clinical data
            common_indices = self.rna_data.index.intersection(self.clinical.index)
            self.rna_data = self.rna_data.loc[common_indices]
            self.clinical = self.clinical.loc[common_indices]
            
            # Extract Time and Event (PFI)
            self.times = self.clinical['PFI.time'].values.astype(np.float32)
            self.events = self.clinical['PFI'].values.astype(np.float32) # 1=Uncensored, 0=Censored
        else:
            # For unlabeled data, time/event are dummy values
            self.times = np.zeros(len(self.rna_data))
            self.events = np.zeros(len(self.rna_data)) # Treated as censored/unlabeled

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        # 1. Load RNA Features
        rna_feat = self.rna_data.iloc[idx].values.astype(np.float32)
        sample_id = self.rna_data.index[idx]
        
        # 2. Load WSI Features (if available)
        # Returns zero vector if WSI is not used/found
        wsi_feat = np.zeros((1024,), dtype=np.float32) # Default placeholder
        if self.wsi_dir:
            wsi_path = f"{self.wsi_dir}/{sample_id}_avg.npy" # Load precomputed avg
            try:
                wsi_feat = np.load(wsi_path).astype(np.float32)
            except FileNotFoundError:
                pass # Handle missing WSI if necessary

        return {
            'x_rna': torch.tensor(rna_feat),
            'x_wsi': torch.tensor(wsi_feat),
            't': torch.tensor(self.times[idx]),
            'e': torch.tensor(self.events[idx])
        }

def get_dataloaders(config):
    """
    Creates dataloaders for labeled and unlabeled data.
    """
    # Labeled Data (TCGA)
    train_ds = SurvivalDataset(
        rna_file=config['tcga_rna_path'],
        clinical_file=config['tcga_clinical_path'],
        wsi_dir=config.get('wsi_dir')
    )
    
    # Unlabeled Data (GEO) - Optional
    unlabeled_ds = None
    if config.get('geo_rna_path'):
        unlabeled_ds = SurvivalDataset(
            rna_file=config['geo_rna_path'],
            mode='unlabeled'
        )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    unlabeled_loader = None
    if unlabeled_ds:
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        
    return train_loader, unlabeled_loader
