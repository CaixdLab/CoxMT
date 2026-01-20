import torch
import numpy as np
from lifelines.utils import concordance_index

def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Updates Teacher weights using Exponential Moving Average (EMA) of Student weights.
    theta'_t = alpha * theta'_{t-1} + (1 - alpha) * theta_t
    """
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def add_noise(x, sigma=0.1):
    """
    Adds Gaussian noise to input features[cite: 173].
    Used for consistency regularization.
    """
    if sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise

def compute_c_index(y_pred, t_true, e_true):
    """
    Computes Harrell's C-index.
    y_pred: Predicted Hazard Ratios (Higher = Higher Risk)
    """
    try:
        # Convert to numpy and flatten
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy().flatten()
            t_true = t_true.detach().cpu().numpy().flatten()
            e_true = e_true.detach().cpu().numpy().flatten()
            
        # Note: lifelines expects predicted *survival* time usually, 
        # but c_index calculation is rank based. 
        # If y_pred is Hazard: Higher Hazard -> Lower Survival Time.
        # So we pass -y_pred to align direction if checking concordance with Time.
        return concordance_index(t_true, -y_pred, e_true)
    except Exception as e:
        print(f"C-index error: {e}")
        return 0.5
