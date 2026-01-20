import torch
import torch.optim as optim
import numpy as np
import os
from itertools import cycle

# Import our custom modules
from config import CONFIG
from src.dataset import get_dataloaders
from src.models import SingleModalCox, MultiModalFusion
from src.losses import cox_ph_loss, consistency_loss
from src.utils import update_ema_variables, add_noise, compute_c_index

def create_model(config):
    """Initializes Student and Teacher models based on configuration."""
    if config['model_type'] == 'single_modal':
        model = SingleModalCox(
            input_dim=config['input_dim_rna'], 
            hidden_dims=config['hidden_dims'],
            dropout_rate=config['dropout_rate']
        )
    elif config['model_type'] == 'multi_modal':
        model = MultiModalFusion(
            rna_dim=config['input_dim_rna'],
            wsi_dim=config['input_dim_wsi']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    return model.to(config['device'])

def train_cox_mt():
    # 1. Setup
    device_name = CONFIG['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_name = 'cpu'
    device = torch.device(device_name)
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    print(f"Initializing Cox-MT Training ({CONFIG['model_type']})...")

    # 2. Data Loaders
    labeled_loader, unlabeled_loader = get_dataloaders(CONFIG)
    
    # If using unlabeled data, we cycle it to match labeled batches or vice versa
    if unlabeled_loader:
        # Create an infinite iterator for unlabeled data
        unlabeled_iter = cycle(unlabeled_loader)
        print(f"Semi-Supervised Mode: {len(labeled_loader.dataset)} Labeled, {len(unlabeled_loader.dataset)} Unlabeled.")
    else:
        unlabeled_iter = None
        print("Supervised Mode: Labeled data only.")

    # 3. Initialize Models (Student & Teacher)
    student_model = create_model(CONFIG)
    teacher_model = create_model(CONFIG)
    
    # Initialize Teacher weights to match Student exactly.
    teacher_model.load_state_dict(student_model.state_dict())
    
    # Teacher does not require gradients (updated via EMA)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 4. Optimizer
    optimizer = optim.Adam(
        student_model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )

    # 5. Training Loop
    global_step = 0
    best_c_index = 0.0

    for epoch in range(CONFIG['epochs']):
        student_model.train()
        teacher_model.train() # Teacher is in train mode for Dropout consistency
        
        epoch_loss_s = 0.0
        epoch_loss_u = 0.0
        all_preds = []
        all_times = []
        all_events = []

        for i, batch_labeled in enumerate(labeled_loader):
            # --- Prepare Labeled Data ---
            x_l = batch_labeled['x_rna'].to(device)
            t_l = batch_labeled['t'].to(device)
            e_l = batch_labeled['e'].to(device)
            x_l_wsi = None
            if CONFIG['model_type'] == 'multi_modal':
                if 'x_wsi_seq' not in batch_labeled:
                    raise ValueError("Multi-modal training requires WSI sequence features.")
                x_l_wsi = batch_labeled['x_wsi_seq'].to(device)
            
            # --- Prepare Unlabeled Data (if available) ---
            x_u = None
            x_u_wsi = None
            if unlabeled_iter:
                try:
                    batch_unlabeled = next(unlabeled_iter)
                    x_u = batch_unlabeled['x_rna'].to(device)
                    if CONFIG['model_type'] == 'multi_modal' and 'x_wsi_seq' in batch_unlabeled:
                        x_u_wsi = batch_unlabeled['x_wsi_seq'].to(device)
                except StopIteration:
                    unlabeled_iter = cycle(unlabeled_loader)
                    batch_unlabeled = next(unlabeled_iter)
                    x_u = batch_unlabeled['x_rna'].to(device)
                    if CONFIG['model_type'] == 'multi_modal' and 'x_wsi_seq' in batch_unlabeled:
                        x_u_wsi = batch_unlabeled['x_wsi_seq'].to(device)
            if CONFIG['model_type'] == 'multi_modal' and x_u is not None and x_u_wsi is None:
                # Unlabeled data missing WSI features; disable unlabeled branch for this run.
                x_u = None
                unlabeled_iter = None

            # Combine Labeled and Unlabeled for Consistency Step
            if x_u is not None:
                x_combined = torch.cat([x_l, x_u], dim=0)
                if CONFIG['model_type'] == 'multi_modal':
                    x_wsi_combined = torch.cat([x_l_wsi, x_u_wsi], dim=0)
            else:
                x_combined = x_l
                if CONFIG['model_type'] == 'multi_modal':
                    x_wsi_combined = x_l_wsi

            # --- Noise Injection ---
            # Add Gaussian noise to inputs for both student and teacher
            noise_sigma = CONFIG['noise_sigma']
            x_student = add_noise(x_combined, sigma=noise_sigma)
            x_teacher = add_noise(x_combined, sigma=noise_sigma) # Independent noise instance
            if CONFIG['model_type'] == 'multi_modal':
                x_wsi_student = add_noise(x_wsi_combined, sigma=noise_sigma)
                x_wsi_teacher = add_noise(x_wsi_combined, sigma=noise_sigma)

            # --- Forward Passes ---
            if CONFIG['model_type'] == 'multi_modal':
                pred_student = student_model(x_student, x_wsi_student)
            else:
                pred_student = student_model(x_student)
            
            with torch.no_grad():
                if CONFIG['model_type'] == 'multi_modal':
                    pred_teacher = teacher_model(x_teacher, x_wsi_teacher)
                else:
                    pred_teacher = teacher_model(x_teacher)

            # --- Loss Calculation ---
            
            # 1. Supervised Loss (L_s)
            # Calculated ONLY on Labeled Uncensored Data
            # Note: pred_student contains [labeled_batch, unlabeled_batch]
            # We slice the first len(x_l) parts corresponding to labeled data
            pred_labeled = pred_student[:len(x_l)]
            
            # Filter uncensored events (e=1) for Cox Loss
            mask_uncensored = (e_l == 1)
            if torch.sum(mask_uncensored) > 0:
                loss_s = cox_ph_loss(pred_labeled, t_l, e_l)
            else:
                loss_s = torch.tensor(0.0, device=device)

            # 2. Unsupervised/Consistency Loss (L_u)
            # Calculated on:
            #   a) Labeled Censored Data (e=0)
            #   b) All Unlabeled Data
            
            pred_teacher_labeled = pred_teacher[:len(x_l)]
            
            # Loss for Labeled Censored samples
            mask_censored = (e_l == 0)
            loss_u_censored = 0.0
            if torch.sum(mask_censored) > 0:
                loss_u_censored = consistency_loss(
                    pred_labeled[mask_censored], 
                    pred_teacher_labeled[mask_censored]
                )
            
            # Loss for Unlabeled samples (entire unlabeled batch)
            loss_u_unlabeled = 0.0
            if x_u is not None:
                pred_unlabeled = pred_student[len(x_l):]
                pred_teacher_unlabeled = pred_teacher[len(x_l):]
                loss_u_unlabeled = consistency_loss(pred_unlabeled, pred_teacher_unlabeled)
                
            # Average the consistency losses
            loss_u = (loss_u_censored + loss_u_unlabeled) / 2.0 if x_u is not None else loss_u_censored

            # --- Total Loss ---
            # L = L_s + w * L_u
            w = CONFIG['consistency_w']
            total_loss = loss_s + (w * loss_u)

            # --- Optimization Step ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --- EMA Update ---
            # Update Teacher weights
            update_ema_variables(student_model, teacher_model, CONFIG['ema_alpha'], global_step)
            global_step += 1

            # Logging metrics
            epoch_loss_s += loss_s.item()
            epoch_loss_u += loss_u.item() if torch.is_tensor(loss_u) else loss_u
            
            # Collect for C-index calculation (using Student on Labeled Training Data)
            all_preds.append(pred_labeled.detach().cpu().numpy())
            all_times.append(t_l.cpu().numpy())
            all_events.append(e_l.cpu().numpy())

        # --- End of Epoch Evaluation ---
        # Calculate Training C-Index
        all_preds = np.concatenate(all_preds)
        all_times = np.concatenate(all_times)
        all_events = np.concatenate(all_events)
        train_c_index = compute_c_index(all_preds, all_times, all_events)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Loss S: {epoch_loss_s/len(labeled_loader):.4f} | "
              f"Loss U: {epoch_loss_u/len(labeled_loader):.4f} | "
              f"Train C-Index: {train_c_index:.4f}")

        # Save Checkpoint
        if train_c_index > best_c_index:
            best_c_index = train_c_index
            torch.save(student_model.state_dict(), f"{CONFIG['save_dir']}/best_student.pth")


if __name__ == "__main__":
    train_cox_mt()
            torch.save(teacher_model.state_dict(), f"{CONFIG['save_dir']}/best_teacher.pth")

if __name__ == "__main__":
    train_cox_mt()
