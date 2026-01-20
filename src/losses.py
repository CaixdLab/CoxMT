import torch
import torch.nn.functional as F


def cox_ph_loss(log_hazard, times, events):
    """
    Negative partial log-likelihood for Cox proportional hazards.
    log_hazard: (N, 1) or (N,) risk scores (higher = higher risk)
    times: (N,) survival/censoring times
    events: (N,) 1 if event observed, 0 if censored
    """
    if log_hazard.dim() > 1:
        log_hazard = log_hazard.squeeze(-1)

    # Sort by descending time so risk sets are cumulative
    order = torch.argsort(times, descending=True)
    log_hazard = log_hazard[order]
    events = events[order]

    # log(sum(exp(log_hazard))) over the risk set
    log_cumsum_exp = torch.logcumsumexp(log_hazard, dim=0)
    # Only events contribute
    event_mask = events > 0
    if torch.sum(event_mask) == 0:
        return torch.tensor(0.0, device=log_hazard.device)

    neg_log_likelihood = -(log_hazard[event_mask] - log_cumsum_exp[event_mask]).mean()
    return neg_log_likelihood


def consistency_loss(student_pred, teacher_pred):
    """
    Mean squared error between student and teacher predictions.
    """
    return F.mse_loss(student_pred, teacher_pred)
