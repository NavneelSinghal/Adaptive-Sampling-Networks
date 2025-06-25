import torch
import torch.nn as nn
import torch.nn.functional as F

class TruncatedLogitsLoss(nn.Module):
    """
    Custom loss function for learning to mimic a logit truncation algorithm.

    This loss has two components:
    1. KL Divergence: Measures the difference between the predicted probability
       distribution and the target distribution over the non-truncated classes.
    2. Truncation Penalty: Penalizes the model for assigning probability mass
       to classes that should be truncated (where the target logit is -inf).

    Loss = D_KL(P_target || P_pred) + gamma * sum(P_pred_i for i in truncated_indices)
    """
    def __init__(self, gamma: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be one of 'none', 'mean', or 'sum', but got {reduction}")

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, l_pred: torch.Tensor, l_target: torch.Tensor) -> torch.Tensor:
        truncated_mask = torch.isneginf(l_target)
        log_p_pred = F.log_softmax(l_pred, dim=-1)
        with torch.no_grad():
            p_target = F.softmax(l_target, dim=-1)
            log_p_target = F.log_softmax(l_target, dim=-1)

        kl_div_elements = p_target * (log_p_target - log_p_pred)
        kl_div_elements[truncated_mask] = 0.0
        kl_loss_per_item = kl_div_elements.sum(dim=-1)
        if self.reduction == 'mean':
            kl_loss = kl_loss_per_item.mean()
        elif self.reduction == 'sum':
            kl_loss = kl_loss_per_item.sum()
        else:
            kl_loss = kl_loss_per_item

        p_pred = log_p_pred.exp()
        penalty_probs = p_pred * truncated_mask.float()
        penalty_per_item = penalty_probs.sum(dim=-1)
        if self.reduction == 'mean':
            penalty_loss = penalty_per_item.mean()
        elif self.reduction == 'sum':
            penalty_loss = penalty_per_item.sum()
        else:
            penalty_loss = penalty_per_item

        total_loss = kl_loss + self.gamma * penalty_loss
        return total_loss, kl_loss, penalty_loss
