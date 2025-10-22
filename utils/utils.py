import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

from code_slurm.model_2graph import device

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        logpt = -self.ce_loss(input, target)
        pt = torch.exp(logpt).to(device)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(labels):
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels, minlength=6)[[0, 3, 4, 5]]
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum() * len(weights)
    return weights.to(dtype=torch.float, device=device)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def compute_metrics(y_true, y_pred):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted', zero_division=0),
        recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        cohen_kappa_score(y_true, y_pred, weights='quadratic')
    )
