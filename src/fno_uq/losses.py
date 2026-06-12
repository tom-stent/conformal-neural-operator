import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    """
    Quantile (pinball) loss targeting the 1 - alpha quantile.

    Underprediction (true error above the predicted band) is weighted by
    1 - alpha, overprediction by alpha.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.q = 1.0 - alpha

    def forward(self, u_pred, u_true):
        diff = u_true - u_pred
        loss = torch.maximum(self.q * diff, (self.q - 1.0) * diff)
        return loss.mean()