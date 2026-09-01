import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    """
    Quantile (pinball) loss targeting the 1 - gamma quantile.

    Underprediction, where the true error exceeds the predicted band, is
    weighted by 1 - gamma. Overprediction weighted by gamma. The asymmetry 
    makes the minimiser the 1 - gamma quantile rather than the mean.

    Parameters
    ----------
    gamma : float
        Domain-level miscoverage target in (0, 1). The loss targets the
        1 - gamma quantile of the pointwise error.
    """
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.q = 1.0 - gamma

    def forward(self, u_pred, u_true):
        """
        Mean pinball loss over all points and samples.

        Parameters
        ----------
        u_pred, u_true : Tensor
            Of the same shape.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        diff = u_true - u_pred
        loss = torch.maximum(self.q * diff, (self.q - 1.0) * diff)
        return loss.mean()