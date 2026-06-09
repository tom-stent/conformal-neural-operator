import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    """
    Quantile loss for uncertainty quantification of Neural Operators.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha # tail probability (1 - quantile)

    def __call__(self, y, y_pred, eps=1e-7):
        """
        Parameters
        ----------
        y : torch.tensor, shape (B, nx, ny)
            True absolute pointwise error of neural operator output.

        y_pred : torch.tensor, shape (B, nx, ny)
            Predicted pointwise quantile widths
        """

        diff = y - y_pred
        y_scale, _ = torch.max(y, dim=0) # (nx, ny) pointwise maximum across batches
        y_scale += eps
        ptwise_loss = torch.max((1 - self.alpha) * diff, -self.alpha * diff)

        scaled_ptwise_loss = (
            ptwise_loss / ((2 * self.alpha * (1 - self.alpha)) * y_scale)
            )

        # Average over all points in domain and all batches
        return torch.mean(scaled_ptwise_loss)