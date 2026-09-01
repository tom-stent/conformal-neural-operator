import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalisedOperator(nn.Module):
    """
    Wrap an operator so it consumes and produces data in physical units, while
    the forward applies fitted input normaliser before the model and the inverse 
    output normaliser after it.
    """

    def __init__(
        self,
        model: nn.Module,
        in_normaliser: nn.Module | None = None,
        out_normaliser: nn.Module | None = None,
    ):
        super().__init__()
        self.model = model
        self.in_normaliser = in_normaliser
        self.out_normaliser = out_normaliser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_normaliser is not None:
            x = self.in_normaliser.transform(x)
        out = self.model(x)
        if self.out_normaliser is not None:
            out = self.out_normaliser.inverse_transform(out)
        return out


class PositiveOperator(nn.Module):
    """Wrap an operator so its output is strictly positive (softplus)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        return F.softplus(self.base(x))