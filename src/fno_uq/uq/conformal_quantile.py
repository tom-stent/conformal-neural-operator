import torch
import torch.nn as nn
from fno_uq.train import train
from fno_uq.data.derived_datasets import compute_residual_dataset, pointwise_error
from fno_uq.losses import PinballLoss
import numpy as np
import math

class ConformalQuantileRegressor(nn.Module):
    """
    Risk-controlling quantile neural operator (UQNO, Ma et al. 2024).

    WE HAVE FOR NOW HARDCODED A 2D IMPLEMENTATION

    Wraps a trained base operator with a quantile model that estimates the
    pointwise error magnitude, then scales that estimate by a conformal
    factor so that the resulting pointwise balls form an (alpha, delta)
    risk-controlling prediction set.

    Note: the base predictor passed in is frozen in place (requires_grad
    is set to False on its parameters).
    """

    def __init__(
            self,
            predictor_model: nn.Module,
            quantile_model: nn.Module,
            alpha: float,
            device: torch.device,
            eps: float = 1e-6,
        ):
        """
        Parameters
        ----------
        predictor_model : nn.Module
            Trained base operator. Frozen and set to eval mode.
        quantile_model : nn.Module
            Operator trained (via fit_quantile) to predict the 1 - alpha
            quantile of the pointwise error magnitude.
        alpha : float
            Domain-level miscoverage target. Aim for at least 1 - alpha of
            grid points covered per function.
        device : torch.device
        eps : float, optional
            Floor applied to the quantile model output, which must be a
            positive radius (default 1e-6).
        """
        super().__init__()

        self.alpha = alpha
        self.device = device
        self.eps = eps
        self.quantile_model = quantile_model.to(device)
        self.loss_fn = PinballLoss(alpha=alpha)
        self.scale_factor = 1.0

        for p in predictor_model.parameters():
            p.requires_grad = False
        self.predictor_model = predictor_model.to(device)
        self.predictor_model.eval()


    def fit_quantile(self, loader, optimiser, epochs):
        """
        Train the quantile model on residuals of the base predictor.

        The loader must be disjoint from the data used to train the base
        predictor.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Batches of dicts with keys "x" and "y".
        optimiser : torch.optim.Optimizer
            Must be constructed over self.quantile_model.parameters().
        epochs : int
        """
        quantile_params = {id(p) for p in self.quantile_model.parameters()}
        optimiser_params = {
            id(p) for group in optimiser.param_groups for p in group["params"]
        }
        if not quantile_params & optimiser_params:
            raise ValueError(
                "optimiser does not contain any quantile_model parameters; "
                "construct it with quantile_model.parameters() "
                "(the base predictor is frozen and cannot be trained)"
            )

        self.quantile_model.train()

        loader = compute_residual_dataset(
            self.predictor_model,
            loader=loader,
            device=self.device
        )

        self.quantile_train_history = train(
            model=self.quantile_model,
            train_loader=loader,
            test_loaders=None,
            loss_fn=self.loss_fn,
            optimiser=optimiser,
            device=self.device,
            epochs=epochs
        )


    @torch.no_grad()
    def conformal_calibrate(self, loader, delta: float):
        """
        Compute the conformal scale factor (lambda-hat) on a calibration set.

        Implements Algorithm 1 of the paper.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Calibration batches (disjoint from all training data) of dicts
            with keys "x" and "y". All samples must share one resolution.
        delta : float
            Function-level miscoverage target: at most a delta fraction of
            functions may have domain coverage below 1 - alpha.
        """

        self.predictor_model.eval()
        self.quantile_model.eval()

        n = len(loader.dataset)
        scores = []
        t = None

        for item in loader:
            x = item["x"].to(self.device)
            y = item["y"].to(self.device)

            if t is None:
                m = y.shape[-1] * y.shape[-2] # points in domain (2D)
                t = math.sqrt(math.log(2 / delta) / (2 * m))
                if 1.0 - self.alpha + t > 1.0:
                    raise ValueError(
                        f"alpha={self.alpha} must exceed the resolution "
                        f"correction t={t:.4f} (m={m} grid points): "
                        "increase alpha or use a finer resolution"
                    )

            y_pred = self.predictor_model(x)
            error_pred = self.quantile_model(x).clamp_min(self.eps)
            error_true = pointwise_error(y, y_pred)

            # Compute score values for each function (current batch)
            ratio = (error_true / error_pred).flatten(start_dim=2)
            s = torch.quantile(
                ratio, 1.0 - self.alpha + t, dim=-1, interpolation="higher"
            )
            scores.extend(s.squeeze(1).tolist())

        q = 1 - math.ceil((n + 1) * (delta - math.exp(-2 * m * t**2))) / n
        if q < 0:
            raise ValueError(
                f"score quantile level {q:.4f} < 0: calibration set of size "
                f"n={n} is too small for delta={delta} "
                "(need roughly n >= 2/delta)"
            )
        self.scale_factor = float(np.quantile(scores, q, method="higher"))


    @torch.no_grad()
    def predict_interval(self, a: torch.Tensor):
        """
        Predict the base solution and calibrated uncertainty ball radius.

        Parameters
        ----------
        a : torch.Tensor
            Input function batch of shape (B, in_channels, nx, ny).

        Returns
        -------
        (mean_pred, radius)
            Pointwise prediction and ball radius lambda-hat * E(a), each of
            shape (B, 1, nx, ny). The true value lies within
            mean_pred +- radius with the calibrated (alpha, delta) guarantee.
        """

        mean_pred = self.predictor_model(a)
        radius = self.quantile_model(a).clamp_min(self.eps) * self.scale_factor

        return (mean_pred, radius)

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluate calibrated coverage and bandwidth on a test set.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Batches of dicts with keys "x" and "y".

        Returns
        -------
        dict
            - "coverage": per-function fraction of grid points whose true
              error lies within the predicted ball
            - "calibration_percentage": fraction of functions with
              coverage >= 1 - alpha (the guarantee targets >= 1 - delta)
            - "mean_bandwidth": predicted ball radius averaged over all
              points and functions
        """
        self.predictor_model.eval()
        self.quantile_model.eval()

        coverages, bandwidths = [], []

        for item in loader:
            x = item["x"].to(self.device)
            y = item["y"].to(self.device)

            y_pred, radius = self.predict_interval(x)
            error_true = pointwise_error(y, y_pred)

            covered = (error_true <= radius).float().flatten(start_dim=2)
            coverages.extend(covered.mean(dim=-1).squeeze(1).tolist())
            bandwidths.extend(radius.flatten(start_dim=1).mean(dim=-1).tolist())

        coverages = np.asarray(coverages)

        return {
            "coverage": coverages,
            "calibration_percentage": float(
                (coverages >= 1.0 - self.alpha).mean()
            ),
            "mean_bandwidth": float(np.mean(bandwidths)),
        }