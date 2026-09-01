import torch
import torch.nn as nn
from conformal_no.training.loops import train
from conformal_no.utils.error import compute_residual_dataset, pointwise_error
from conformal_no.utils.losses import PinballLoss
import numpy as np
import math

class ConformalQuantileRegressor(nn.Module):
    """
    Prediction/error operator pair (UQNO, Ma et al. 2024).

    Works for any spatial rank: the domain point count and per-function scores 
    are computed over all trailing spatial axes.

    Wraps a trained base operator with a quantile model that estimates the
    pointwise error magnitude, then scales that estimate by a conformal factor 
    so that the resulting pointwise balls form an (gamma, alpha) prediction band.

    Scaling factor calibration supports three methods: grid-level, Ma et al. and
    continuum calibration.

    Note: the base predictor passed in is frozen in place (requires_grad is set 
    to False on its parameters).
    """

    def __init__(
            self,
            predictor_model: nn.Module,
            quantile_model: nn.Module,
            gamma: float,
            device: torch.device,
            eps: float = 1e-9,
        ):
        """
        Parameters
        ----------
        predictor_model : nn.Module
            Trained base operator. Frozen and set to eval mode.
        quantile_model : nn.Module
            Operator trained (via fit_quantile) to predict the 1 - gamma
            quantile of the pointwise error magnitude.
        gamma : float
            Domain-level miscoverage target. Aim for at least 1 - gamma of
            grid points covered per function.
        device : torch.device
        eps : float, optional
            Floor applied to the quantile model output, which must be a
            positive radius (default 1e-6).
        """
        super().__init__()

        self.register_buffer("residual_scale", torch.tensor(1.0)) # new

        self.gamma = gamma
        self.device = device
        self.eps = eps
        self.quantile_model = quantile_model.to(device)
        self.loss_fn = PinballLoss(gamma=gamma)
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

        loader, scale = compute_residual_dataset(
            self.predictor_model,
            loader=loader,
            device=self.device
        )
        self.residual_scale = scale.to(self.device)
        print(f"residual scale = {self.residual_scale.item():.3e}")

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
    def conformal_calibrate(self, loader, alpha: float):
        """
        Compute the conformal scale factor (lambda-hat) on a calibration set.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Calibration batches (disjoint from all training data) of dicts
            with keys "x" and "y". All samples must share one resolution.
        alpha : float
            Function-level miscoverage target: at most a alpha fraction of
            functions may have domain coverage below 1 - gamma.
        """

        if hasattr(self, "eta"):
            delattr(self, "eta")

        self.predictor_model.eval()
        self.quantile_model.eval()

        n = len(loader.dataset)
        scores = []

        for item in loader:
            a = item["x"].to(self.device)
            u = item["y"].to(self.device)

            u_pred = self.predictor_model(a)
            error_pred = self._radius(a).clamp_min(self.eps)
            error_true = pointwise_error(u, u_pred)

            # Compute score values for each function (current batch)
            ratio = (error_true / error_pred).flatten(start_dim=2)
            s = torch.quantile(
                ratio, 1.0 - self.gamma, dim=-1, interpolation="higher"
            )
            scores.extend(s.squeeze(1).tolist())

        scores.sort()

        k = int(np.ceil((n + 1) * (1 - alpha)))

        self.scale_factor = (
            scores[k - 1]
            if k <= n
            else float("inf")
        )


    @torch.no_grad()
    def ma_conformal_calibrate(self, loader, alpha: float, t_rule: str = "official"):
        """
        Compute the conformal scale factor (lambda-hat) on a calibration set. 
        Procedure is identical to the implementation in the official Neural 
        Operator library.
    
        t_rule : {"official", "min"}
            "official" -> t = lb + (gamma - lb)/3, with lb = sqrt(ln(1/alpha)/(2m))
            "min"      -> t = sqrt(ln(2/alpha)/(2m))  (minimal slack)
        """
        if hasattr(self, "eta"):
            delattr(self, "eta")
    
        self.alpha = alpha
        self.predictor_model.eval()
        self.quantile_model.eval()
    
        n = len(loader.dataset)
        scores = []
        t = None
    
        for item in loader:
            a = item["x"].to(self.device)
            u = item["y"].to(self.device)
    
            if t is None:
                m = math.prod(u.shape[2:]) # grid points per function
                lb = math.sqrt(-math.log(alpha) / (2 * m))  # t > sqrt(-ln delta / 2m)
    
                if t_rule == "official":
                    if self.gamma <= lb:
                        raise ValueError(
                            f"gamma={self.gamma} must exceed the resolution floor "
                            f"lb={lb:.6f} (m={m}, alpha={alpha}): increase gamma, "
                            "increase alpha, or use a finer discretisation"
                        )
                    t = lb + (self.gamma - lb) / 3.0
                elif t_rule == "min":
                    t = math.sqrt(math.log(2.0 / alpha) / (2 * m))
                    if t >= self.gamma:
                        raise ValueError(f"t={t:.6f} >= gamma={self.gamma}")
                else:
                    raise ValueError(f"unknown t_rule={t_rule!r}")
    
                domain_idx = int(math.ceil((self.gamma - t) * m))
    
            u_pred = self.predictor_model(a)
            error_pred = self._radius(a).clamp_min(self.eps)
            error_true = pointwise_error(u, u_pred)
    
            ratio = (error_true / error_pred).flatten(start_dim=1)   # (B, m)
            s = ratio.topk(domain_idx + 1, dim=1).values[:, -1]      # (B,)
            scores.extend(s.tolist())
    
        # function-level index, exactly as in get_coeff_quantile_idx
        delta_eff = alpha - math.exp(-2 * m * t * t)
        K = int(math.ceil((n + 1) * delta_eff))
        function_idx = max(K, 0)
    
        if function_idx + 1 > n:
            self.scale_factor = float("inf")
            return
    
        scores_desc = sorted(scores, reverse=True)
        self.scale_factor = abs(scores_desc[function_idx]) # (K+1)-th largest


    @staticmethod
    def _lipschitz_bound(p, domain_len=1.0):
        """
        Compute the Lipschitz constant for a discrete signal
        """
        dims     = tuple(range(2, p.ndim))
        spatial  = p.shape[2:]
        n_points = int(torch.tensor(spatial).prod())
        P = torch.fft.fftn(p, dim=dims) / n_points
        freqs = [torch.fft.fftfreq(n, d=1.0/n, device=p.device, dtype=p.dtype) for n in spatial]
        grids = torch.meshgrid(*freqs, indexing="ij")
        knorm = torch.sqrt(sum(g**2 for g in grids))
        coeff_norm = torch.sqrt((P.abs()**2).sum(dim=1))
        return (2*math.pi/domain_len) * (coeff_norm * knorm).flatten(1).sum(dim=1)   # (B,)

    @torch.no_grad()
    def continuum_conformal_calibrate(self, loader, alpha, eta, omega_h,
                                      eps_data=0.0, domain_len=1.0, 
                                      show_breakdown=False):
        """
        """
        if eta <= 0:
            raise ValueError("eta must be strictly positive")
        self.predictor_model.eval(); self.quantile_model.eval()
        self.eta = eta

        if show_breakdown:
            s_tilde_sum = 0.0
            d2 = d3 = 0.0

        n, scores = len(loader.dataset), []
        for item in loader:
            a = item["x"].to(self.device)
            u = item["y"].to(self.device)

            d = u.ndim - 2
            N = u.shape[-1] # grid resolution
            h = domain_len * math.sqrt(d) / (2 * N) # fill distance

            u_pred = self.predictor_model(a)
            error_forward = self._radius(a)
            error_pred = error_forward.clamp_min(eta)
            error_true = pointwise_error(u, u_pred)

            # per-sample node quantities
            max_node_error = error_true.flatten(1).max(dim=1).values
            residuals = (error_true / error_pred).flatten(1)
            s_tilde = torch.quantile(residuals, 1 - self.gamma, dim=1,
                                     interpolation="higher")

            L_u = self._lipschitz_bound(u_pred, domain_len)
            L_e = self._lipschitz_bound(error_forward, domain_len)

            R = max_node_error + eps_data + omega_h + L_u * h
            Delta = (omega_h + L_u*h)/eta + (R*L_e*h)/eta**2 + eps_data/eta

            scores.extend((s_tilde + Delta).tolist())

            if show_breakdown:
                B = a.shape[0]
                s_tilde_sum += s_tilde.sum().item()
                d2 += ((L_u * h) / eta).sum().item()
                d3 += ((R * L_e * h) / eta**2).sum().item()

        scores.sort()
        k = math.ceil((n + 1) * (1 - alpha))
        self.scale_factor = scores[k-1] if k <= n else float("inf")

        if show_breakdown:
            return {
                "s_tilde":      s_tilde_sum / n,
                "delta_1":      omega_h /eta,
                "delta_2":      d2 / n,
                "delta_3":      d3 / n,
                "delta_4":      eps_data / eta,
                "scale_factor": self.scale_factor,
            }


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
            mean_pred +- radius with the calibrated (gamma, alpha) guarantee.
        """
        min_error = self.eta if hasattr(self, "eta") else self.eps

        pred = self.predictor_model(a)
        radius = self._radius(a).clamp_min(min_error) * self.scale_factor

        return (pred, radius)

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
              coverage >= 1 - gamma (the guarantee targets >= 1 - alpha)
            - "mean_bandwidth": predicted ball radius averaged over all
              points and functions
        """
        self.predictor_model.eval()
        self.quantile_model.eval()

        coverages, bandwidths = [], []

        for item in loader:
            a = item["x"].to(self.device)
            u = item["y"].to(self.device)

            u_pred, radius = self.predict_interval(a)
            error_true = pointwise_error(u, u_pred)

            covered = (error_true <= radius).float().flatten(start_dim=2)
            coverages.extend(covered.mean(dim=-1).squeeze(1).tolist())
            bandwidths.extend(radius.flatten(start_dim=1).mean(dim=-1).tolist())

        coverages = np.asarray(coverages)

        return {
            "coverage": coverages,
            "calibration_percentage": float(
                (coverages >= 1.0 - self.gamma).mean()
            ),
            "mean_bandwidth": float(np.mean(bandwidths)),
        }

    def _radius(self, a):
        """Quantile model output in physical units."""
        return self.quantile_model(a) * self.residual_scale