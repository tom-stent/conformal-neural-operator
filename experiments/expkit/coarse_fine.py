import torch
import numpy as np
from conformal_no.utils.field_utils import fourier_interpolate
from conformal_no.utils.error import pointwise_error


@torch.no_grad()
def interp_continuum_calibration(uqfno, loader, gamma):
    """
    Fraction of functions whose interpolated bands sufficiently cover the fine 
    solution.

    The coarse prediction and radius are extended to the fine grid by Fourier 
    interpolation, which is the FNO's own continuum extension of its grid 
    output.

    Parameters
    ----------
    model : ConformalQuantileRegressor
        Calibrated model. Both submodels are switched to eval mode here.
    loader : DataLoader
        Yields "x" at the model's resolution and "y" at a finer one.
    gamma : float
        Pointwise miscoverage level. A function counts as covered when at
        least 1 - gamma of its fine-grid points lie inside the band.

    Returns
    -------
    float
        Proportion of functions covered, in [0, 1].
    """
    uqfno.predictor_model.eval()
    uqfno.quantile_model.eval()

    coverages = []

    for item in loader:
        a = item["x"].to(uqfno.device)
        u_fine = item["y"].to(uqfno.device)
        fine_shape = list(u_fine.shape[2:])

        u_pred, radius = uqfno.predict_interval(a)
        u_interp = fourier_interpolate(u_pred, fine_shape)
        r_interp = fourier_interpolate(radius, fine_shape)

        r_interp = r_interp.clamp_min(uqfno.eps)

        error_true = pointwise_error(u_fine, u_interp)
        covered = (error_true <= r_interp).float().flatten(start_dim=2)
        coverages.extend(covered.mean(dim=-1).squeeze(1).tolist())
        
    coverages = np.asarray(coverages)
    return float((coverages >= 1.0 - gamma).mean())