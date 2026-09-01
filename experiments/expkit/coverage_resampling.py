import math
import warnings

from experiments.expkit.style import COLOURS, LINESTYLES, PLOT_ALPHA

import numpy as np
import torch

from conformal_no.utils.error import pointwise_error


@torch.no_grad()
def compute_function_scores(uqfno, loader):
    """
    Per-function nonconformity scores under our calibration rule.

    Parameters
    ----------
    uqfno : ConformalQuantileRegressor
    loader : DataLoader
        Must not shuffle if the scores are to be aligned sample-by-sample with 
        those from compute_ma_function_scores.

    Returns
    -------
    ndarray of shape (n_samples,)
        Scores in the loader's sample order.
    """
    uqfno.predictor_model.eval()
    uqfno.quantile_model.eval()

    scores = []
    for item in loader:
        a = item["x"].to(uqfno.device)
        u = item["y"].to(uqfno.device)

        u_pred = uqfno.predictor_model(a)
        error_pred = uqfno._radius(a).clamp_min(uqfno.eps)
        error_true = pointwise_error(u, u_pred)

        ratio = (error_true / error_pred).flatten(start_dim=2).squeeze(1)
        m = ratio.shape[-1]
        rank = math.ceil((1.0 - uqfno.gamma) * m)
        s = ratio.kthvalue(rank, dim=-1).values
        scores.extend(s.tolist())

    scores = np.asarray(scores, dtype=np.float64)
    return scores


@torch.no_grad()
def compute_ma_function_scores(uqfno, loader, alpha, t_rule="official"):
    """
    Per-function scores under the Ma et al. convention exactly.

    Parameters
    ----------
    uqfno : ConformalQuantileRegressor
    loader : DataLoader
    alpha : float
        Target miscoverage level in (0, 1).
    t_rule : {"official", "min"}, default "official"
        How the concentration term is chosen. "official" reproduces the 
        published choice, one third of the way from the resolution floor to 
        gamma. "min" takes the smallest admissible t.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
    delta_eff : float
        Miscoverage remaining for the conformal step after the correction.
    """
    uqfno.predictor_model.eval()
    uqfno.quantile_model.eval()

    scores = []
    delta_eff = None
    for item in loader:
        a = item["x"].to(uqfno.device)
        u = item["y"].to(uqfno.device)

        if delta_eff is None:
            m = math.prod(u.shape[2:])
            lb = math.sqrt(-math.log(alpha) / (2 * m))
            if t_rule == "official":
                if uqfno.gamma <= lb:
                    raise ValueError(
                        f"gamma={uqfno.gamma} <= resolution floor lb={lb:.6f} "
                        f"(m={m}): Ma et al. undefined at this resolution"
                    )
                t_corr = lb + (uqfno.gamma - lb) / 3.0
            elif t_rule == "min":
                t_corr = math.sqrt(math.log(2.0 / alpha) / (2 * m))
                if t_corr >= uqfno.gamma:
                    raise ValueError(f"t={t_corr:.6f} >= gamma={uqfno.gamma}")
            else:
                raise ValueError(f"unknown t_rule={t_rule!r}")
            domain_idx = int(math.ceil((uqfno.gamma - t_corr) * m))
            delta_eff = alpha - math.exp(-2 * m * t_corr * t_corr)

        u_pred = uqfno.predictor_model(a)
        error_pred = uqfno._radius(a).clamp_min(uqfno.eps)
        error_true = pointwise_error(u, u_pred)

        ratio = (error_true / error_pred).flatten(start_dim=1)
        s = ratio.topk(domain_idx + 1, dim=1).values[:, -1]
        scores.extend(s.tolist())

    return np.asarray(scores, dtype=np.float64), delta_eff


def resample_coverage_counts_paired(our_scores, ma_scores, delta_eff, n_cal,
                                    alpha, T=2000, seed=0):
    """
    Resample calibration/test splits and count covered test functions.

    Parameters
    ----------
    our_scores, ma_scores : array_like of shape (n,)
        Sample-aligned score arrays from the two scoring functions.
    delta_eff : float
        Effective miscoverage from compute_ma_function_scores.
    n_cal : int
        Calibration size drawn per resample.
    alpha : float
        Target miscoverage level in (0, 1).
    T : int, default 2000
        Number of resamples.
    seed : int, default 0
        Seeds a local generator, so the result does not depend on global numpy 
        state.

    Returns
    -------
    counts_ours, counts_ma : ndarray of shape (T,), int
        Covered test functions per resample.
    lambdas_ours, lambdas_ma : ndarray of shape (T,), float
        Calibrated scale factors per resample. 
    """
    our_scores = np.asarray(our_scores, dtype=np.float64)
    ma_scores = np.asarray(ma_scores, dtype=np.float64)
    if our_scores.shape != ma_scores.shape:
        raise ValueError("score arrays must be sample-aligned (same length/order)")

    n = len(our_scores)
    n_test = n - n_cal
    k = math.ceil((n_cal + 1) * (1.0 - alpha))
    K = max(int(math.ceil((n_cal + 1) * delta_eff)), 0)
    ma_defined = (K + 1) <= n_cal

    rng = np.random.default_rng(seed)
    counts_ours = np.empty(T, dtype=np.int64)
    counts_ma = np.empty(T, dtype=np.int64)
    lambdas_ours = np.empty(T, dtype=np.float64)
    lambdas_ma = np.empty(T, dtype=np.float64)

    for t in range(T):
        perm = rng.permutation(n)
        cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

        lam = np.partition(our_scores[cal_idx], k - 1)[k - 1]
        lambdas_ours[t] = lam
        counts_ours[t] = int((our_scores[test_idx] <= lam).sum())

        if ma_defined:
            # (K+1)-th largest == (n_cal - K)-th smallest, 0-indexed n_cal-1-K
            lam_ma = np.partition(ma_scores[cal_idx], n_cal - 1 - K)[n_cal - 1 - K]
            lambdas_ma[t] = lam_ma
            counts_ma[t] = int((our_scores[test_idx] <= lam_ma).sum())
        else:
            lambdas_ma[t] = np.inf
            counts_ma[t] = n_test

    return counts_ours, counts_ma, lambdas_ours, lambdas_ma