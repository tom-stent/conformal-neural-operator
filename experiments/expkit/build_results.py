"""
Assemble the results DataFrame for one experiment.
"""

import pandas as pd

from experiments.expkit.coarse_fine import interp_continuum_calibration
from experiments.expkit.coverage_resampling import (
    compute_function_scores,
    compute_ma_function_scores,
    resample_coverage_counts_paired,
)


def build_results(models, loaders, combined_loaders, interp_loaders,
                  finest_loader, configs, resolutions, n_cal_resample,
                  n_resamples, seed=0, verbose=True):
    """Compute results_df used in all figure plots. One row per 
    (method, resolution) pair.

    Parameters
    ----------
    models : dict
        Maps (method, resolution) to a calibrated UQ-FNO model.
    loaders : dict
        Maps resolution to a dict with a "test" key, at the model's own
        resolution.
    combined_loaders : dict
        Maps resolution to a loader over the calibration and test sets
        concatenated. Must not shuffle, as the two score arrays are aligned
        by sample order.
    interp_loaders : dict
        Maps resolution to a loader with coarse inputs and fine targets.
    finest_loader : DataLoader
        Inputs and targets both on the fine grid, for zero-shot evaluation.
    configs : dict
        Maps resolution to a config dict, read for "alpha" and "gamma".
    resolutions : sequence of int
    n_cal_resample : int
        Calibration set size drawn in each resample.
    n_resamples : int
        Number of resampled calibration/test splits.
    seed : int, default 0
        Fixed seed for the resampling permutations.
    verbose : bool, default True
        Print progress; the resampling stage is slow.

    Returns
    -------
    pandas.DataFrame
        Columns: method, resolution, mean bandwidth, scale factor,
        coverage zero-shot, coverage interpolated, coverage counts,
        resampled lambda.
    """
    single = _single_split_summaries(models, loaders, verbose)
    resampled = _resampled_coverage(
        models, combined_loaders, configs, resolutions,
        n_cal_resample, n_resamples, seed, verbose,
    )
    coarse_fine = _coarse_fine_coverage(
        models, interp_loaders, finest_loader, configs, verbose
    )

    results_df = (
        single
        .merge(resampled, on=["method", "resolution"], how="left")
        .merge(coarse_fine, on=["method", "resolution"], how="left")
    )
    return results_df.sort_values(["method", "resolution"]).reset_index(drop=True)


def _single_split_summaries(models, loaders, verbose):
    """Mean bandwidth and calibrated scale factor on the single split."""
    rows = []
    for (method, res), model in models.items():
        rows.append({
            "method": method,
            "resolution": res,
            "mean bandwidth": model.evaluate(loaders[res]["test"])["mean_bandwidth"],
            "scale factor": model.scale_factor,
        })
        if verbose:
            print(f"[summary] {method}, resolution {res}")
    return pd.DataFrame(rows)


def _resampled_coverage(models, combined_loaders, configs, resolutions,
                        n_cal_resample, n_resamples, seed, verbose):
    """Paired resampling of coverage counts over calibration/test splits.

    Both methods see the same permutations at a given resolution, so their
    counts are comparable draw by draw. The seed is offset per resolution
    so the resolutions are independent of one another but each is fixed.
    """
    rows = []
    for i, res in enumerate(resolutions):
        alpha = configs[res]["alpha"]

        our_scores = compute_function_scores(
            models[("Ours", res)], combined_loaders[res]
        )
        ma_scores, delta_eff = compute_ma_function_scores(
            models[("Ma et al.", res)], combined_loaders[res], alpha=alpha
        )

        counts_our, counts_ma, lam_our, lam_ma = resample_coverage_counts_paired(
            our_scores, ma_scores, delta_eff,
            n_cal=n_cal_resample, alpha=alpha,
            T=n_resamples, seed=seed + i,
        )

        rows.append({"method": "Ours", "resolution": res,
                     "coverage counts": counts_our,
                     "resampled lambda": lam_our})
        rows.append({"method": "Ma et al.", "resolution": res,
                     "coverage counts": counts_ma,
                     "resampled lambda": lam_ma})
        if verbose:
            print(f"[resampling] resolution {res}")
    return pd.DataFrame(rows)


def _coarse_fine_coverage(models, interp_loaders, finest_loader, configs,
                          verbose):
    """Coverage on the fine grid, by zero-shot evaluation and by spectral
    interpolation of the coarse prediction."""
    rows = []
    for (method, res), model in models.items():
        rows.append({
            "method": method,
            "resolution": res,
            "coverage zero-shot":
                model.evaluate(finest_loader)["calibration_percentage"],
            "coverage interpolated":
                interp_continuum_calibration(
                    model, interp_loaders[res], gamma=configs[res]["gamma"]
                ),
        })
        if verbose:
            print(f"[coarse-fine] {method}, resolution {res}")
    return pd.DataFrame(rows)