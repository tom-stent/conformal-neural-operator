import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from conformal_no.utils.error import pointwise_error

SOLUTION_CMAP = "viridis"
ERROR_CMAP = "afmhot"
COVERAGE_CMAP = "binary_r"


def _cbar(fig, im, ax=None, cax=None, ticks=None):
    """Attach a colourbar styled to match the surrounding rcParams."""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    kw = dict(cax=cax) if cax is not None else dict(ax=ax, fraction=0.046, pad=0.02)
    cb = fig.colorbar(im, ticks=ticks if ticks is not None else MaxNLocator(3),
                      format=fmt, **kw)
    labelsize = mpl.rcParams["ytick.labelsize"] - 1
    cb.ax.tick_params(labelsize=labelsize, length=1.5,
                      width=mpl.rcParams["ytick.major.width"])
    cb.ax.yaxis.get_offset_text().set_fontsize(labelsize)
    cb.outline.set_linewidth(mpl.rcParams["axes.linewidth"])
    return cb


def visualise(loader, model, scale_factor=None, idx=None, seed=None,
              tie_error_to_solution=False, separate_input_scale=False,
              error_vmax_scale=1.0, verbose=False):
    """Plot one sample's input, solution, prediction, error and coverage.

    Six panels in a 2 x 3 grid. The true solution and prediction always
    share a colour scale so they can be compared directly; the true and
    predicted error share a second scale for the same reason.

    Parameters
    ----------
    loader : DataLoader
        The sample is drawn from the first batch. Inputs and targets must
        be at the same resolution.
    model : ConformalQuantileRegressor
        Calibrated model. Its scale factor is restored on exit.
    scale_factor : float, optional
        Override the calibrated factor for this figure only.
    idx : int, optional
        Index within the first batch. Chosen at random if omitted.
    seed : int, optional
    tie_error_to_solution : bool, default False
        Force the error panels onto the solution scale.
    separate_input_scale : bool, default False
        Give the input its own scale and colourbar, dropping it from the
        solution scale.
    error_vmax_scale : float, default 1.0
        Multiplies the error vmax, so values above 1 darken the error
        panels. Ignored when the error is tied to the solution scale.
    verbose : bool, default False
        Print RMS values for each field.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If error_vmax_scale is not positive, or if the loader's targets
        and the model's predictions differ in resolution.
    """
    if error_vmax_scale <= 0:
        raise ValueError("error_vmax_scale must be positive, "
                         f"got {error_vmax_scale}")

    old_scale_factor = model.scale_factor
    if scale_factor is not None:
        model.scale_factor = scale_factor

    try:
        batch = next(iter(loader))
        if idx is None:
            idx = int(np.random.default_rng(seed).integers(batch["x"].shape[0]))
        a = batch["x"][idx:idx + 1]
        u = batch["y"][idx:idx + 1]

        u_pred, radius = model.predict_interval(a)
        if u.shape != u_pred.shape:
            raise ValueError(
                f"target {tuple(u.shape)} and prediction {tuple(u_pred.shape)} "
                "differ; visualise expects a matched-resolution loader"
            )

        error_true = pointwise_error(u, u_pred)
        covered = (error_true <= radius).float()

        img = lambda t: t[0, 0].detach().cpu()

        # true solution and prediction always share one scale, the input joins
        # them unless it has been given a scale of its own
        if separate_input_scale:
            sol = dict(cmap=SOLUTION_CMAP,
                       vmin=float(min(u.min(), u_pred.min())),
                       vmax=float(max(u.max(), u_pred.max())))
            inp = dict(cmap=SOLUTION_CMAP,
                       vmin=float(a.min()), vmax=float(a.max()))
        else:
            sol = dict(cmap=SOLUTION_CMAP,
                       vmin=float(min(a.min(), u.min(), u_pred.min())),
                       vmax=float(max(a.max(), u.max(), u_pred.max())))
            inp = sol

        err = dict(cmap=ERROR_CMAP, vmin=0.0,
                   vmax=float(max(error_true.max(), radius.max()))
                   * error_vmax_scale)
        if tie_error_to_solution:
            err = dict(cmap=ERROR_CMAP, vmin=sol["vmin"], vmax=sol["vmax"])

        fig = plt.figure(figsize=(4.8, 3.0), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.01, wspace=0.0)

        gs = fig.add_gridspec(2, 8,
                              width_ratios=[0.5, 0.04, 0.05,   # input  | cbar | gap
                                            0.5, 0.04, 0.001,  # sol    | cbar | gap
                                            0.5, 0.04])        # err    | cbar

        axes = np.array([[fig.add_subplot(gs[r, c]) for c in (0, 3, 6)]
                         for r in range(2)])
        cax_cov = fig.add_subplot(gs[1, 1])
        cax_in = fig.add_subplot(gs[0, 1]) if separate_input_scale else None
        cax_sol = fig.add_subplot(gs[:, 4])
        cax_err = fig.add_subplot(gs[:, 7])

        im_in = axes[0, 0].imshow(img(a), **inp)
        axes[0, 0].set_title("Input")

        im_sol = axes[0, 1].imshow(img(u), **sol)
        axes[0, 1].set_title("True solution")

        im_err = axes[0, 2].imshow(img(error_true), **err)
        axes[0, 2].set_title("True error")

        im_cov = axes[1, 0].imshow(img(covered), vmin=0, vmax=1,
                                   cmap=COVERAGE_CMAP)
        axes[1, 0].set_title(f"Containment ({covered.mean():.1%})", fontsize=9.3)

        axes[1, 1].imshow(img(u_pred), **sol)
        axes[1, 1].set_title("Predicted solution", fontsize=10.6)

        axes[1, 2].imshow(img(radius), **err)
        axes[1, 2].set_title("Predicted error")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        _cbar(fig, im_cov, cax=cax_cov, ticks=[0, 1])
        if cax_in is not None:
            _cbar(fig, im_in, cax=cax_in)
        _cbar(fig, im_sol, cax=cax_sol)
        _cbar(fig, im_err, cax=cax_err)

        if verbose:
            rms = lambda t: float(torch.sqrt(torch.mean(t ** 2)))
            print(f"RMS  true solution {rms(u):.4g}  "
                  f"predicted solution {rms(u_pred):.4g}")
            print(f"RMS  true error {rms(error_true):.4g}  "
                  f"predicted error {rms(radius):.4g}")

        return fig

    finally:
        model.scale_factor = old_scale_factor