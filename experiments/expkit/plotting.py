import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import betabinom

from experiments.expkit.paths import FIGURE_DIR
from experiments.expkit.style import COLOURS, LINESTYLES, PLOT_ALPHA

DEFAULT_COLOURS = {"Ours": COLOURS["c1"], "Ma et al.": COLOURS["c2"]}
DEFAULT_METHODS = ("Ours", "Ma et al.")


# ---------------------------------------------------------------- helpers

def save_figure(fig, name, subdir):
    """
    Save a figure as PDF into report/figures/<subdir>/.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Filename without extension, e.g. "violins_darcy".
    subdir : str
        Subdirectory, e.g. "darcy" or "navier".
    """
    out_dir = FIGURE_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path.relative_to(FIGURE_DIR.parents[1])}")


def _prepare_ax(ax, **subplot_kw):
    """Return (fig, ax), creating a new figure only if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(**subplot_kw)
    else:
        fig = ax.figure
    return fig, ax


def _resolutions_in(results_df, resolutions=None):
    if resolutions is None:
        return sorted(results_df["resolution"].unique())
    return list(resolutions)


def _cell(results_df, method, resolution, column):
    """Extract a single cell from the results DataFrame as an array."""
    mask = (results_df.method == method) & (results_df.resolution == resolution)
    return np.asarray(results_df.loc[mask, column].iloc[0])


def betabinom_reference(n_cal, n_test, alpha):
    """Exact reference law for the coverage histogram."""
    k = math.ceil((n_cal + 1) * (1.0 - alpha))
    dist = betabinom(n_test, k, n_cal + 1 - k)
    x = np.arange(n_test + 1)
    return x, dist.pmf(x), dist


# ------------------------------------------------------- line/error plots


def plot_resampled_lambda(results_df, resolutions=None, column="resampled lambda",
                          percentiles=(5, 50, 95), xlim=None, ylim=None,
                          yticks=None, ax=None):
    """
    Median and interval of the resampled scaling factor, against resolution.

    Draws the median with asymmetric error bars denoting the requested
    percentiles of the resampled draws. Non-finite draws, where the rule is
    undefined, are dropped.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Contains "method", "resolution" and column, with column holding an array 
        of resampled values per row.
    percentiles : tuple of float, default (5, 50, 95)
        Lower, central and upper percentiles.
    xlim, ylim : tuple, optional
    yticks : sequence, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare_ax(ax)
    resolutions = _resolutions_in(results_df, resolutions)
    p_lo, p_mid, p_hi = percentiles

    for method, g in results_df.groupby("method"):
        g = g.sort_values("resolution")

        res_vals, med, lo, hi = [], [], [], []
        for _, row in g.iterrows():
            lam = np.asarray(row[column], dtype=float)
            lam = lam[np.isfinite(lam)]
            if lam.size == 0:
                continue
            q_lo, q_mid, q_hi = np.percentile(lam, [p_lo, p_mid, p_hi])
            res_vals.append(row["resolution"])
            med.append(q_mid)
            lo.append(q_lo)
            hi.append(q_hi)

        if not res_vals:
            continue

        res_vals = np.asarray(res_vals, dtype=float)
        med, lo, hi = map(np.asarray, (med, lo, hi))
        yerr = np.vstack([med - lo, hi - med])

        style = dict(LINESTYLES[method])
        marker = style.pop("marker", style.pop("m", "o"))
        markersize = style.pop("markersize", style.pop("ms", 3.5))
        ax.errorbar(res_vals, med, yerr=yerr, label=method,
                    capsize=3, elinewidth=0.5, capthick=0.7,
                    marker=marker, markersize=markersize, **style)

    ax.set_xlabel(r"Resolution $N$")
    ax.set_ylabel(r"Calibrated scaling factor $\hat{\lambda}$")
    ax.set_xticks(resolutions)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.legend(frameon=False)
    return fig


def plot_coarse_fine(results_df, alpha, resolutions=None,
                     interpolated_column="coverage interpolated",
                     zero_shot_column="coverage zero-shot",
                     xlim=None, ylim=(-0.05, 1.05),
                     figsize=(4.95, 2.0), axes=None):
    """
    High-resolution coverage under spectral interpolation and zero-shot
    super-resolution, side by side.

    Parameters
    ----------
    results_df : pandas.DataFrame
    alpha : float
        Target miscoverage level, a reference line is drawn at 1 - alpha.
    axes : sequence of two Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        created = True
    else:
        fig = axes[0].figure
        created = False

    resolutions = _resolutions_in(results_df, resolutions)

    for method, g in results_df.groupby("method"):
        g = g.sort_values("resolution")
        axes[0].plot(g.resolution, g[interpolated_column],
                     label=method, **LINESTYLES[method])
        axes[1].plot(g.resolution, g[zero_shot_column],
                     label=method, **LINESTYLES[method])

    titles = ["Spectral interpolation", "Zero-shot super-resolution"]
    for ax, title in zip(axes, titles):
        ax.axhline(1 - alpha, **LINESTYLES["reference"])
        ax.set_title(title)
        ax.set_xlabel(r"Resolution $N$")
        ax.set_ylabel("High-res empirical coverage", fontsize=8)
        ax.set_xticks(resolutions)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)

    if created:
        fig.set_constrained_layout_pads(wspace=0.3)
    return fig


# --------------------------------------------------- distribution plots

def plot_coverage_histogram(results_df, resolution, n_cal, n_test, alpha,
                            column="coverage counts", methods=DEFAULT_METHODS, 
                            colours=None, bin_width=5, xlim=None, ylim=None, 
                            xticks=None, yticks=None, reference=True, 
                            plot_alpha=True, legend_fontsize=8, ax=None):
    """
    Histogram of resampled coverage at one resolution, with the exact
    Beta-Binomial reference.

    Bars and reference are drawn as densities in fraction units, so the
    vertical scale does not depend on bin_width.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain `column`, holding per-resample counts in 0..n_test.
    resolution : int
        Which resolution to show.
    n_cal, n_test : int
        Calibration and test sizes used in the resampling.
    alpha : float
    reference : bool
        Draw the Beta-Binomial law implied by an exactly calibrated split
        conformal procedure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare_ax(ax)
    colours = colours or DEFAULT_COLOURS

    edges = np.arange(0, n_test + bin_width + 1, bin_width) - 0.5
    centres = (edges[:-1] + edges[1:]) / 2 / n_test
    bin_frac = bin_width / n_test

    all_counts = []
    for method in methods:
        counts = _cell(results_df, method, resolution, column)
        all_counts.append(counts)
        hist, _ = np.histogram(counts, bins=edges)
        ax.bar(centres, hist / len(counts) / bin_frac, width=bin_frac,
               alpha=PLOT_ALPHA, label=method, color=colours.get(method),
               zorder=2)

    x, pmf, dist = betabinom_reference(n_cal, n_test, alpha)
    if reference:
        ax.plot(x / n_test, pmf * n_test, color="k", lw=0.8, zorder=4,
                label="Beta-binomial (exact)")

    if plot_alpha:
        ax.axvline(1.0 - alpha, label=r"$1-\alpha$", zorder=3,
                   **LINESTYLES["target"])

    if xlim is None:
        obs = np.concatenate(all_counts)
        lo = min(dist.ppf(1e-5), obs.min()) / n_test
        hi = max(dist.ppf(1.0 - 1e-5), obs.max()) / n_test
        pad = 0.02 * (hi - lo + 1e-12)
        xlim = (max(0.0, lo - pad), min(1.0, hi + pad))
    ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xlabel("Resampled empirical coverage")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=legend_fontsize)
    return fig


def plot_coverage_violins(results_df, n_test, alpha,
                          column="coverage counts",
                          methods=DEFAULT_METHODS, resolutions=None,
                          colours=None, width=0.22, offset=False,
                          legend_shift=0.0, ax=None, yticks=None):
    """
    Per-resolution distribution of resampled coverage, one violin per
    method per resolution.

    Plots the same quantity as plot_coverage_histogram, summarised across
    resolutions rather than resolved at one.

    Parameters
    ----------
    offset : bool, default False
        Nudge each method's violins apart horizontally.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare_ax(ax)
    colours = colours or DEFAULT_COLOURS
    resolutions = _resolutions_in(results_df, resolutions)

    x = np.arange(len(resolutions))
    if offset and len(methods) > 1:
        offs = np.linspace(-1, 1, len(methods)) * width * 0.6
    else:
        offs = np.zeros(len(methods))

    for i, (method, off) in enumerate(zip(methods, offs)):
        data = [_cell(results_df, method, r, column) / n_test
                for r in resolutions]
        vp = ax.violinplot(data, positions=x + off, widths=width,
                           showmeans=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colours.get(method, f"C{i}"))
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
            body.set_alpha(PLOT_ALPHA)
        vp["cmeans"].set_color("black")
        vp["cmeans"].set_linewidth(0.5)

    target = ax.axhline(1 - alpha, zorder=0, label=r"$1-\alpha$",
                        **LINESTYLES["target"])

    handles = [mpatches.Patch(facecolor=colours.get(m, f"C{i}"),
                              alpha=PLOT_ALPHA, edgecolor="black",
                              linewidth=0.0, label=m)
               for i, m in enumerate(methods)]
    ax.legend(handles=handles + [target], frameon=False, fontsize=8,
              loc="lower right", bbox_to_anchor=(1.0 - legend_shift, 0.0))

    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel(r"Resolution $N$")
    ax.set_ylabel("Resampled empirical coverage")
    if yticks is not None:
        ax.set_yticks(yticks)
    return fig