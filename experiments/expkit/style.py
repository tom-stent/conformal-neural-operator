import matplotlib as mpl

# 4.8 inches ~ 345pt LaTeX default.
FIGSIZE = (4.5, 3.0)

COLOURS = {
    "c1": "#0e33ba",
    "c2": "#A73D00FF",
    "c3": "#C05300FF",
    "c4": "#F0E442",
    "c5": "#56CEE9",
    "c6": "#5a5a5a"
}

LINESTYLES = {
    "Ours":       dict(color=COLOURS["c1"], marker="o", markersize=3, linestyle="-", linewidth=0.8),
    "Ma et al.":  dict(color=COLOURS["c2"], marker="o", markersize=3, linestyle="-", linewidth=0.8),
    "Continuum":  dict(color=COLOURS["c5"], marker="o", markersize=3, markerfacecolor="none", linewidth=0.8),
    "target":  dict(color=COLOURS["c6"], linestyle="--", linewidth=0.7),
    "reference":  dict(color=COLOURS["c6"], linestyle="--", linewidth=0.7)
}

PLOT_ALPHA = 0.7

def set_style(fontsize=11):
    mpl.rcParams.update({
        # Computer Modern for LaTeX
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,  # keeps minus signs in Computer Modern

        # Text sizes
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "legend.fontsize": fontsize - 3,
        "xtick.labelsize": fontsize - 3,
        "ytick.labelsize": fontsize - 3,

        # Size and quality
        "figure.figsize": FIGSIZE,
        "figure.dpi": 150, # on-screen in the notebook
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, # embed fonts properly

        # Looks
        "axes.grid": False,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
    })




r"""
Once, in the first cell of a notebook:
 
    import matplotlib.pyplot as plt
    from plotting.style import set_style
    set_style()
 
Pass the font size of the document if it is not 11pt, e.g. set_style(12).
 
A single figure:
Do not pass figsize as set_style already provides one sized for the page.
 
    fig, ax = plt.subplots()
    ax.plot(resolutions, errors, "o-")
    ax.set_xlabel("...")
    ax.set_ylabel("...")
    fig.savefig("...")
 
Multiple subplots:
This is the one case where figsize is passed explicitly, because the default
is sized for a single panel.
 
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.8), constrained_layout=True)
 
    ...
 
    fig.savefig("results/figures/darcy_fields.pdf")
 
Keep the width at 4.8 inches in every figure. That is the text width of the
document, and a wider figure is simply scaled down on insertion, which shrinks
the fonts with it. Vary only the height to suit the grid:
 
    1x1   omit figsize
    1x2   (4.8, 2.2)
    1x3   (4.8, 1.8)
    2x2   (4.8, 3.8)
    2x3   (4.8, 3.0)
 
Use constrained_layout=True for any grid, or axis labels will overlap the
neighbouring panel.
"""