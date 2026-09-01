import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FIGURE_DIR = REPO_ROOT / "report" / "figures"
CHECKPOINT_DIR = REPO_ROOT / "experiments" / "pretrained"

DATA_ROOT = Path(
    os.environ.get("CONFORMAL_NO_DATA", "~/Datasets/conformal_no")
).expanduser()


def dataset_path(name):
    """
    Return the full path to a dataset file.

    Parameters
    ----------
    name : str
        Filename or relative path under DATA_ROOT,
        e.g. "darcy_421/darcy_train_421.pt".

    Returns
    -------
    pathlib.Path
    """
    path = DATA_ROOT / name
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{name}' at {path}.\n\n"
            f"Place the datasets in {DATA_ROOT}, or set CONFORMAL_NO_DATA:\n"
            f"    export CONFORMAL_NO_DATA=/your/path\n\n"
            f"See data/README.md for download links."
        )
    return path