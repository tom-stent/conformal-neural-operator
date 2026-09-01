import h5py
import numpy as np
import torch
from scipy.io import loadmat

from conformal_no.data.datasets import add_channel
from experiments.expkit.paths import dataset_path


def load_darcy(drop_last=True):
    """
    Darcy flow at 421 x 421, pre-split into train and test files.

    Parameters
    ----------
    drop_last : bool, default True
        Drop the duplicated boundary node, leaving 420 points spanning [0, 1).

    Returns
    -------
    dict
        Maps "train" and "test" to (x, y) pairs of shape (N, 1, H, W).
    """
    out = {}
    for split, fname in [("train", "darcy_421/darcy_train_421.pt"),
                         ("test",  "darcy_421/darcy_test_421.pt")]:
        d = torch.load(dataset_path(fname))
        x, y = d["x"].float(), d["y"].float()
        if drop_last:
            x, y = x[:, :-1, :-1], y[:, :-1, :-1]
        out[split] = (add_channel(x, 2), add_channel(y, 2))
    return out


def load_navier_stokes(input_time=10, output_time=20,
                       n_train=9000, n_test=1000, field="u"):
    """
    2D Navier-Stokes on the torus: the u(t_in) -> u(t_out) operator.

    Parameters
    ----------
    input_time, output_time : int
        Time indices.
    n_train, n_test : int
        Split sizes, taken in file order from the front.
    field : str, default "u"

    Returns
    -------
    dict
        Maps "train" and "test" to (x, y) pairs of shape (N, 1, H, W).
    """
    path = dataset_path("ns_V1e-4_N10000_T30.mat")

    with h5py.File(path, "r") as f:
        U = f[field]
        n_times, ny, nx, _ = U.shape
        assert ny == nx, f"expected a square grid, got {ny} x {nx}"
        for t in (input_time, output_time):
            assert 0 <= t < n_times, f"time {t} outside 0..{n_times - 1}"

        x = np.ascontiguousarray(np.asarray(U[input_time]).transpose(2, 0, 1))
        y = np.ascontiguousarray(np.asarray(U[output_time]).transpose(2, 0, 1))

    x = add_channel(torch.from_numpy(x).float(), 2)
    y = add_channel(torch.from_numpy(y).float(), 2)

    assert n_train + n_test <= x.shape[0]
    return {
        "train": (x[:n_train], y[:n_train]),
        "test":  (x[n_train:n_train + n_test], y[n_train:n_train + n_test]),
    }


def load_ns_highres(t_in, t_out, field="u", time_key="t"):
    """
    High-resolution Navier-Stokes test set at 256 x 256.

    Parameters
    ----------
    t_in, t_out : float
        Physical times, matched to the nearest recorded time. This file uses a 
        different time grid to the training file, so times cannot be matched by 
        index.
    field, time_key : str
        Variable names in the MATLAB file.

    Returns
    -------
    tuple of Tensor
        (x, y), each of shape (N, 1, 256, 256).
    """
    path = dataset_path("ns_data_V1e-4_N20_T50_R256test.mat")

    d = loadmat(path, variable_names=[field, time_key])
    U = np.asarray(d[field])  # logical order: no transpose
    t = np.asarray(d[time_key]).squeeze()

    n_sample, nx, ny, n_time = U.shape
    assert nx == ny and len(t) == n_time

    i_in = int(np.argmin(np.abs(t - t_in)))
    i_out = int(np.argmin(np.abs(t - t_out)))

    x = np.ascontiguousarray(U[:, :, :, i_in].transpose(0, 2, 1))
    y = np.ascontiguousarray(U[:, :, :, i_out].transpose(0, 2, 1))
    del d, U

    return (torch.from_numpy(x).float().unsqueeze(1),
            torch.from_numpy(y).float().unsqueeze(1))

def load_ns_times(field="t"):
    """
    Physical times recorded in the training file, indexed by time step.

    Returns
    -------
    ndarray of shape (n_times,)
    """
    path = dataset_path("ns_V1e-4_N10000_T30.mat")
    with h5py.File(path, "r") as f:
        return np.asarray(f[field]).squeeze()