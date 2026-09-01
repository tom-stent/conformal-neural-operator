import math
import torch
from torch.utils.data import Dataset
from neuralop.data.datasets.tensor_dataset import TensorDataset


class SubsampleDataset(Dataset):
    """
    Strided subsampling onto a coarser uniform grid.

    Works for any spatial rank, the stride is applied to every trailing axis 
    after the channel dimension.

    Parameters
    ----------
    dataset : Dataset
        Yields dicts with keys "x" and "y", each of shape (C, *spatial).
    resolution : int
        Target points per spatial axis for the input.
    output_resolution : int, optional
        Target for the output, if different from the input.
    """

    def __init__(self, dataset, resolution, output_resolution=None):
        self.dataset = dataset
        self.resolution = resolution
        self.output_resolution = output_resolution or resolution

        sample = dataset[0]
        self._slices_x = self._build_slices(sample["x"].shape[1:], self.resolution)
        self._slices_y = self._build_slices(sample["y"].shape[1:], self.output_resolution)
        self.dim = len(sample["x"].shape) - 1

    @staticmethod
    def _build_slices(spatial, resolution):
        for n in spatial:
            if n % resolution != 0:
                raise ValueError(
                    f"cannot uniformly subsample grid {tuple(spatial)} to "
                    f"resolution {resolution}: {n} is not divisible by it"
                )
        return (slice(None),) + tuple(
            slice(None, None, n // resolution) for n in spatial
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "x": sample["x"][self._slices_x],
            "y": sample["y"][self._slices_y],
        }


def add_channel(t, dim):
    """
    Insert a single channel axis, passing if already present.

    Parameters
    ----------
    t : Tensor
        Of shape (N, *spatial) or (N, C, *spatial).
    dim : int
        Spatial rank.

    Returns
    -------
    Tensor
        Of shape (N, 1, *spatial), or t unchanged.
    """
    return t.unsqueeze(1) if t.ndim == dim + 1 else t


def make_splits(xy, sizes, seed=42):
    """
    Split a pool of samples into disjoint subsets by random index.

    Parameters
    ----------
    xy : tuple of Tensor
        (x, y), each of shape (N, C, *spatial).
    sizes : sequence of int
        Subset sizes, taken in order from a single permutation.
    seed : int, default 42
        Seeds a local generator.

    Returns
    -------
    list of TensorDataset
        One per entry in sizes, in the same order.
    """
    x, y = xy
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))
    out, start = [], 0
    for n in sizes:
        idx = perm[start:start + n]
        start += n
        out.append(TensorDataset(x=x[idx], y=y[idx]))
    return out


def standardise(splits, ref_split="train", n_ref=None):
    """
    Standardise inputs using statistics from one split only.

    Fitting on the predictor-training data alone keeps the calibration and test 
    sets untouched, so exchangeability between them is preserved.
    """
    ref_x = splits[ref_split][0]
    if n_ref is not None:
        ref_x = ref_x[:n_ref]
    mu, sd = ref_x.mean(), ref_x.std()
    return {k: ((x - mu) / sd, y) for k, (x, y) in splits.items()}, (mu, sd)


def ma_valid_resolutions(H, dim=2, gamma=0.1, alpha=0.1):
    """
    Resolutions reachable by strided subsampling that also admit a valid Ma et 
    al. correction.

    Their t must satisfy sqrt(ln(1/alpha) / 2m) < t < gamma, so the procedure is 
    only defined when m = N**dim > ln(1/alpha) / (2 gamma**2).
    """
    m_min = math.log(1.0 / alpha) / (2.0 * gamma ** 2)
    return [r for r in range(2, H + 1) if H % r == 0 and r ** dim > m_min]