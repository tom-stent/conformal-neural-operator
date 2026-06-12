import torch
import torch.nn as nn
from neuralop.data.datasets.tensor_dataset import TensorDataset
from torch.utils.data import DataLoader


def pointwise_error(u: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
    """
    Pointwise error magnitude with shape (B, 1, nx, ny).

    Absolute error for scalar-valued outputs, 2-norm over the channel
    dimension otherwise.
    """
    if u.shape[1] == 1:
        return (u - u_pred).abs()
    return torch.linalg.vector_norm(u - u_pred, ord=2, dim=1, keepdim=True)


@torch.no_grad()
def compute_residual_dataset(
        model: nn.Module,
        loader: torch.utils.data.dataloader.DataLoader,
        device: torch.device,
        batch_size: int | None = None,
        shuffle: bool = True,
    ):
    """
    Convert a dataset of solution data to residual data for uncertainty training.

    Parameters
    ----------
    model : nn.Module
        Trained base predictor.
    loader : torch.utils.data.DataLoader
        Batches of dicts with keys "x" and "y".
    device : torch.device
    batch_size : int, optional
        Batch size of the returned loader (defaults to loader.batch_size).
    shuffle : bool, optional
        Whether the returned loader shuffles (default True).
    """

    model = model.to(device)
    model.eval()

    a_list, errors = [], []

    for item in loader:
        a, u = item["x"].to(device), item["y"].to(device)
        u_pred = model(a)
        error = pointwise_error(u, u_pred)

        a_list.append(a.cpu())
        errors.append(error.cpu())

    a_list = torch.cat(a_list, dim=0)
    errors = torch.cat(errors, dim=0)

    return DataLoader(
        TensorDataset(x=a_list, y=errors),
        batch_size=batch_size if batch_size is not None else loader.batch_size,
        shuffle=shuffle,
    )
