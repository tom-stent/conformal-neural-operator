import torch
import torch.nn as nn
from neuralop.data.datasets.tensor_dataset import TensorDataset
from torch.utils.data import DataLoader


def pointwise_error(u: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
    """
    Pointwise error magnitude.

    Parameters
    ----------
    u, u_pred : Tensor
        Of shape (B, C, *spatial).

    Returns
    -------
    Tensor
        Of shape (B, 1, *spatial).
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
    Convert solution data into residual data for training the error model.

    Parameters
    ----------
    model : nn.Module
        Trained base predictor.
    loader : torch.utils.data.DataLoader
        Batches of dicts with keys "x" and "y".
    device : torch.device
    batch_size : int, optional
    shuffle : bool, default True

    Returns
    -------
    residual_loader : torch.utils.data.DataLoader
        Yields dicts with "x" the original input and "y" the normalised
        pointwise error.
    scale : Tensor
        Scalar mean error the targets were divided by.
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

    scale = errors.mean().clone()
    errors = errors / scale

    return DataLoader(
        TensorDataset(x=a_list, y=errors),
        batch_size=batch_size if batch_size is not None else loader.batch_size,
        shuffle=shuffle,
    ), scale