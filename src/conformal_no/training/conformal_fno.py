import copy
import torch
from torch.optim import AdamW
from conformal_no.models.fno import FNO
from conformal_no.models.wrappers import PositiveOperator
from conformal_no.uq.conformal_quantile import ConformalQuantileRegressor
from neuralop import LpLoss
from conformal_no.training.loops import train


def train_uqfno(train_loader, error_train_loader, hyperparam_config,
                        device, save_path=None):
    """
    Create, train and optionally save a UQ-FNO, consisting of a predictor FNO
    and an error FNO.

    First the predictor FNO is fitted on train_loader under a relative L2 loss. 
    The error FNO is then created as a copy of the trained predictor, wrapped in 
    PositiveOperator so its output is non-negative, and fitted on 
    error_train_loader against the residuals the predictor makes there.

    Parameters
    ----------
    train_loader : DataLoader
        Predictor training data.
    error_train_loader : DataLoader
        Error model training data. Must come from a split disjoint from
        train_loader.
    hyperparam_config : dict
        Reads "hidden_channels", "n_modes", "n_layers", "lifting_channels", 
        "projection_channels", "lr", "weight_decay", "epochs" and "gamma". 
        The same lr and epoch count are used for both stages.
    device : torch.device
    save_path : str or Path, optional
        If given, the trained state_dict is written here. The file does not 
        record which config produced it, so keep that identifiable from the 
        filename and pass the same config to load_uqfno.

    Returns
    -------
    ConformalQuantileRegressor
        Trained but not yet calibrated. Run the conformal calibration step on
        the held-out calibration split before using any interval widths.

    Notes
    -----
    The train and error loaders must be disjoint.
    """

    # Create base FNO
    fno = FNO(
        in_channels=hyperparam_config["in_channels"],
        out_channels=hyperparam_config["out_channels"],
        hidden_channels=hyperparam_config["hidden_channels"],
        n_modes=hyperparam_config["n_modes"],
        n_layers=hyperparam_config["n_layers"],
        lifting_channels=hyperparam_config["lifting_channels"],
        projection_channels=hyperparam_config["projection_channels"]
    )

    l2loss = LpLoss(d=2, p=2)
    optimiser = AdamW(
        fno.parameters(),
        lr=hyperparam_config["lr"],
        weight_decay=hyperparam_config["weight_decay"]
    )

    # Train base FNO
    train(
        fno,
        train_loader,
        test_loaders=None,
        loss_fn=l2loss,
        optimiser=optimiser,
        device=device,
        epochs=hyperparam_config["epochs"]
    )

    # Create UQ-FNO
    error_fno = PositiveOperator(copy.deepcopy(fno))
    uqfno = ConformalQuantileRegressor(
        fno,
        error_fno,
        gamma=hyperparam_config["gamma"],
        device=device
    )

    # Train error FNO
    optimiser = AdamW(
        uqfno.quantile_model.parameters(),
        lr=hyperparam_config["lr"],
        weight_decay=hyperparam_config["weight_decay"]
    )

    uqfno.fit_quantile(error_train_loader, optimiser=optimiser,
                        epochs=hyperparam_config["epochs"])

    if save_path is not None:
        torch.save(uqfno.state_dict(), save_path)
        print("UQFNO state saved")

    return uqfno


def load_uqfno(hyperparam_config, path, device):
    """
    Rebuild a UQ-FNO from a config and load saved weights into it.

    The architecture is reconstructed exactly as train_uqfno builds it. The 
    initial weights are irrelevant since load_state_dict overwrites them, only
    the structure has to match.

    Parameters
    ----------
    hyperparam_config : dict
        Must be the config the checkpoint was trained with. A mismatch in any
        architectural key raises an error.
    path : str or Path
        Checkpoint written by train_uqfno.
    device : torch.device
        Weights are mapped here on load.

    Returns
    -------
    ConformalQuantileRegressor
        On device and in eval mode, ready for inference.
    """

    # Create FNO
    fno = FNO(
        in_channels=hyperparam_config["in_channels"],
        out_channels=hyperparam_config["out_channels"],
        hidden_channels=hyperparam_config["hidden_channels"],
        n_modes=hyperparam_config["n_modes"],
        n_layers=hyperparam_config["n_layers"],
        lifting_channels=hyperparam_config["lifting_channels"],
        projection_channels=hyperparam_config["projection_channels"]
    )

    error_fno = PositiveOperator(copy.deepcopy(fno))
    uqfno = ConformalQuantileRegressor(
        fno,
        error_fno,
        hyperparam_config["gamma"],
        device
    )

    # Load state from file path
    state_dict = torch.load(path, map_location=device)
    missing, unexpected = uqfno.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={missing}, unexpected={unexpected}")

    # Move to device, and set to evaluation mode
    uqfno.to(device)
    uqfno.eval()

    return uqfno