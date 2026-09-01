import torch

def train_one_epoch(model, train_loader, optimiser, loss_fn, device):
    """
    Run one full training epoch over the training dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Neural operator model to be trained.
    train_loader
        DataLoader providing training batches. Each batch must be a dict
        containing keys "x" and "y".
    optimiser : torch.optim.Optimizer
    loss_fn : callable
    device : torch.device

    Returns
    -------
    float
        Average training loss over the entire dataset.
    """
    model.train()

    total_loss = 0
    for item in train_loader:
        optimiser.zero_grad(set_to_none=True)
        a, u = item["x"].to(device), item["y"].to(device)
        u_pred = model(a)
        loss = loss_fn(u_pred, u)
        total_loss += loss.item() * a.size(0)
        loss.backward()
        optimiser.step()
    
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """
    Evaluate the model on a dataset without gradient computation.

    Parameters
    ----------
    model : torch.nn.Module
        Neural operator model to evaluate.
    loader : torch.utils.data.DataLoader
        DataLoader providing evaluation batches. Each batch must be a dict
        containing keys "x" and "y".
    loss_fn : callable
    device : torch.device

    Returns
    -------
    float
        Average loss over the dataset.
    """

    model.eval()

    total_loss = 0
    for item in loader:
        a, u = item["x"].to(device), item["y"].to(device)
        u_pred = model(a)

        loss = loss_fn(u_pred, u)
        total_loss += loss.item() * a.size(0)

    return total_loss / len(loader.dataset)


def train(model, train_loader, test_loaders, loss_fn, optimiser, device, 
          epochs=50):
    """
    Train a model and evaluate it periodically across multiple validation 
    datasets.

    Parameters
    ----------
    model : torch.nn.Module
        Neural operator model to train.
    train_loader : torch.utils.data.DataLoader
        Training DataLoader. Each batch must contain "x" and "y".
    test_loaders : dict[int, torch.utils.data.DataLoader]
        Dictionary mapping resolution to validation DataLoader.
        Each loader must contain "x" and "y".
    loss_fn : callable
    optimiser : torch.optim.Optimizer
    device : torch.device
    epochs : int, optional
        Number of training epochs (default is 50).

    Returns
    -------
    dict
        Training history containing:
        - "train_loss": list of training losses per epoch
        - "val_loss": dict mapping each test_loader resolution to a list of 
                      validation losses per epoch
    """

    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": {res: [] for res in test_loaders} if test_loaders else {}
    }

    for epoch in range(epochs):

        # Train
        train_loss = train_one_epoch(model, train_loader, optimiser, loss_fn, device)
        history["train_loss"].append(train_loss)
        
        # Evaluate at all resolutions
        if test_loaders:
            for res, loader in test_loaders.items():
                val_loss = evaluate(model, loader, loss_fn, device)
                history["val_loss"][res].append(val_loss)
        
        # Print updates
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, avg train loss: {train_loss}")

    return history