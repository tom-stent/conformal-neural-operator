import torch
import torch.nn as nn
import math

@torch.no_grad()
def compute_sol_reg(loader, device, domain_len=1.0, safe_scaling=1.0):
    """
    Estimate the modulus of continuity of the solution at the fill radius.

    Bounds the solution increment over the cell fill radius 
    h = sqrt(d) * domain_len / (2N) by the maximum resolved gradient norm,
    a beta = 1 (Lipschitz) estimate. Userd for omega_h in 
    continuum_conformal_calibrate.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Batches of dicts with key "y". Must be a split disjoint from the
        calibration set, since reusing it would break exchangeability.
    device : torch.device
    domain_len : float, default 1.0
        Physical length of each spatial axis.
    safe_scaling : float, default 1.0
        Multiplies the estimate. Values above 1 give a conservative bound
        to offset the gradient being resolved only on the grid.

    Returns
    -------
    float
        Estimated bound on the solution increment over one fill radius.
    """
    N = d = spacing = None
    L = 0.0
    for item in loader:
        u = item["y"].to(device) # (B, C, *spatial)
        if N is None:
            N, d = u.shape[-1], u.ndim - 2
            spacing = domain_len / N
        gsq = None
        for dim in range(d):
            d_i = (torch.roll(u, -1, dims=dim + 2) - u) / spacing # partial_i g (periodic)
            s  = (d_i ** 2).sum(dim=1, keepdim=True) # sum over channels
            gsq = s if gsq is None else gsq + s
        L = max(L, gsq.sqrt().amax().item()) # max grad over nodes
    h = math.sqrt(d) * domain_len / (2 * N)
    return L * h * safe_scaling


def fourier_interpolate(x, target_size):
    """
    Spectrally interpolate a real field onto a finer grid.

    Zero-pads the Fourier coefficients, which is the trigonometric interpolant 
    of the coarse field and so the FNO's own continuum extension of its grid 
    output.

    Parameters
    ----------
    x : Tensor
        Real, of shape (..., *spatial). Leading batch and channel axes are
        untouched.
    target_size : tuple of int
        Target spatial shape, e.g. (H2,), (H2, W2) or (D2, H2, W2). Must
        be at least the current size on every axis, upsampling only.

    Returns
    -------
    Tensor
        Real, of shape (..., *target_size).
    """
    d    = len(target_size)
    dims = tuple(range(-d, 0))
    cur  = x.shape[-d:]

    X = torch.fft.fftn(x, dim=dims, norm="forward") # norm="forward" preserves amplitude
    X = torch.fft.fftshift(X, dim=dims) # low freqs to centre

    pad = [] # F.pad wants last dim first
    for n_old, n_new in zip(reversed(cur), reversed(target_size)):
        p = n_new - n_old
        pad.extend((p // 2, p - p // 2)) # symmetric split
    X = nn.functional.pad(X, pad)

    X = torch.fft.ifftshift(X, dim=dims)
    return torch.fft.ifftn(X, dim=dims, norm="forward").real