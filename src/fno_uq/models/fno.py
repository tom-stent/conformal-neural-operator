import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


# Complex multiplication helper in Fourier space
def compl_mul(
    v_ft: torch.Tensor,   # shape: (B, in_channels, *n_modes), complex
    R: torch.Tensor,      # shape: (in_channels, out_channels, *n_modes), complex
) -> torch.Tensor:
    """
    Multiply Fourier coefficients by learned complex weights.

    For each retained Fourier mode, matrix multiply summing over in channels.
    Works for any spatial rank (1D, 2D, ...): the spatial mode axes are shared
    between the two operands and carried through unchanged.

    Output shape:
        (B, out_channels, *n_modes)
    """
    # ndim spatial axes follow the (batch, channel) axes
    ndim = v_ft.dim() - 2
    spatial = "xyzwuv"[:ndim]
    equation = f"bi{spatial},io{spatial}->bo{spatial}"
    return torch.einsum(equation, v_ft, R)


class SpectralConv(nn.Module):
    """
    Simple dense spectral convolution for an arbitrary spatial rank.

    Input:
        v of shape (B, in_channels, *spatial), real-valued

    Output:
        v_spec of shape (B, out_channels, *spatial), real-valued

    Performs FFT over the spatial axes
    Retains only low modes
    Multiply retained modes by learned complex weights
    Inverse FFT back to physical space

    n_modes is a per-axis tuple, e.g. (n_modes_x,) for 1D or 
    (n_modes_x, n_modes_y) for 2D. The number of spatial dimensions is
    inferred as len(n_modes).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = tuple(n_modes)
        self.ndim = len(self.n_modes)

        # Modes kept per axis. For all but the last spatial axis we keep the
        # requested number of (fftshifted) modes centred on zero frequency.
        # The last axis uses a real FFT, so it stores only the nonredundant
        # half and we keep n_modes_last // 2 + 1 of those.
        self.kept_modes = self.n_modes[:-1] + (self.n_modes[-1] // 2 + 1,)

        # Glorot scaling
        scale = (2.0 / (in_channels + out_channels))**0.5

        self.weights = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, *self.kept_modes,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def _centred_slice(size: int, kept: int) -> slice:
        """
        Slice of length 'kept' centred around fftshifted zero frequency
        """
        centre = size // 2
        neg = kept // 2
        pos = kept - neg
        return slice(centre - neg, centre + pos)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: shape (B, in_channels, *spatial), real
        returns: shape (B, out_channels, *spatial), real
        """
        batch_size = v.shape[0]
        spatial_sizes = v.shape[-self.ndim:] 

        # FFT over the spatial axes; fftshift every axis except the last
        # (rfft) one so each axis' zero frequency sits in the centre block.
        fft_dims = tuple(range(-self.ndim, 0))
        shift_dims = tuple(range(-self.ndim, -1))

        for i in range(self.ndim - 1):
            assert self.n_modes[i] <= spatial_sizes[i], (
                f"n_modes[{i}]={self.n_modes[i]} must be <= "
                f"spatial size {spatial_sizes[i]}"
            )
        assert self.kept_modes[-1] <= spatial_sizes[-1] // 2 + 1, (
            "kept modes on the last axis must be <= size//2 + 1 for rfft"
        )

        # Real FFT over the spatial axes
        # last axis length becomes spatial_sizes[-1] // 2 + 1, complex
        v_ft = torch.fft.rfftn(v, dim=fft_dims, norm="forward")

        if shift_dims:
            v_ft = torch.fft.fftshift(v_ft, dim=shift_dims)

        # Create zero Fourier tensor with same shape but out_channels
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            *spatial_sizes[:-1],
            spatial_sizes[-1] // 2 + 1,
            dtype=v_ft.dtype,
            device=v.device,
        )

        # Centred slices on the shifted axes, low-pass slice on the rfft axis
        spatial_slices = tuple(
            self._centred_slice(spatial_sizes[i], self.n_modes[i])
            for i in range(self.ndim - 1)
        ) + (slice(0, self.kept_modes[-1]),)
        index = (slice(None), slice(None)) + spatial_slices

        # Truncate high frequency modes, then map kept modes
        v_ft_trunc = v_ft[index]
        out_ft[index] = compl_mul(v_ft_trunc, self.weights)
        # all other Fourier modes remain zero

        # Undo shift before inverse FFT
        if shift_dims:
            out_ft = torch.fft.ifftshift(out_ft, dim=shift_dims)

        # Inverse FFT back to real space
        v_spec = torch.fft.irfftn(
            out_ft, s=tuple(spatial_sizes), dim=fft_dims, norm="forward"
        )

        return v_spec


# Simple FNO block
class FNOBlock(nn.Module):
    """
    Activation(spectral convolution + pointwise linear map)

    Input and output shape:
        (B, hidden_channels, *spatial)
    """

    def __init__(
        self,
        hidden_channels: int,
        n_modes: tuple[int, ...],
        activation=F.gelu,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.n_modes = tuple(n_modes)
        self.ndim = len(self.n_modes)
        self.activation = activation

        # Fourier branch
        self.spectral_conv = SpectralConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
        )

        # Local pointwise branch (kernel_size=1 i.e. mix channels at each grid
        # point), using the convolution matching the spatial rank.
        Conv = getattr(nn, f"Conv{self.ndim}d")
        self.pointwise_conv = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: shape (B, hidden_channels, *spatial)
        """
        v_spec = self.spectral_conv(v)
        v_pw = self.pointwise_conv(v)

        v = v_spec + v_pw
        v = self.activation(v)

        return v


class FNO(nn.Module):
    """
    Fourier Neural Operator for an arbitrary spatial rank.

    Expected input shape:  (B, in_channels, *spatial)
    Expected output shape: (B, out_channels, *spatial)

    Architecture:
        Optional grid append
        Lifting (pointwise channel mixing)
        Repeated FNO blocks
        Projection (pointwise channel mixing back to output)

    The spatial rank is set by len(n_modes); prefer the FNO1D / FNO2D
    convenience subclasses for the common cases.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: tuple[int, ...],
        n_layers: int = 4,
        domain_padding: float = 0.0,
        lifting_channels: int | None = None,
        projection_channels: int | None = None,
        activation=F.gelu,
        append_grid: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes = tuple(n_modes)
        self.ndim = len(self.n_modes)
        self.n_layers = n_layers
        self.domain_padding = domain_padding
        self.append_grid = append_grid

        # Appending a coordinate grid adds one input channel per spatial axis
        lifting_in_channels = (
            in_channels + self.ndim if append_grid else in_channels
        )

        # Optional hidden widths for lift/projection MLPs
        if lifting_channels is None:
            lifting_channels = 2 * hidden_channels

        if projection_channels is None:
            projection_channels = 2 * hidden_channels

        # Pointwise (1x1) convolution matching the spatial rank
        Conv = getattr(nn, f"Conv{self.ndim}d")

        # Lifting layer (small pointwise MLP using 1x1 convolutions)
        # biases here
        self.lifting = nn.Sequential(
            Conv(lifting_in_channels, lifting_channels, kernel_size=1, bias=True),
            nn.GELU(),
            Conv(lifting_channels, hidden_channels, kernel_size=1, bias=True),
        )

        # Repeated Fourier blocks
        self.blocks = nn.ModuleList(
            [
                FNOBlock(
                    hidden_channels=hidden_channels,
                    n_modes=self.n_modes,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

        # Projection layer
        # Again, small pointwise MLP via 1x1 convolutions.
        self.projection = nn.Sequential(
            Conv(hidden_channels, projection_channels, kernel_size=1, bias=True),
            nn.GELU(),
            Conv(projection_channels, out_channels, kernel_size=1, bias=True),
        )

    def get_grid(self, a: torch.Tensor) -> torch.Tensor:
        """
        Build a coordinate grid spanning [0, 1] along each spatial axis.

        Input:
            a of shape (B, in_channels, *spatial)

        Output:
            grid of shape (B, ndim, *spatial)
            one channel per spatial axis storing that axis' coordinate
        """
        batch_size = a.shape[0]
        spatial_sizes = a.shape[-self.ndim:]
        device = a.device
        dtype = a.dtype

        coords = []
        for axis, n in enumerate(spatial_sizes):
            c = torch.linspace(0, 1, n, device=device, dtype=dtype)
            # reshape so this axis' coordinate broadcasts over the others
            shape = [1, 1] + [1] * self.ndim
            shape[2 + axis] = n
            c = c.reshape(shape).expand(batch_size, 1, *spatial_sizes)
            coords.append(c)

        return torch.cat(coords, dim=1)  # shape (B, ndim, *spatial)

    def pad(self, v):
        """
        Pad each spatial axis symmetrically by a fixed fraction of its length.

        Returns
        -------
        v_padded   : padded tensor
        crop_slices: slices to remove the padding later

        Discretisation-invariance note
        ------------------------------
        Invariance requires the realised padding fraction p/N to be identical at
        every resolution. Because p is an integer number of cells, that holds 
        only when domain_padding * N is an integer.
        """
        if self.domain_padding == 0:
            return v, (...,)

        spatial = v.shape[-self.ndim:]

        pad_cells = []
        for n in spatial:
            exact = self.domain_padding * n
            p = round(exact)
            if abs(exact - p) > 1e-6:
                warnings.warn(
                    f"domain_padding={self.domain_padding} x grid size {n} = "
                    f"{exact:.4f} is not an integer; rounding to {p} cells gives an "
                    f"effective fraction {p / n:.4f}. The padded domain length then "
                    f"depends on resolution and discretisation invariance is lost. "
                    f"Pick domain_padding so domain_padding*N is integer at every "
                    f"resolution you use (e.g. 0.0625 / 0.125 / 0.25 on power-of-two "
                    f"grids).",
                    stacklevel=2,
                )
            pad_cells.append(p)

        # F.pad expects the last spatial axis first, as (left, right) pairs.
        pad_sizes = []
        for p in reversed(pad_cells):
            pad_sizes.extend([p, p])

        # Crop uses the original spatial order.
        crop_slices = tuple(slice(p, -p if p else None) for p in pad_cells)

        v = F.pad(v, pad_sizes, mode="constant")
        return v, (..., *crop_slices)


    def unpad(self, v, crop_slices):
        """Remove domain padding."""
        return v[crop_slices]


    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: shape (B, in_channels, *spatial)

        returns:
            u: shape (B, out_channels, *spatial)
        """
        # Optionally append coordinate channels
        if self.append_grid:
            grid = self.get_grid(a)
            a = torch.cat([a, grid], dim=1)  # concat along channel dimension

        # Lift to hidden width d_v
        v = self.lifting(a)  # shape (B, hidden_channels, *spatial)

        v, crop_slices = self.pad(v)

        # Repeated FNO blocks
        for block in self.blocks:
            v = block(v)

        v = self.unpad(v, crop_slices)

        # Project back down to output channels
        u = self.projection(v)

        return u
    

class FNO1D(FNO):
    """
    1D Fourier Neural Operator.

    Expected input shape:  (B, in_channels, nx)
    Expected output shape: (B, out_channels, nx)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes_x: int,
        n_layers: int = 4,
        domain_padding: float = 0.0,
        lifting_channels: int | None = None,
        projection_channels: int | None = None,
        activation=F.gelu,
        append_grid: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_modes=(n_modes_x,),
            n_layers=n_layers,
            domain_padding=domain_padding,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            activation=activation,
            append_grid=append_grid,
        )


class FNO2D(FNO):
    """
    2D Fourier Neural Operator.

    Expected input shape:  (B, in_channels, nx, ny)
    Expected output shape: (B, out_channels, nx, ny)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes_x: int,
        n_modes_y: int,
        n_layers: int = 4,
        domain_padding: float = 0.0,
        lifting_channels: int | None = None,
        projection_channels: int | None = None,
        activation=F.gelu,
        append_grid: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_modes=(n_modes_x, n_modes_y),
            n_layers=n_layers,
            domain_padding=domain_padding,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            activation=activation,
            append_grid=append_grid,
        )