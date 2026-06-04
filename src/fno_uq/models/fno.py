import torch
import torch.nn as nn
import torch.nn.functional as F


# Complex multiplication helper in Fourier space
def compl_mul2d(
    v_ft: torch.Tensor,   # shape: (B, in_channels, n_modes_x, n_modes_y), complex
    R: torch.Tensor  # shape: (in_channels, out_channels, n_modes_x, n_modes_y), complex
) -> torch.Tensor:
    """
    Multiply Fourier coefficients by learned complex weights.
 
    For each retained Fourier mode (kx, ky), matrix multiply summing over in channels.
 
    Output shape:
        (B, out_channels, n_modes_x, n_modes_y)
    """
    return torch.einsum("bixy,ioxy->boxy", v_ft, R)

 
class SpectralConv2d(nn.Module):
    """
    Simple dense 2D spectral convolution.
 
    Input:
        v of shape (B, in_channels, nx, ny), real-valued
 
    Output:
        v_spec of shape (B, out_channels, nx, ny), real-valued
 
    Performs FFT over spatial axes
    Retains only low modes
    Multiply retained modes by learned complex weights
    Inverse FFT back to physical space
    """
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_x: int,
        n_modes_y: int,
    ):
        super().__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of modes kept in first spatial frequency direction
        self.n_modes_x = n_modes_x 

        # Number of modes kept in second spatial frequency direction
        # (for real FFT, the last Fourier axis stores only nonredundant half)
        self.n_modes_y_rfft = n_modes_y // 2 + 1 

        # Glorot scaling
        scale = (2.0 / (in_channels + out_channels))**0.5
        
        self.weights = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, self.n_modes_x, self.n_modes_y_rfft, 
                dtype=torch.cfloat
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
        v: shape (B, in_channels, nx, ny), real
        returns: shape (B, out_channels, nx, ny), real
        """
        batch_size = v.shape[0]
        nx = v.shape[-2]
        ny = v.shape[-1]

        assert self.n_modes_x <= nx, "n_modes_x must be <= nx"
        assert self.n_modes_y_rfft <= ny // 2 + 1, (
            "n_modes_y_rfft must be <= ny//2 + 1 for rfft2"
        )
 
        # Real FFT over the two spatial axes
        # output shape: (B, in_channels, nx, ny//2 + 1), complex
        v_ft = torch.fft.rfftn(v, dim=(-2, -1), norm="forward")

        # FFT shift on non-last Fourier axis (so zero frequencies in centre block)
        v_ft = torch.fft.fftshift(v_ft, dim=(-2,))

        # Create zero Fourier tensor with same shape but out_channels
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            nx,
            ny // 2 + 1,
            dtype=v_ft.dtype,
            device=v.device,
        )

        slice_x = self._centred_slice(nx, self.n_modes_x)
        slice_y = slice(0, self.n_modes_y_rfft)

        # Truncate high frequency modes
        v_ft_trunc = v_ft[:, :, slice_x, slice_y]

        out_ft[:, :, slice_x, slice_y] = compl_mul2d(v_ft_trunc, self.weights)
        # all other Fourier modes remain zero

        # Undo shift in non-last Fourier axis before inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=(-2,))
 
        # Inverse FFT back to real space
        v_spec = torch.fft.irfftn(out_ft, s=(nx, ny), dim=(-2, -1), norm="forward")
 
        return v_spec
 
 
# Simple FNO block
class FNOBlock2d(nn.Module):
    """
    Activation(spectral convolution + pointwise linear map)

    Input and output shape:
        (B, hidden_channels, nx, ny)
    """
 
    def __init__(
        self,
        hidden_channels: int,
        n_modes_x: int,
        n_modes_y: int,
        activation = F.gelu,
    ):
        super().__init__()
 
        self.hidden_channels = hidden_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        self.activation = activation
 
        # Fourier branch
        self.spectral_conv = SpectralConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes_x=n_modes_x,
            n_modes_y=n_modes_y,
        )

        # Local pointwise branch
        # kernel_size=1 i.e. mix channels at each grid point
        self.pointwise_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=True
        )
 
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: shape (B, hidden_channels, nx, ny)
        """

        v_spec = self.spectral_conv(v)
        v_pw = self.pointwise_conv(v)
 
        v = v_spec + v_pw
        v = self.activation(v)
 
        return v
 
 
class FNO2d(nn.Module):
    """ 
    Expected input shape: (B, in_channels, nx, ny)
 
    Expected output shape: (B, out_channels, nx, ny)
 
    Architecture:
        Optional grid append
        Lifting (pointwise channel mixing)
        Repeated FNO blocks
        Projection (pointwise channel mixing back to output)
    """
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes_x: int,
        n_modes_y: int,
        n_layers: int = 4,
        lifting_channels: int | None = None,
        projection_channels: int | None = None,
        activation = F.gelu,
        append_grid: bool = True
    ):
        super().__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        self.n_layers = n_layers
        self.append_grid = append_grid
 
        # If appending a 2D coordinate grid add 2 extra input channels
        lifting_in_channels = in_channels + 2 if append_grid else in_channels
 
        # Optional hidden widths for lift/projection MLPs
        if lifting_channels is None:
            lifting_channels = 2 * hidden_channels
 
        if projection_channels is None:
            projection_channels = 2 * hidden_channels
 
        # Lifting layer (small pointwise MLP using 1x1 convolutions)
        # biases here
        self.lifting = nn.Sequential(
            nn.Conv2d(lifting_in_channels, lifting_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(lifting_channels, hidden_channels, kernel_size=1, bias=True),
        )
 
        # Repeated Fourier blocks
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    hidden_channels=hidden_channels,
                    n_modes_x=n_modes_x,
                    n_modes_y=n_modes_y,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )
 
        # Projection layer
        # Again, small pointwise MLP via 1x1 convolutions.
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, projection_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(projection_channels, out_channels, kernel_size=1, bias=True),
        )
 
    def get_grid(self, a: torch.Tensor) -> torch.Tensor:
        """
        Build a [0,1] x [0,1] coordinate grid.
 
        Input:
            a of shape (B, in_channels, nx, ny)
 
        Output:
            grid of shape (B, 2, nx, ny)
            2 channels store x and y coordinates
        """
        batch_size = a.shape[0]
        nx = a.shape[-2]
        ny = a.shape[-1]
        device = a.device
        dtype = a.dtype
 
        grid_x = torch.linspace(0, 1, nx, device=device, dtype=dtype)
        grid_y = torch.linspace(0, 1, ny, device=device, dtype=dtype)

        grid_x = grid_x[None, None, :, None].repeat(batch_size, 1, 1, ny)
        grid_y = grid_y[None, None, None, :].repeat(batch_size, 1, nx, 1)

        grid = torch.cat([grid_x, grid_y], dim=1)
        return grid # shape (B, 2, nx, ny)
 

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: shape (B, in_channels, nx, ny)
 
        returns:
            u: shape (B, out_channels, nx, ny)
        """
 
        # Optionally append coordinate channels
        if self.append_grid:
            grid = self.get_grid(a)
            a = torch.cat([a, grid], dim=1)  # concat along channel dimension
 
        # Lift to hidden width d_v
        v = self.lifting(a)  # shape (B, hidden_channels, nx, ny)
 
        # Repeated FNO blocks
        for block in self.blocks:
            v = block(v)
 
        # Project back down to output channels
        u = self.projection(v)
 
        return u