RESOLUTIONS = [16, 32, 64]

COMMON = dict(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_layers=4,
    lifting_channels=64,
    projection_channels=64,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=75,
    alpha=0.1,
    gamma=0.1,
)

CONFIGS = {
    16: dict(n_modes=(16, 16), **COMMON),   # full spectrum, nothing truncated
    32: dict(n_modes=(24, 24), **COMMON),
    64: dict(n_modes=(32, 32), **COMMON),
}

SPLIT_SIZES = [5000, 3000, 1000]  # predictor, error, calibration

N_RESAMPLES = 3000
N_CAL_RESAMPLE = 500

INPUT_TIME = 10
OUTPUT_TIME = 20

SEED = 42

MODEL_VERSION = 1