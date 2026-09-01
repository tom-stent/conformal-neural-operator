RESOLUTIONS = [12, 20, 30, 42, 60, 84]

COMMON = dict(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_layers=4,
    lifting_channels=64,
    projection_channels=64,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=75,
    alpha=0.1,
    gamma=0.1,
)

CONFIGS = {
    12: dict(n_modes=(10, 10), **COMMON),
    20: dict(n_modes=(16, 16), **COMMON),
    30: dict(n_modes=(16, 16), **COMMON),
    42: dict(n_modes=(16, 16), **COMMON),
    60: dict(n_modes=(16, 16), **COMMON),
    84: dict(n_modes=(16, 16), **COMMON),
}

SPLIT_SIZES = [2500, 1000, 500]  # predictor, error, calibration

N_RESAMPLES = 3000
N_CAL_RESAMPLE = 500

SEED = 42

MODEL_VERSION = 1