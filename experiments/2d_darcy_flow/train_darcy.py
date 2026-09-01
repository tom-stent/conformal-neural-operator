"""Train UQ-FNOs for the 2D Darcy flow experiment at each resolution.

Run from the repository root:

    python experiments/2d_darcy_flow/train_darcy.py
"""

import torch
from torch.utils.data import DataLoader

from conformal_no.data.datasets import SubsampleDataset, make_splits
from conformal_no.training.conformal_fno import train_uqfno
from experiments.expkit.datasets import load_darcy
from experiments.expkit.paths import CHECKPOINT_DIR

from configs import RESOLUTIONS, CONFIGS, SPLIT_SIZES, SEED

MODEL_VERSION = 2


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_darcy()
    predictor_dataset, error_dataset, _ = make_splits(raw["train"], SPLIT_SIZES,
                                                      seed=SEED)

    for res in RESOLUTIONS:
        print(f"\n--- Training resolution {res} ---")

        predictor_loader = DataLoader(
            SubsampleDataset(predictor_dataset, res),
            batch_size=CONFIGS[res]["batch_size"],
            shuffle=True,
        )
        error_loader = DataLoader(
            SubsampleDataset(error_dataset, res),
            batch_size=CONFIGS[res]["batch_size"],
            shuffle=True,
        )

        save_path = CHECKPOINT_DIR / f"darcy_uqfno_{res}_v{MODEL_VERSION}.pt"
        train_uqfno(
            predictor_loader,
            error_loader,
            CONFIGS[res],
            device,
            save_path=save_path,
        )
        print(f"Saved {save_path.name}")


if __name__ == "__main__":
    main()