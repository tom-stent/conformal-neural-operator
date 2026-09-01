# Conformal Uncertainty Quantification Guarantees for Neural Operators

Code accompanying:

> Tom Stent and Nicolas Boullé. *Conformal Uncertainty Quantification
> Guarantees for Neural Operators*. arXiv:2608.28515, 2026.
> [https://arxiv.org/abs/2608.28515](https://arxiv.org/abs/2608.28515)

The method calibrates a neural operator's pointwise error estimate by a single conformal scale factor, giving a prediction band with a finite-sample  (gamma, alpha) coverage guarantee. Results compared against procedure of Ma et al. (2024).

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/tom-stent/conformal-neural-operator.git
cd conformal-neural-operator
pip install -e .
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used to produce the results in
the thesis.

## Data

The datasets are not tracked in git. Download them and place them in
`~/Datasets/conformal_no/`:

| File | Source |
|------|--------|
| `darcy_421/darcy_train_421.pt` | `DarcyDataset` from the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library |
| `darcy_421/darcy_test_421.pt` | as above |
| `ns_V1e-4_N10000_T30.mat` | [FNO data release](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) (accessed August 2026) |
| `ns_data_V1e-4_N20_T50_R256test.mat` | as above |

To keep the data elsewhere, set the data directory instead:

```bash
export CONFORMAL_NO_DATA=/your/path
```

## Reproducing the results

Pretrained checkpoints and precomputed results are included, so every
figure reproduces without re-running the experiments.

```bash
jupyter notebook experiments/2d_darcy_flow/darcy_experiments.ipynb
jupyter notebook experiments/2d_navier_stokes/ns_experiments.ipynb
```

Run each top to bottom. Figures are written to `report/figures/`, which is
created on first run.

To recompute the results from the checkpoints rather than loading them,
set `RECOMPUTE = True` at the top of the notebook.

To retrain from scratch:

```bash
python experiments/2d_darcy_flow/train_darcy.py
python experiments/2d_navier_stokes/train_ns.py
```

### Compute

All training and evaluation was run on CPU on a single 2024 MacBook Pro
(Apple M4, 24 GB memory). No GPU or cluster resources were used.

| Stage | Time |
|-------|------|
| Training one prediction/error pair | 5 minutes at the coarsest resolution, 90 minutes at the finest |
| Calibration and the T = 3000 resampling experiment | Under 3 minutes in total |
| Fine-resolution transfer test | Approximately 90 minutes |

## Repository layout

```
src/conformal_no/     Reusable library: FNO, conformal calibration, losses
├── data/             Dataset wrappers and splitting
├── models/           FNO and operator wrappers
├── training/         Training loops and model construction
├── uq/               Conformal quantile regressor calibration methods
└── utils/            Pointwise error, field utilities, losses

experiments/          Everything specific to reproducing the results
├── expkit/           Dataset loaders, plotting, results assembly, paths
├── 2d_darcy_flow/    Config, training script, experiment notebook, results
├── 2d_navier_stokes/ Config, training script, experiment notebook, results
└── pretrained/       Model checkpoints
```

## Citation

```bibtex
@misc{stent2026conformalno,
      title={Conformal Uncertainty Quantification Guarantees for Neural Operators}, 
      author={Tom Stent and Nicolas Boullé},
      year={2026},
      eprint={2608.28515},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2608.28515}, 
}
```

## Reference

Ziqi Ma, David Pitt, Kamyar Azizzadenesheli, and Anima Anandkumar. *Calibrated uncertainty quantification for operator learning via conformal prediction*. Transactions on Machine Learning Research, August 2024.