# Critical Properties

This repository contains data splitting, model training, fine-tuning, and plotting workflows for molecular critical property prediction.

The main model families are:

- Transformer models over SMILES strings
- Chemprop MPNN models
- Fingerprint-based classical ML (SVM, RF, XGBoost) and MLP baselines

## Important Data Requirement

SMILES validity must be checked before running any split, training, fine-tuning, or plotting workflow.

The project assumes all SMILES in the input CSV files are valid and parseable by RDKit/Chemprop. The code does not silently filter invalid SMILES during feature generation or training. If invalid SMILES are present, RDKit fingerprint generation or Chemprop datapoint creation may fail. Validate and clean the raw data first, then generate folds and train models.

Expected input columns for experimental data are generally:

- `SMILES`
- `Tc`
- `Pc`
- `Vc`
- `omega`

Fold CSV files also require a `split` column with `train`, `val`, and `test`.

## Transformer Vocabulary Requirement

Transformer workflows use a shared character vocabulary and maximum SMILES length saved under `transformer_models/`.
Build or confirm these files before running Transformer training, pretraining, or fine-tuning.

All SMILES used for Transformer modeling must be compatible with the prepared vocabulary. In particular, pretraining data should be filtered or regenerated so it does not introduce characters outside the vocabulary used for the experimental modeling runs. The current Transformer tokenizer does not define a separate unknown-token pathway, so vocabulary preparation is a required preprocessing step rather than something to leave to the training scripts.

## Repository Layout

- `split/`: random, K-means, Butina, and UMAP split generation utilities.
- `run/configs/`: configuration files for Transformer, MPNN, and fingerprint workflows.
- `run/models/`: model definitions.
- `run/utils/`: shared training, logging, metrics, data loading, and model utility functions.
- `run/`: training, pretraining, fine-tuning, and Optuna search entrypoints.
- `plots/`: result aggregation and plotting scripts.
- `training_data/`: cleaned experimental and pretraining CSV files.
- `data_folds/`: generated cross-validation folds.

## Typical Workflow

1. Validate and clean SMILES in the raw data.
2. Generate cleaned training data, for example with `split/cutting_exp_data.py`.
3. Generate fold files using one of the split scripts in `split/`.
4. Run reference Optuna searches and training scripts from `run/`.
5. Use scripts under `plots/` to aggregate metrics and create figures.

## Installation

Create an environment with Python 3.10+ or 3.11, then install dependencies:

```bash
pip install -r requirements.txt
```

Some packages, especially `torch`, `rdkit`, `chemprop`, and CUDA-enabled builds, may need environment-specific installation commands depending on your platform.
