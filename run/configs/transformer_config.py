import json
from pathlib import Path
from typing import Optional

import torch

project_root = Path(__file__).resolve().parents[2]
exp_file = project_root / "training_data" / "exp_data_no_outliers_035.csv"
pretrain_file = project_root / "training_data" / "pretraining_data_035.csv"

fold_num = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtl_target_cols = ("Tc", "Pc", "Vc", "omega")
stl_properties = ["Tc", "Pc", "Vc", "omega"]

default_max_epochs = 600
default_patience = 60
default_warmup_epochs = 30
pretrain_epochs = 300
pretrain_patience = 30
pretrain_lr = 1e-3
pretrain_weight_decay = 1e-5
pretrain_batch_size = 64
pretrain_warmup_epochs = 30

random_seed = 42
n_trials = 30
reference_split_name = "kmeans_no_outliers_035"
fixed_structure_split_name = "random_no_outliers_035"
reference_save_tag = f"{reference_split_name}_optuna"
fixed_structure_save_tag = f"{fixed_structure_split_name}_optuna_trainopt"
pretrain_tag = "pretrain_mtl_035"

structure_keys = [
    "d_model",
    "n_head",
    "hidden_factor",
    "dropout_model",
    "dropout_pos",
    "num_layers",
    "head_hidden_factor",
    "dropout_head",
]

training_keys = [
    "lr",
    "weight_decay",
    "batch_size",
]


def select_params(params: dict, keys: list[str]) -> dict:
    return {key: params[key] for key in keys}


def load_params(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_structure_params(path: Optional[Path] = None) -> dict:
    return select_params(load_params(path or get_reference_best_params_path()), structure_keys)


def suggest_structure_params(trial) -> dict:
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 96, 128]),
        "n_head": trial.suggest_categorical("n_head", [1, 2, 4, 8]),
        "hidden_factor": trial.suggest_categorical("hidden_factor", [2, 4, 6, 8]),
        "dropout_model": trial.suggest_float("dropout_model", 0.0, 0.3),
        "dropout_pos": trial.suggest_float("dropout_pos", 0.0, 0.3),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "head_hidden_factor": trial.suggest_categorical("head_hidden_factor", [1, 2, 4]),
        "dropout_head": trial.suggest_float("dropout_head", 0.0, 0.5),
    }


def format_params(params: dict, keys: list[str]) -> str:
    formatted = []
    for key in keys:
        value = params[key]
        if isinstance(value, float):
            spec = ".1e" if key in {"lr", "weight_decay"} else ".3f"
            value = format(value, spec)
        formatted.append(f"{key}={value}")
    return ", ".join(formatted)


def get_vocab_path() -> Path:
    return project_root / "transformer_models" / "vocab.json"


def get_max_len_path() -> Path:
    return project_root / "transformer_models" / "max_len.txt"


def get_reference_save_root() -> Path:
    return project_root / "transformer_models" / reference_save_tag


def get_reference_best_params_path() -> Path:
    return get_reference_save_root() / "best_hyperparams.json"


def get_split_dir(split_name: str) -> Path:
    return project_root / "data_folds" / split_name


def suggest_training_params(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


def suggest_hparams(trial) -> dict:
    return {**suggest_structure_params(trial), **suggest_training_params(trial)}
