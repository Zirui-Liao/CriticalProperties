from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
exp_file = project_root / "training_data" / "exp_data_no_outliers_035.csv"
pretrain_file = project_root / "training_data" / "pretraining_data_035.csv"

fold_num = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtl_target_cols = ["Tc", "Pc", "Vc", "omega"]
stl_properties = ["Tc", "Pc", "Vc", "omega"]

default_max_epochs = 600
default_patience = 60
default_warmup_epochs = 30
random_seed = 42
n_trials = 30
num_workers = 0

reference_batch_size = 32
fixed_structure_batch_size = 128
stl_batch_size = 128
pretrain_batch_size = 32

pretrain_max_epochs = 300
pretrain_patience = 30
pretrain_init_lr = 1e-4
pretrain_max_lr = 1e-3
pretrain_final_lr = 1e-6
pretrain_warmup_epochs = 10
pretrain_val_fraction = 0.1
pretrain_random_seed = 42

finetune_hpo_max_epochs = default_max_epochs
finetune_hpo_patience = default_patience

reference_split_name = "kmeans_no_outliers_035"
fixed_structure_split_name = "random_no_outliers_035"
pretrain_tag = "pretrain_mtl_035"
reference_save_tag = f"{reference_split_name}_optuna"
reference_final_tag = reference_save_tag
fixed_structure_hpo_tag = f"{fixed_structure_split_name}_optuna_trainopt"
fixed_structure_final_tag = fixed_structure_hpo_tag
structure_keys = [
    "message_hidden_dim",
    "depth",
    "MPNN_DROPOUT",
    "ffn_hidden_dim",
    "ffn_num_layers",
    "FFN_DROPOUT",
]
training_keys = [
    "init_lr",
    "max_lr",
    "final_lr",
    "warmup_epochs",
]


def get_reference_save_root() -> Path:
    return project_root / "mpnn_models" / reference_save_tag

def get_reference_best_params_path() -> Path:
    return get_reference_save_root() / "best_hyperparams.json"


def get_pretrain_root() -> Path:
    return project_root / "mpnn_models" / pretrain_tag


def get_pretrained_checkpoint_path() -> Path:
    return get_pretrain_root() / "checkpoints" / "best_model.pt"


def get_split_dir(split_name: str) -> Path:
    return project_root / "data_folds" / split_name


def get_mtl_save_root(tag: str) -> Path:
    return project_root / "mpnn_models" / tag


def suggest_training_params(trial, max_epochs: int) -> dict:
    return {
        "init_lr": trial.suggest_float("init_lr", 1e-5, 5e-4, log=True),
        "max_lr": trial.suggest_float("max_lr", 5e-4, 5e-3, log=True),
        "final_lr": trial.suggest_float("final_lr", 1e-6, 1e-4, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, min(50, max_epochs // 2)),
    }
