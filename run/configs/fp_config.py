from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
target_columns = ["Tc", "Pc", "Vc", "omega"]
property_columns = target_columns

random_seed = 42
fold_num = 5
fingerprint_radius = 2
fingerprint_n_bits = 1024

classical_model_names = ["mlp", "svm", "rf", "xgboost"]
classical_split_prefix = "random"
classical_split_suffix = "no_outliers_035"
classical_output_root = project_root / "clscl_models" / "outputs_clscl_ml_optuna_random"
classical_n_trials = 30

mlp_mtl_split_name = "random_no_outliers_035"
mlp_mtl_data_folds_dir = project_root / "data_folds" / mlp_mtl_split_name
mlp_mtl_output_root = project_root / "clscl_models" / "outputs_multitask_mlp_optuna_random"
mlp_mtl_max_trials = 30
mlp_mtl_epochs = 300
mlp_mtl_patience = 30
mlp_mtl_batch_size = 64
mlp_mtl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
