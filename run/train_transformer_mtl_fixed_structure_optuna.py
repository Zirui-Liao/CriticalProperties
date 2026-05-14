import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import transformer_config as config
from search_transformer_reference_optuna import (
    compute_fold_val_mse,
    run_final_folds,
)
from utils.optuna_utils import save_study_results
from utils.trainer_utils import set_seed
from utils.transformer_utils import load_vocab_and_max_len

# ===============================
# 2. Optuna objective for MTL on random split
# ===============================
def objective(trial: optuna.Trial, ctx: SimpleNamespace) -> float:
    """
    Multitask objective on RANDOM split (no pretraining).
    Uses fixed structural hyperparameters and optimizes only training hyperparameters.
    """
    train_hparams = config.suggest_training_params(trial)
    hparams = {**ctx.base_struct_hparams, **train_hparams}

    val_mse_list = []
    for k in range(ctx.fold_num):
        split_file = ctx.data_dir / f"fold_{k}.csv"
        fold_mse = compute_fold_val_mse(
            hparams=hparams,
            split_file=split_file,
            char_to_idx=ctx.char_to_idx,
            max_len=ctx.max_len,
            device=ctx.device,
            max_epochs=ctx.max_epochs,
            patience=ctx.patience,
            target_cols=ctx.target_cols,  # multi-task: all properties
            seed=ctx.random_seed + k,
        )
        val_mse_list.append(fold_mse)

    mean_val_mse = float(np.mean(val_mse_list))
    trial.set_user_attr("fold_rmse_list", val_mse_list)

    print(
        f"[Trial {trial.number}] "
        f"{config.format_params(hparams, config.structure_keys + config.training_keys)} | "
        f"val_mse={mean_val_mse:.6f}"
    )

    return mean_val_mse


# ===============================
# 3. Run Optuna + final per-fold training/eval on RANDOM split (MTL)
# ===============================


if __name__ == "__main__":
    set_seed(config.random_seed)

    split_name = config.fixed_structure_split_name
    data_dir = config.get_split_dir(split_name)
    save_root = config.project_root / "transformer_models" / config.fixed_structure_save_tag
    fold_num = config.fold_num
    max_epochs = config.default_max_epochs
    patience = config.default_patience
    device = config.device
    target_cols = config.mtl_target_cols
    char_to_idx, max_len = load_vocab_and_max_len(
        vocab_path=config.get_vocab_path(),
        max_len_path=config.get_max_len_path(),
    )
    base_struct_hparams = config.load_structure_params(config.get_reference_best_params_path())

    ctx = SimpleNamespace(
        data_dir=data_dir,
        fold_num=fold_num,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        target_cols=target_cols,
        char_to_idx=char_to_idx,
        max_len=max_len,
        base_struct_hparams=base_struct_hparams,
        split_name=split_name,
        final_root=save_root,
        random_seed=config.random_seed,
    )

    os.makedirs(save_root, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda trial: objective(trial, ctx), n_trials=config.n_trials)

    best = study.best_trial
    print("Best trial value (mean val MSE):", best.value)
    print("Best trial params (training only):", best.params)

    best_params = {**base_struct_hparams, **best.params}

    print(
        "Using best hyperparameters on RANDOM split (MTL, no pretraining): "
        f"{config.format_params(best_params, config.structure_keys + config.training_keys)}"
    )

    save_study_results(
        study,
        save_root,
        best_params,
        params_label="Best (struct+train) params",
    )

    run_final_folds(
        best_params=best_params,
        ctx=ctx,
        metric_keys=("mean_r2", "mean_mae", "mean_rmse"),
    )
