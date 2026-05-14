"""
Fixed-structure MTL training hyperparameter optimization for Chemprop MPNN.

Stage:
    - Load structural hyperparameters from the reference kmeans MTL search.
    - Optimize training hyperparameters on random_no_outliers_035.
    - Final train+val retrain and test evaluation for each fold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import mpnn_config as config
from search_mpnn_reference_optuna import (
    compute_fold_val_rmse,
    run_final_folds,
)
from utils.mpnn_utils import load_structure_params
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    compute_multitask_array_metrics,
    set_seed,
)


def objective(trial: optuna.Trial, ctx: SimpleNamespace) -> float:
    """MTL objective with fixed structure and training hyperparameters only."""
    train_hparams = config.suggest_training_params(trial, ctx.max_epochs)
    params = {**ctx.base_struct_hparams, **train_hparams}

    fold_rmse_list = []
    for fold_idx in range(ctx.fold_num):
        fold_rmse = compute_fold_val_rmse(hparams=params, split_file=ctx.data_dir / f"fold_{fold_idx}.csv",
                                          max_epochs=ctx.max_epochs, patience=ctx.patience, target_cols=ctx.target_cols,
                                          batch_size=ctx.batch_size, num_workers=ctx.num_workers,
                                          metric_fn=ctx.metric_fn, enable_progress_bar=ctx.enable_progress_bar,
                                          precision=ctx.precision, seed=ctx.random_seed + fold_idx)
        fold_rmse_list.append(fold_rmse)

    avg_rmse = float(np.mean(fold_rmse_list))
    trial.set_user_attr("fold_rmse_list", fold_rmse_list)
    print(f"[Trial {trial.number}] avg_val_mean_rmse={avg_rmse:.4f}")
    return avg_rmse




if __name__ == "__main__":
    set_seed(config.random_seed)

    split_name = config.fixed_structure_split_name
    data_dir = config.get_split_dir(split_name)
    hpo_root = config.get_mtl_save_root(config.fixed_structure_hpo_tag)
    final_root = config.get_mtl_save_root(config.fixed_structure_final_tag)
    hpo_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    base_struct_hparams = load_structure_params(config.get_reference_best_params_path(), config.structure_keys)
    ctx = SimpleNamespace(
        data_dir=data_dir,
        batch_size=config.fixed_structure_batch_size,
        num_workers=config.num_workers,
        max_epochs=config.default_max_epochs,
        patience=config.default_patience,
        fold_num=config.fold_num,
        target_cols=config.mtl_target_cols,
        hpo_root=hpo_root,
        final_root=final_root,
        base_struct_hparams=base_struct_hparams,
        metric_fn=compute_multitask_array_metrics,
        enable_progress_bar=True,
        precision="16-mixed",
        random_seed=config.random_seed,
    )

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"mpnn_train_hpo_{split_name}",
    )
    study.optimize(lambda trial: objective(trial, ctx), n_trials=config.n_trials)

    print("[HPO] Best value (avg_val_mean_rmse):", study.best_value)
    best_params = {**base_struct_hparams, **study.best_params}
    print("[HPO] Best training params:", study.best_params)
    save_study_results(
        study,
        hpo_root,
        best_params,
        params_label="Best (struct+train) params",
    )

    print("[Final] Start training and testing with best hyperparameters...")
    run_final_folds(
        best_params=best_params,
        ctx=ctx,
        metric_keys=("mean_r2", "mean_mae", "mean_rmse"),
    )
