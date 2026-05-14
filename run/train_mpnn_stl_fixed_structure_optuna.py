"""
Fixed-structure STL training hyperparameter optimization for Chemprop MPNN.

For each property and each random/kmeans single-property split:
    - load structural hyperparameters from reference MTL search
    - optimize training hyperparameters only
    - final train+val retrain and test evaluation for each fold
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import numpy as np
import optuna

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import mpnn_config as config
from search_mpnn_reference_optuna import (
    compute_fold_val_rmse,
    final_train_val_test,
)
from utils.mpnn_utils import load_structure_params
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    compute_single_target_array_metrics,
    save_cv_metrics,
    set_seed,
)


def compute_stl_metrics(y_true: np.ndarray, preds: np.ndarray, target_cols: Sequence[str]) -> Dict[str, float]:
    return compute_single_target_array_metrics(y_true, preds)


def objective(
    trial: optuna.Trial,
    prop: str,
    data_dir: Path,
    struct_params: Dict[str, Any],
    ctx: SimpleNamespace,
) -> float:
    """STL objective with fixed structure and training hyperparameters only."""
    train_params = config.suggest_training_params(trial, ctx.max_epochs)
    params = {**struct_params, **train_params}

    fold_rmse_list = []
    for fold_idx in range(ctx.fold_num):
        fold_rmse = compute_fold_val_rmse(hparams=params, split_file=data_dir / f"fold_{fold_idx}.csv",
                                          max_epochs=ctx.max_epochs, patience=ctx.patience, target_cols=(prop,),
                                          batch_size=ctx.batch_size, num_workers=ctx.num_workers,
                                          metric_fn=compute_stl_metrics, enable_progress_bar=ctx.enable_progress_bar,
                                          precision=ctx.precision, seed=ctx.random_seed + fold_idx)
        fold_rmse_list.append(fold_rmse)

    avg_rmse = float(np.mean(fold_rmse_list))
    trial.set_user_attr("fold_rmse_list", fold_rmse_list)
    print(f"[{prop}][Trial {trial.number}] avg_val_rmse={avg_rmse:.4f}")
    return avg_rmse


def run_final_folds_for_property(
    prop: str,
    data_dir: Path,
    final_root: Path,
    best_params: Dict[str, Any],
    ctx: SimpleNamespace,
) -> None:
    fold_metrics = []
    for fold_idx in range(ctx.fold_num):
        print(f"\n--- {prop} | Fold {fold_idx} ---")
        metrics = final_train_val_test(
            split_file=data_dir / f"fold_{fold_idx}.csv",
            fold_dir=final_root / f"fold_{fold_idx}",
            best_params=best_params,
            max_epochs=ctx.max_epochs,
            patience=ctx.patience,
            target_cols=(prop,),
            batch_size=ctx.batch_size,
            num_workers=ctx.num_workers,
            metric_fn=compute_stl_metrics,
            enable_progress_bar=ctx.enable_progress_bar,
            precision=ctx.precision,
            seed=getattr(ctx, "random_seed", config.random_seed) + fold_idx,
        )
        fold_metrics.append(metrics)

    cv_metrics_path = save_cv_metrics(final_root, fold_metrics, ("r2", "mae", "rmse"))
    print(f"[{prop}][Final] Cross-validation metrics saved to: {cv_metrics_path}")




if __name__ == "__main__":
    set_seed(config.random_seed)

    models_root = config.project_root / "mpnn_models"
    struct_params = load_structure_params(config.get_reference_best_params_path(), config.structure_keys)
    print("[INFO] Loaded structural params from multi-task mpnn_optuna:", struct_params)

    ctx = SimpleNamespace(
        batch_size=config.stl_batch_size,
        num_workers=config.num_workers,
        max_epochs=config.default_max_epochs,
        patience=config.default_patience,
        fold_num=config.fold_num,
        enable_progress_bar=False,
        precision=None,
        random_seed=config.random_seed,
    )

    for prop in config.stl_properties:
        split_settings = [
            {
                "split_name": f"kmeans_{prop}_no_outliers_035",
                "data_dir": config.project_root / "data_folds" / f"kmeans_{prop}_no_outliers_035",
                "save_root": models_root / f"kmeans_{prop}_no_outliers_035_optuna",
            },
            {
                "split_name": f"random_{prop}_no_outliers_035",
                "data_dir": config.project_root / "data_folds" / f"random_{prop}_no_outliers_035",
                "save_root": models_root / f"random_{prop}_no_outliers_035_optuna",
            },
        ]

        for split_setting in split_settings:
            split_name = split_setting["split_name"]
            data_dir = split_setting["data_dir"]
            save_root = split_setting["save_root"]
            split_prefix = split_name.split("_", 1)[0]
            hpo_root = save_root
            final_root = save_root
            hpo_root.mkdir(parents=True, exist_ok=True)
            final_root.mkdir(parents=True, exist_ok=True)

            print("=" * 80)
            print(f"[{prop}][{split_prefix}] Starting HPO on split: {split_name}")
            print("=" * 80)

            sampler = optuna.samplers.TPESampler(seed=config.random_seed)
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=f"mpnn_optuna_stl_{split_prefix}_{prop}",
            )
            study.optimize(
                lambda trial: objective(
                    trial=trial,
                    prop=prop,
                    data_dir=data_dir,
                    struct_params=struct_params,
                    ctx=ctx,
                ),
                n_trials=config.n_trials,
            )

            print(f"[{prop}][{split_prefix}][HPO] Best value (avg_val_rmse): {study.best_value}")
            print(f"[{prop}][{split_prefix}][HPO] Best training params: {study.best_params}")

            best_params = {**struct_params, **study.best_params}
            save_study_results(
                study,
                hpo_root,
                best_params,
                params_label="Best (struct+train) params",
            )

            print(f"[{prop}][{split_prefix}][Final] Start training and testing with best hyperparameters...")
            run_final_folds_for_property(
                prop=prop,
                data_dir=data_dir,
                final_root=final_root,
                best_params=best_params,
                ctx=ctx,
            )
