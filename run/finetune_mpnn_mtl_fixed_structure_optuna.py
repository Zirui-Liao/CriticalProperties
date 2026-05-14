"""
Fixed-structure MTL fine-tuning for Chemprop MPNN from a pretrained checkpoint.

For each MTL split:
    - load pretrained MPNN checkpoint
    - load fixed structural hyperparameters
    - optimize training hyperparameters only
    - final train+val fine-tune and test evaluation for each fold
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import numpy as np
import optuna
from chemprop import data as cp_data
from chemprop import featurizers

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import mpnn_config as config
from models.mpnn_model import build_mpnn_from_params
from utils.mpnn_utils import (
    create_trainer,
    load_structure_params,
    predict_array,
    save_last_and_best_model_aliases,
)
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    compute_multitask_array_metrics,
    load_pretrained_weights_into_model,
    load_split_data,
    save_cv_metrics,
    save_fold_outputs,
    set_seed,
)


def compute_fold_val_rmse_finetune(
    params: Dict[str, Any],
    split_file: Path,
    pretrain_ckpt_path: Path,
    max_epochs: int,
    patience: int,
    target_cols: Sequence[str],
    batch_size: int,
    num_workers: int,
    seed: int | None = None,
) -> float:
    """Fine-tune on one fold and return validation mean RMSE."""
    set_seed(config.random_seed if seed is None else seed)

    train_smi, train_y, val_smi, val_y, test_smi, test_y = load_split_data(split_file, target_cols=target_cols)
    if len(train_smi) == 0:
        raise ValueError(f"[{split_file.stem}] Train set is empty.")
    if len(val_smi) == 0:
        val_smi, val_y = test_smi, test_y
    if len(val_smi) == 0:
        raise ValueError(f"[{split_file.stem}] Val set is empty.")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_ds = cp_data.MoleculeDataset(
        [cp_data.MoleculeDatapoint.from_smi(smi, yy) for smi, yy in zip(train_smi, train_y)],
        featurizer,
    )
    scaler = train_ds.normalize_targets()
    val_ds = cp_data.MoleculeDataset(
        [cp_data.MoleculeDatapoint.from_smi(smi, yy) for smi, yy in zip(val_smi, val_y)],
        featurizer,
    )
    val_ds.normalize_targets(scaler)
    train_loader = cp_data.build_dataloader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = cp_data.build_dataloader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model = build_mpnn_from_params(n_tasks=len(target_cols), scaler=scaler, hparams=params)
    load_pretrained_weights_into_model(model, pretrain_ckpt_path)

    trainer = create_trainer(
        max_epochs=max_epochs,
        patience=patience,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    preds = predict_array(trainer, model, val_loader)
    metrics = compute_multitask_array_metrics(val_y, preds, target_cols)
    mean_rmse = metrics["mean_rmse"]

    print(
        f"[{split_file.stem}] train_params="
        f"{ {key: params[key] for key in config.training_keys} }"
    )
    print(f"[{split_file.stem}] val mean_rmse={mean_rmse:.4f}, mean_r2={metrics['mean_r2']:.4f}")
    return float(mean_rmse)


def final_train_val_test_finetune(
    split_file: Path,
    fold_dir: Path,
    pretrain_ckpt_path: Path,
    best_params: Dict[str, Any],
    max_epochs: int,
    patience: int,
    target_cols: Sequence[str],
    batch_size: int,
    num_workers: int,
    seed: int | None = None,
) -> Dict[str, float]:
    """Final train+val fine-tuning and test evaluation for one fold."""
    set_seed(config.random_seed if seed is None else seed)

    train_smi, train_y, val_smi, val_y, test_smi, test_y = load_split_data(split_file, target_cols=target_cols)
    if len(train_smi) == 0:
        raise ValueError(f"[Final {split_file.stem}] Train set is empty.")
    if len(val_smi) == 0:
        raise ValueError(f"[Final {split_file.stem}] Val set is empty.")
    if len(test_smi) == 0:
        raise ValueError(f"[Final {split_file.stem}] Test set is empty.")

    train_val_smi = train_smi + val_smi
    train_val_y = np.concatenate([train_y, val_y], axis=0)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_ds = cp_data.MoleculeDataset(
        [cp_data.MoleculeDatapoint.from_smi(smi, yy) for smi, yy in zip(train_val_smi, train_val_y)],
        featurizer,
    )
    scaler = train_ds.normalize_targets()
    val_ds = cp_data.MoleculeDataset(
        [cp_data.MoleculeDatapoint.from_smi(smi, yy) for smi, yy in zip(val_smi, val_y)],
        featurizer,
    )
    val_ds.normalize_targets(scaler)
    test_ds = cp_data.MoleculeDataset(
        [cp_data.MoleculeDatapoint.from_smi(smi, yy) for smi, yy in zip(test_smi, test_y)],
        featurizer,
    )
    train_loader = cp_data.build_dataloader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = cp_data.build_dataloader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = cp_data.build_dataloader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model = build_mpnn_from_params(n_tasks=len(target_cols), scaler=scaler, hparams=best_params)
    load_pretrained_weights_into_model(model, pretrain_ckpt_path)

    fold_dir.mkdir(parents=True, exist_ok=True)
    trainer = create_trainer(
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=fold_dir / "checkpoints",
        default_root_dir=fold_dir,
        enable_progress_bar=False,
        log_dir=fold_dir / "trainer_logs",
    )
    trainer.fit(model, train_loader, val_loader)
    last_model_path, best_model_alias_path = save_last_and_best_model_aliases(
        trainer,
        fold_dir / "checkpoints",
    )

    pred_ckpt_path = best_model_alias_path
    preds = predict_array(trainer, model, test_loader, ckpt_path=pred_ckpt_path)
    metrics = compute_multitask_array_metrics(test_y, preds, target_cols)
    preds_path = save_fold_outputs(
        fold_dir=fold_dir,
        smiles=test_smi,
        preds=preds,
        target_cols=target_cols,
    )

    print(f"[Final] {split_file.stem} test metrics: {metrics}")
    print(f"[Final] {split_file.stem} predictions saved to: {preds_path}")
    print(f"[Final] {split_file.stem} last checkpoint saved to: {last_model_path}")
    if best_model_alias_path is not None:
        print(f"[Final] {split_file.stem} best checkpoint alias saved to: {best_model_alias_path}")
        print(f"[Final] {split_file.stem} predictions used best checkpoint: {best_model_alias_path}")
    else:
        print(f"[Final] {split_file.stem} no best checkpoint found; predictions used the current model state.")
    print(f"[Final] {split_file.stem} checkpoints saved under: {fold_dir / 'checkpoints'}")
    return metrics


def make_finetune_objective(ctx: SimpleNamespace):
    """Create the Optuna objective for one MTL fine-tuning split."""

    def objective(trial: optuna.Trial) -> float:
        train_params = config.suggest_training_params(trial, ctx.max_epochs)
        params = {**ctx.base_struct_hparams, **train_params}

        fold_rmse_list = []
        for fold_idx in range(ctx.fold_num):
            fold_rmse = compute_fold_val_rmse_finetune(
                params=params,
                split_file=ctx.data_dir / f"fold_{fold_idx}.csv",
                pretrain_ckpt_path=ctx.pretrain_ckpt_path,
                max_epochs=ctx.max_epochs,
                patience=ctx.patience,
                target_cols=ctx.target_cols,
                batch_size=ctx.batch_size,
                num_workers=ctx.num_workers,
                seed=ctx.random_seed + fold_idx,
            )
            fold_rmse_list.append(fold_rmse)

        avg_rmse = float(np.mean(fold_rmse_list))
        trial.set_user_attr("fold_rmse_list", fold_rmse_list)
        print(f"[HPO][Trial {trial.number}] avg_val_mean_rmse={avg_rmse:.4f}")
        return avg_rmse

    return objective


def run_final_finetune_folds(best_params: Dict[str, Any], ctx: SimpleNamespace) -> None:
    fold_metrics = []
    for fold_idx in range(ctx.fold_num):
        print(f"\n========== {ctx.split_name} | Fold {fold_idx} ==========")
        metrics = final_train_val_test_finetune(
            split_file=ctx.data_dir / f"fold_{fold_idx}.csv",
            fold_dir=ctx.final_root / f"fold_{fold_idx}",
            pretrain_ckpt_path=ctx.pretrain_ckpt_path,
            best_params=best_params,
            max_epochs=ctx.max_epochs,
            patience=ctx.patience,
            target_cols=ctx.target_cols,
            batch_size=ctx.batch_size,
            num_workers=ctx.num_workers,
            seed=getattr(ctx, "random_seed", config.random_seed) + fold_idx,
        )
        fold_metrics.append(metrics)

    cv_metrics_path = save_cv_metrics(
        ctx.final_root,
        fold_metrics,
        ("mean_r2", "mean_mae", "mean_rmse"),
    )
    print(f"[Final] Cross-validation metrics saved to: {cv_metrics_path}")




if __name__ == "__main__":
    set_seed(config.random_seed)

    pretrain_ckpt_path = config.get_pretrained_checkpoint_path()
    if not pretrain_ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrain_ckpt_path}. "
            "Run pretrain_mpnn.py first."
        )

    base_struct_hparams = load_structure_params(config.get_reference_best_params_path(), config.structure_keys)
    print("[INFO] Loaded structural params from multi-task mpnn_optuna:", base_struct_hparams)

    split_settings = [
        {
            "split_name": "kmeans_no_outliers_035",
            "data_dir": config.project_root / "data_folds" / "kmeans_no_outliers_035",
            "save_root": config.project_root / "mpnn_models" / "kmeans_no_outliers_035_optuna_ft_from_pretrain",
        },
        {
            "split_name": "random_no_outliers_035",
            "data_dir": config.project_root / "data_folds" / "random_no_outliers_035",
            "save_root": config.project_root / "mpnn_models" / "random_no_outliers_035_optuna_ft_from_pretrain",
        },
    ]

    for split_setting in split_settings:
        split_name = split_setting["split_name"]
        data_dir = split_setting["data_dir"]
        save_root = split_setting["save_root"]
        print(f"\n==================== Fine-tune on split: {split_name} ====================")
        hpo_root = save_root
        final_root = save_root
        hpo_root.mkdir(parents=True, exist_ok=True)
        final_root.mkdir(parents=True, exist_ok=True)

        ctx = SimpleNamespace(
            split_name=split_name,
            data_dir=data_dir,
            batch_size=config.pretrain_batch_size,
            num_workers=config.num_workers,
            max_epochs=config.finetune_hpo_max_epochs,
            patience=config.finetune_hpo_patience,
            fold_num=config.fold_num,
            target_cols=config.mtl_target_cols,
            hpo_root=hpo_root,
            final_root=final_root,
            pretrain_ckpt_path=pretrain_ckpt_path,
            base_struct_hparams=base_struct_hparams,
            random_seed=config.random_seed,
        )

        sampler = optuna.samplers.TPESampler(seed=config.random_seed)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=f"mpnn_ft_hpo_from_pretrain_{split_name}",
        )
        study.optimize(make_finetune_objective(ctx), n_trials=config.n_trials)

        print(f"[{split_name}] Best value (avg_val_mean_rmse):", study.best_value)
        print(f"[{split_name}] Best training params:", study.best_params)
        best_params = {**base_struct_hparams, **study.best_params}
        save_study_results(
            study,
            hpo_root,
            best_params,
            params_label="Best (struct+train) params",
        )

        print("=" * 80)
        print(f"[{split_name}] Final fine-tuning (train+val) and testing")
        print("=" * 80)
        run_final_finetune_folds(best_params, ctx)
