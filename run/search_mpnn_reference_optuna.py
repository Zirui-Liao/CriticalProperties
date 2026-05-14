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
    predict_array,
    save_last_and_best_model_aliases,
)
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    compute_multitask_array_metrics,
    load_split_data,
    save_cv_metrics,
    save_fold_outputs,
    set_seed,
)


def compute_fold_val_rmse(
        hparams: Dict[str, Any],
        split_file: Path,
        max_epochs: int,
        patience: int,
        target_cols: Sequence[str],
        batch_size: int,
        num_workers: int,
        metric_fn=compute_multitask_array_metrics,
        enable_progress_bar: bool = False,
        precision: str | None = None,
        seed: int | None = None,
) -> float:
    """Train on one fold and return validation RMSE in original target space."""
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

    model = build_mpnn_from_params(n_tasks=len(target_cols), scaler=scaler, hparams=hparams)
    trainer = create_trainer(
        max_epochs=max_epochs,
        patience=patience,
        enable_progress_bar=enable_progress_bar,
        precision=precision,
    )
    trainer.fit(model, train_loader, val_loader)

    preds = predict_array(trainer, model, val_loader)
    metrics = metric_fn(val_y, preds, target_cols)
    val_rmse = metrics["mean_rmse"] if "mean_rmse" in metrics else metrics["rmse"]

    print(
        f"[{split_file.stem}] params={hparams}\n"
        f"[{split_file.stem}] val_rmse={val_rmse:.4f}"
    )
    return float(val_rmse)


def final_train_val_test(
        split_file: Path,
        fold_dir: Path,
        best_params: Dict[str, Any],
        max_epochs: int,
        patience: int,
        target_cols: Sequence[str],
        batch_size: int,
        num_workers: int,
        metric_fn=compute_multitask_array_metrics,
        enable_progress_bar: bool = False,
        precision: str | None = None,
        seed: int | None = None,
) -> Dict[str, float]:
    """Final train+val training and test evaluation for one fold."""
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

    fold_dir.mkdir(parents=True, exist_ok=True)
    trainer = create_trainer(
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=fold_dir / "checkpoints",
        default_root_dir=fold_dir,
        enable_progress_bar=enable_progress_bar,
        precision=precision,
        log_dir=fold_dir / "trainer_logs",
    )
    trainer.fit(model, train_loader, val_loader)
    last_model_path, best_model_alias_path = save_last_and_best_model_aliases(
        trainer,
        fold_dir / "checkpoints",
    )

    pred_ckpt_path = best_model_alias_path
    preds = predict_array(trainer, model, test_loader, ckpt_path=pred_ckpt_path)
    metrics = metric_fn(test_y, preds, target_cols)
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


def objective(trial: optuna.Trial, ctx: SimpleNamespace) -> float:
    params = {
        "message_hidden_dim": trial.suggest_categorical("message_hidden_dim", [128, 256, 300, 384, 512]),
        "depth": trial.suggest_int("depth", 2, 6),
        "MPNN_DROPOUT": trial.suggest_float("MPNN_DROPOUT", 0.0, 0.4),
        "ffn_hidden_dim": trial.suggest_categorical("ffn_hidden_dim", [128, 256, 300, 384, 512]),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
        "FFN_DROPOUT": trial.suggest_float("FFN_DROPOUT", 0.0, 0.4),
        "init_lr": trial.suggest_float("init_lr", 1e-5, 5e-4, log=True),
        "max_lr": trial.suggest_float("max_lr", 5e-4, 5e-3, log=True),
        "final_lr": trial.suggest_float("final_lr", 1e-6, 1e-4, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, min(50, ctx.max_epochs // 2)),
    }

    fold_rmse_list = []
    for fold_idx in range(ctx.fold_num):
        split_file = ctx.data_dir / f"fold_{fold_idx}.csv"
        fold_rmse = compute_fold_val_rmse(hparams=params, split_file=split_file, max_epochs=ctx.max_epochs,
                                          patience=ctx.patience, target_cols=ctx.target_cols, batch_size=ctx.batch_size,
                                          num_workers=ctx.num_workers, metric_fn=ctx.metric_fn,
                                          enable_progress_bar=ctx.enable_progress_bar, precision=ctx.precision,
                                          seed=ctx.random_seed + fold_idx)
        fold_rmse_list.append(fold_rmse)

    avg_rmse = float(np.mean(fold_rmse_list))
    trial.set_user_attr("fold_rmse_list", fold_rmse_list)
    print(f"[Trial {trial.number}] avg_val_mean_rmse={avg_rmse:.4f}")
    return avg_rmse


def run_final_folds(
        best_params: Dict[str, Any],
        ctx: SimpleNamespace,
        metric_keys: Sequence[str],
) -> None:
    fold_metrics = []
    for fold_idx in range(ctx.fold_num):
        print(f"\n========== Fold {fold_idx} ==========")
        metrics = final_train_val_test(
            split_file=ctx.data_dir / f"fold_{fold_idx}.csv",
            fold_dir=ctx.final_root / f"fold_{fold_idx}",
            best_params=best_params,
            max_epochs=ctx.max_epochs,
            patience=ctx.patience,
            target_cols=ctx.target_cols,
            batch_size=ctx.batch_size,
            num_workers=ctx.num_workers,
            metric_fn=ctx.metric_fn,
            enable_progress_bar=ctx.enable_progress_bar,
            precision=ctx.precision,
            seed=getattr(ctx, "random_seed", config.random_seed) + fold_idx,
        )
        fold_metrics.append(metrics)

    cv_metrics_path = save_cv_metrics(ctx.final_root, fold_metrics, metric_keys)
    print(f"[Final] Cross-validation metrics saved to: {cv_metrics_path}")


if __name__ == "__main__":
    set_seed(config.random_seed)

    split_name = config.reference_split_name
    data_dir = config.get_split_dir(split_name)
    hpo_root = config.get_reference_save_root()
    final_root = config.project_root / "mpnn_models" / config.reference_final_tag
    hpo_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    ctx = SimpleNamespace(
        data_dir=data_dir,
        batch_size=config.reference_batch_size,
        num_workers=config.num_workers,
        max_epochs=config.default_max_epochs,
        patience=config.default_patience,
        fold_num=config.fold_num,
        target_cols=config.mtl_target_cols,
        hpo_root=hpo_root,
        final_root=final_root,
        metric_fn=compute_multitask_array_metrics,
        enable_progress_bar=False,
        precision=None,
        random_seed=config.random_seed,
    )

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"mpnn_hpo_{split_name}",
    )
    study.optimize(lambda trial: objective(trial, ctx), n_trials=config.n_trials)

    print("[HPO] Best value (avg_val_mean_rmse):", study.best_value)
    print("[HPO] Best params:", study.best_params)
    save_study_results(study, hpo_root, study.best_params)

    print("[Final] Start training and testing with best hyperparameters...")
    run_final_folds(
        best_params=study.best_params,
        ctx=ctx,
        metric_keys=("mean_r2", "mean_mae", "mean_rmse"),
    )
