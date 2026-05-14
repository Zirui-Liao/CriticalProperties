import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import transformer_config as config
from models.transformer_model import (
    SmilesDataset,
    TransformerEncoderModel,
)
from utils.transformer_utils import (
    load_vocab_and_max_len,
    train_model,
    evaluate,
)
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    load_pretrained_weights_into_model,
    load_split_data,
    masked_mse_loss,
    save_cv_metrics,
    seed_worker,
    set_seed,
)


def compute_fold_val_mse_finetune(
        hparams: dict,
        split_file: Path,
        pretrain_ckpt_path: Path,
        char_to_idx: dict,
        max_len: int,
        device: torch.device,
        max_epochs: int,
        patience: int,
        finetune_warmup_epochs: int,
        target_cols=None,
        seed: int | None = None,
) -> float:
    """
    Fine-tune on one fold starting from a pretrained checkpoint,
    and return validation MSE in scaled space.
    """
    if not split_file.exists():
        raise optuna.exceptions.TrialPruned()

    train_smi, train_y, val_smi, val_y, _, _ = load_split_data(
        split_file, target_cols=target_cols
    )
    if len(train_smi) == 0 or len(val_smi) == 0:
        raise optuna.exceptions.TrialPruned()

    generator = set_seed(config.random_seed if seed is None else seed)

    train_ds = SmilesDataset(train_smi, train_y, char_to_idx, max_len=max_len)
    val_ds = SmilesDataset(val_smi, val_y, char_to_idx, max_len=max_len)

    train_params = config.select_params(hparams, config.training_keys)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_params["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: tuple(torch.stack(items) for items in zip(*batch)),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_params["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: tuple(torch.stack(items) for items in zip(*batch)),
        worker_init_fn=seed_worker,
    )

    # Fit scaler on train labels only
    y_train_list = [y for _, y, _ in train_loader]
    if not y_train_list:
        raise optuna.exceptions.TrialPruned()
    y_train = torch.cat(y_train_list, dim=0).numpy()
    scaler = StandardScaler().fit(y_train)

    output_dim = train_y.shape[1]
    model = TransformerEncoderModel(
        vocab_size=len(char_to_idx),
        max_len=max_len,
        output_dim=output_dim,
        **config.select_params(hparams, config.structure_keys),
    ).to(device)

    # Load pretrained weights
    load_pretrained_weights_into_model(model, pretrain_ckpt_path)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    # Fine-tune
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        save_logs=False,
        log_path=None,
        save_model=False,
        model_path=None,
        epochs=max_epochs,
        patience=patience,
        base_lr=train_params["lr"],
        warmup_epochs=finetune_warmup_epochs,
    )

    # Compute validation loss (scaled space)
    model.eval()
    val_loss_total, num_val_batches = 0.0, 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, mask = x.to(device), mask.to(device)
            y_scaled = scaler.transform(y.numpy())
            y_scaled = torch.tensor(
                y_scaled, dtype=torch.float32, device=device
            )
            pred = model(x, mask=mask)
            val_loss_total += masked_mse_loss(pred, y_scaled).item()
            num_val_batches += 1

    if num_val_batches == 0:
        raise optuna.exceptions.TrialPruned()

    return val_loss_total / num_val_batches


def final_train_val_test_finetune(
        split_file: Path,
        fold_dir: Path,
        pretrain_ckpt_path: Path,
        char_to_idx: dict,
        max_len: int,
        best_params: dict,
        device: torch.device,
        max_epochs: int,
        patience: int,
        finetune_warmup_epochs: int,
        target_cols=None,
        eval_fn=None,
        eval_extra_kwargs=None,
        return_metrics=False,
        seed: int | None = None,
) -> float:
    """
    Final fine-tuning on train+val starting from pretrained weights,
    and evaluation on test for one fold.
    """
    if not split_file.exists():
        print(f"[WARN] Split file not found: {split_file}")
        if return_metrics:
            return float("nan"), {}
        return float("nan")

    fold_dir.mkdir(parents=True, exist_ok=True)
    log_dir = fold_dir / "trainer_logs"
    ckpt_dir = fold_dir / "checkpoints"
    preds_path = fold_dir / "test_predictions.csv"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # load per-split data
    train_smi, train_y, val_smi, val_y, test_smi, test_y = load_split_data(
        split_file, target_cols=target_cols
    )
    if (len(train_smi) + len(val_smi)) == 0 or len(test_smi) == 0:
        print(f"[WARN] Empty train+val or test for split: {split_file}")
        if return_metrics:
            return float("nan"), {}
        return float("nan")

    # build train+val set for final fine-tuning
    train_val_smi = train_smi + val_smi
    train_val_y = np.concatenate([train_y, val_y], axis=0)

    # datasets
    train_val_ds = SmilesDataset(train_val_smi, train_val_y, char_to_idx, max_len=max_len)
    val_ds = SmilesDataset(val_smi, val_y, char_to_idx, max_len=max_len)
    test_ds = SmilesDataset(test_smi, test_y, char_to_idx, max_len=max_len)

    # loaders
    train_params = config.select_params(best_params, config.training_keys)
    batch_size = train_params["batch_size"]
    generator = set_seed(config.random_seed if seed is None else seed)
    train_val_loader = DataLoader(
        train_val_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: tuple(torch.stack(items) for items in zip(*batch)),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: tuple(torch.stack(items) for items in zip(*batch)),
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: tuple(torch.stack(items) for items in zip(*batch)),
        worker_init_fn=seed_worker,
    )

    # fit label scaler on train+val
    y_all_list = [y for _, y, _ in train_val_loader]
    if not y_all_list:
        print(f"[WARN] No train+val data for split: {split_file}")
        if return_metrics:
            return float("nan"), {}
        return float("nan")
    y_all = torch.cat(y_all_list, dim=0).numpy()
    scaler = StandardScaler().fit(y_all)

    model = TransformerEncoderModel(
        vocab_size=len(char_to_idx),
        max_len=max_len,
        output_dim=train_val_y.shape[1],
        **config.select_params(best_params, config.structure_keys),
    ).to(device)

    # Load pretrained weights
    load_pretrained_weights_into_model(model, pretrain_ckpt_path)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    # fine-tune on train+val, monitor val
    model = train_model(
        model=model,
        train_loader=train_val_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        save_logs=True,
        log_path=log_dir,
        save_model=True,
        model_path=ckpt_dir,
        epochs=max_epochs,
        patience=patience,
        base_lr=train_params["lr"],
        warmup_epochs=finetune_warmup_epochs,
    )

    # evaluate on test using test_smi from the split
    if eval_fn is None:
        eval_fn = evaluate
        eval_kwargs = {
            "save_test_preds": True,
            "return_metrics": return_metrics,
        }
        if target_cols is not None:
            eval_kwargs["target_cols"] = target_cols
    else:
        eval_kwargs = {"save_test_preds": True, "return_metrics": return_metrics}
        if eval_extra_kwargs:
            eval_kwargs.update(eval_extra_kwargs)

    test_mse = eval_fn(
        model=model,
        loader=test_loader,
        test_smi=test_smi,
        preds_path=preds_path,
        scaler=scaler,
        device=device,
        **eval_kwargs,
    )
    if return_metrics:
        test_mse, metrics = test_mse
        return float(test_mse), metrics
    return float(test_mse)


# ===============================
# 5. Build Optuna objective for fine-tuning
# ===============================
def make_finetune_objective(
        data_dir: Path,
        pretrain_ckpt_path: Path,
        char_to_idx: dict,
        max_len: int,
        device: torch.device,
        max_epochs: int,
        patience: int,
        base_struct_hparams: dict,
        finetune_warmup_epochs: int,
        fold_num: int,
        target_cols=None,
        random_seed: int = config.random_seed,
):
    """
    Create an Optuna objective function for MTL fine-tuning on a given split directory.
    """

    def objective(trial: optuna.Trial) -> float:
        hparams = {**base_struct_hparams, **config.suggest_training_params(trial)}

        val_mse_list = []
        for k in range(fold_num):
            split_file = data_dir / f"fold_{k}.csv"
            fold_mse = compute_fold_val_mse_finetune(
                hparams=hparams,
                split_file=split_file,
                pretrain_ckpt_path=pretrain_ckpt_path,
                char_to_idx=char_to_idx,
                max_len=max_len,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                finetune_warmup_epochs=finetune_warmup_epochs,
                target_cols=target_cols,
                seed=random_seed + k,
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

    return objective


def run_final_finetune_folds(best_params: dict, ctx: SimpleNamespace) -> None:
    fold_metrics = []
    for fold_idx in range(ctx.fold_num):
        print(f"\n========== {ctx.split_name} | Fold {fold_idx} ==========")
        test_mse, metrics = final_train_val_test_finetune(
            split_file=ctx.data_dir / f"fold_{fold_idx}.csv",
            fold_dir=ctx.final_root / f"fold_{fold_idx}",
            pretrain_ckpt_path=ctx.pretrain_ckpt_path,
            char_to_idx=ctx.char_to_idx,
            max_len=ctx.max_len,
            best_params=best_params,
            device=ctx.device,
            max_epochs=ctx.max_epochs,
            patience=ctx.patience,
            finetune_warmup_epochs=ctx.finetune_warmup_epochs,
            target_cols=ctx.target_cols,
            eval_fn=None,
            eval_extra_kwargs=None,
            return_metrics=True,
            seed=getattr(ctx, "random_seed", config.random_seed) + fold_idx,
        )

        if not np.isnan(test_mse):
            fold_metrics.append(metrics)
            print(
                f"[{ctx.split_name}][Fold {fold_idx}] "
                f"Test MSE (original scale, MTL finetune): {test_mse:.6f}"
            )

    cv_metrics_path = save_cv_metrics(
        ctx.final_root,
        fold_metrics,
        ("mean_r2", "mean_mae", "mean_rmse"),
    )
    print(f"[{ctx.split_name}][Final] Cross-validation metrics saved to: {cv_metrics_path}")


# ===============================
# 6. Main: pretrain + finetune on random & kmeans splits
# ===============================


if __name__ == "__main__":
    set_seed(config.random_seed)

    fold_num = config.fold_num
    device = config.device
    finetune_max_epochs = config.default_max_epochs
    finetune_patience = config.default_patience
    finetune_warmup_epochs = config.default_warmup_epochs
    target_cols = config.mtl_target_cols
    char_to_idx, max_len = load_vocab_and_max_len(
        vocab_path=config.get_vocab_path(),
        max_len_path=config.get_max_len_path(),
    )
    best_path = config.get_reference_save_root() / "best_hyperparams.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Cannot find best_hyperparams.json at {best_path}. "
            f"Please run transformer_optuna.py (kmeans MTL) first."
        )
    base_struct_hparams = config.load_structure_params(best_path)

    pretrain_ckpt_path = config.project_root / "transformer_models" / config.pretrain_tag / "checkpoints" / "best_model.pt"
    if not pretrain_ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrain_ckpt_path}. "
            "Run pretrain_transformer.py first."
        )

    split_settings = [
        {
            "split_name": "kmeans_no_outliers_035",
            "data_dir": config.project_root / "data_folds" / "kmeans_no_outliers_035",
            "save_root": config.project_root / "transformer_models" / "kmeans_no_outliers_035_mtl_ft_from_pretrain",
        },
        {
            "split_name": "random_no_outliers_035",
            "data_dir": config.project_root / "data_folds" / "random_no_outliers_035",
            "save_root": config.project_root / "transformer_models" / "random_no_outliers_035_mtl_ft_from_pretrain",
        },
    ]

    for split_setting in split_settings:
        split_name = split_setting["split_name"]
        data_dir = split_setting["data_dir"]
        save_root = split_setting["save_root"]
        print(f"\n==================== Fine-tune on split: {split_name} ====================")
        os.makedirs(save_root, exist_ok=True)
        final_ctx = SimpleNamespace(
            split_name=split_name,
            data_dir=data_dir,
            final_root=save_root,
            fold_num=fold_num,
            max_epochs=finetune_max_epochs,
            patience=finetune_patience,
            device=device,
            target_cols=target_cols,
            char_to_idx=char_to_idx,
            max_len=max_len,
            pretrain_ckpt_path=pretrain_ckpt_path,
            finetune_warmup_epochs=finetune_warmup_epochs,
            random_seed=config.random_seed,
        )

        sampler = optuna.samplers.TPESampler(seed=config.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        objective = make_finetune_objective(
            data_dir=data_dir,
            pretrain_ckpt_path=pretrain_ckpt_path,
            char_to_idx=char_to_idx,
            max_len=max_len,
            device=device,
            max_epochs=finetune_max_epochs,
            patience=finetune_patience,
            base_struct_hparams=base_struct_hparams,
            finetune_warmup_epochs=finetune_warmup_epochs,
            fold_num=fold_num,
            target_cols=target_cols,
            random_seed=config.random_seed,
        )

        study.optimize(objective, n_trials=config.n_trials)

        best = study.best_trial
        print(f"[{split_name}] Best trial value (mean val MSE):", best.value)
        print(f"[{split_name}] Best trial params (training only):", best.params)

        best_train_params = best.params
        best_params = {**base_struct_hparams, **best_train_params}

        print(
            f"[{split_name}] Using best hyperparameters (struct + training): "
            f"{config.format_params(best_params, config.structure_keys + config.training_keys)}"
        )

        save_study_results(
            study,
            save_root,
            best_params,
            params_label="Best (struct+train) params",
        )

        run_final_finetune_folds(best_params, final_ctx)
