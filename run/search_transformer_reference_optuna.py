import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
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
from utils.transformer_utils import evaluate, train_model
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    load_split_data,
    masked_mse_loss,
    save_cv_metrics,
    seed_worker,
    set_seed,
)


def compute_fold_val_mse(
        hparams: dict,
        split_file: Path,
        char_to_idx: dict,
        max_len: int,
        device: torch.device,
        max_epochs: int,
        patience: int,
        target_cols=None,
        seed: int | None = None,
) -> float:
    """
    Train on one fold and return validation MSE in scaled space.
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

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
        warmup_epochs=30,
    )

    # Compute validation loss (scaled space)
    model.eval()
    val_loss_total, num_val_batches = 0.0, 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, mask = x.to(device), mask.to(device)
            y_scaled = scaler.transform(y.numpy())
            y_scaled = torch.tensor(y_scaled, dtype=torch.float32, device=device)
            pred = model(x, mask=mask)
            val_loss_total += masked_mse_loss(pred, y_scaled).item()
            num_val_batches += 1

    if num_val_batches == 0:
        raise optuna.exceptions.TrialPruned()

    return val_loss_total / num_val_batches


def final_train_val_test(
        split_file: Path,
        fold_dir: Path,
        char_to_idx: dict,
        max_len: int,
        best_params: dict,
        device: torch.device,
        max_epochs: int,
        patience: int,
        target_cols=None,
        eval_fn=None,
        eval_extra_kwargs=None,
        return_metrics=False,
        seed: int | None = None,
) -> float:
    """
    Final training on train+val and evaluation on test for one fold.
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

    # build train+val set for final training
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    # train on train+val, monitor val
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
        warmup_epochs=30,
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


def objective(trial: optuna.Trial, ctx: SimpleNamespace) -> float:
    hparams = config.suggest_hparams(trial)
    if hparams["d_model"] % hparams["n_head"] != 0:
        raise optuna.exceptions.TrialPruned()

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
            target_cols=ctx.target_cols,  # multi-task: all targets
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


def run_final_folds(
        best_params: dict,
        ctx: SimpleNamespace,
        metric_keys=("mean_r2", "mean_mae", "mean_rmse"),
) -> None:
    fold_metrics = []
    split_label = getattr(ctx, "split_name", "Final")
    for fold_idx in range(ctx.fold_num):
        print(f"\n========== {split_label} | Fold {fold_idx} ==========")
        test_mse, metrics = final_train_val_test(
            split_file=ctx.data_dir / f"fold_{fold_idx}.csv",
            fold_dir=ctx.final_root / f"fold_{fold_idx}",
            char_to_idx=ctx.char_to_idx,
            max_len=ctx.max_len,
            best_params=best_params,
            device=ctx.device,
            max_epochs=ctx.max_epochs,
            patience=ctx.patience,
            target_cols=ctx.target_cols,
            eval_fn=getattr(ctx, "eval_fn", None),
            eval_extra_kwargs=getattr(ctx, "eval_extra_kwargs", None),
            return_metrics=True,
            seed=getattr(ctx, "random_seed", config.random_seed) + fold_idx,
        )
        if not np.isnan(test_mse):
            fold_metrics.append(metrics)
            print(f"[{split_label}][Fold {fold_idx}] Test MSE (original scale): {test_mse:.6f}")

    cv_metrics_path = save_cv_metrics(ctx.final_root, fold_metrics, metric_keys)
    print(f"[{split_label}][Final] Cross-validation metrics saved to: {cv_metrics_path}")




if __name__ == "__main__":
    set_seed(config.random_seed)

    exp_file = config.exp_file
    split_name = config.reference_split_name
    data_dir = config.get_split_dir(split_name)
    save_root = config.get_reference_save_root()
    fold_num = config.fold_num
    max_epochs = config.default_max_epochs
    patience = config.default_patience
    device = config.device
    target_cols = config.mtl_target_cols

    exp_df = pd.read_csv(exp_file)
    smiles_list = exp_df["SMILES"].tolist()
    charset = set(ch for smi in smiles_list for ch in smi)
    char_to_idx = {ch: i + 1 for i, ch in enumerate(sorted(charset))}
    char_to_idx["<PAD>"] = 0
    max_len = max(len(smi) for smi in smiles_list) if smiles_list else 0

    vocab_path = config.get_vocab_path()
    max_len_path = config.get_max_len_path()
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    max_len_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(char_to_idx, f)
    with open(max_len_path, "w") as f:
        f.write(str(max_len))
    print(
        f"[INFO] Transformer vocabulary built from {exp_file}. "
        f"Size: {len(char_to_idx)}, Max length: {max_len}"
    )
    print(f"[INFO] Saved vocab to: {config.get_vocab_path()}")
    print(f"[INFO] Saved max_len to: {config.get_max_len_path()}")

    ctx = SimpleNamespace(
        data_dir=data_dir,
        fold_num=fold_num,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        target_cols=target_cols,
        char_to_idx=char_to_idx,
        max_len=max_len,
        split_name=split_name,
        final_root=save_root,
        random_seed=config.random_seed,
    )

    os.makedirs(save_root, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda trial: objective(trial, ctx), n_trials=config.n_trials)

    best = study.best_trial
    print("Best trial value:", best.value)
    print("Best trial params:", best.params)

    best_params = best.params
    print(
        "Using best hyperparameters: "
        f"{config.format_params(best_params, config.structure_keys + config.training_keys)}"
    )

    save_study_results(study, save_root, best_params)

    run_final_folds(
        best_params=best_params,
        ctx=ctx,
        metric_keys=("mean_r2", "mean_mae", "mean_rmse"),
    )
