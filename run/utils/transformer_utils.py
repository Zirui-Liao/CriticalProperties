import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainer_utils import (
    compute_multitask_array_metrics,
    masked_mse_loss,
    save_fold_outputs,
)


def _clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_vocab_and_max_len(vocab_path: Path, max_len_path: Path):
    """
    Load shared Transformer vocab and max_len from fixed files.
    """
    vocab_path = Path(vocab_path)
    max_len_path = Path(max_len_path)
    missing = [str(path) for path in (vocab_path, max_len_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Transformer vocab/max_len file(s): "
            f"{missing}. Please run search_transformer_reference_optuna.py first."
        )

    with open(vocab_path, "r") as f:
        char_to_idx = json.load(f)
    with open(max_len_path, "r") as f:
        max_len_content = f.read().strip()
    if not max_len_content:
        raise ValueError(
            f"Transformer max_len file is empty: {max_len_path}. "
            "Please run search_transformer_reference_optuna.py first."
        )
    try:
        max_len = int(max_len_content)
    except ValueError as exc:
        raise ValueError(
            f"Transformer max_len file must contain an integer: {max_len_path}. "
            "Please run search_transformer_reference_optuna.py first."
        ) from exc

    print(
        f"[INFO] Loaded Transformer vocab/max_len from {vocab_path.parent}. "
        f"Vocab size: {len(char_to_idx)}, max_len: {max_len}"
    )
    return char_to_idx, max_len


def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        scaler,
        save_logs=False,
        log_path=None,
        save_model=False,
        model_path=None,
        epochs=50,
        patience=5,
        base_lr=1e-3,
        warmup_epochs=0,
):
    """
    Train the model with:
    - Label standardization via `scaler`
    - Optional warmup on learning rate
    - ReduceLROnPlateau scheduler on validation loss
    - Early stopping based on validation loss
    - Optional TensorBoard logging and checkpoint saving
    """

    if save_model:
        os.makedirs(model_path, exist_ok=True)
        best_ckpt_path = os.path.join(model_path, "best_model.pt")
        last_ckpt_path = os.path.join(model_path, "last_model.pt")
    else:
        best_ckpt_path = None
        last_ckpt_path = None

    if save_logs:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)
    else:
        writer = None

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best_val_loss = float("inf")
    best_model_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0

        # Learning rate warmup (if enabled)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr_scale = float(epoch + 1) / float(warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr * lr_scale

        # -----------------
        # Train one epoch
        # -----------------
        for x, y, mask in train_loader:
            x = x.to(device)
            mask = mask.to(device)

            # Scale labels using training scaler
            y_scaled = scaler.transform(y.cpu().numpy())
            y_scaled = torch.tensor(y_scaled, dtype=torch.float32, device=device)

            pred = model(x, mask=mask)
            loss = masked_mse_loss(pred, y_scaled)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_total += loss.item()

        train_loss_avg = train_loss_total / max(len(train_loader), 1)

        # -----------------
        # Validation
        # -----------------
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x = x.to(device)
                mask = mask.to(device)

                y_scaled = scaler.transform(y.cpu().numpy())
                y_scaled = torch.tensor(y_scaled, dtype=torch.float32, device=device)

                pred = model(x, mask=mask)
                val_loss_total += masked_mse_loss(pred, y_scaled).item()

        val_loss = val_loss_total / max(len(val_loader), 1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss_avg:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.4g}"
        )

        # Logging
        if writer is not None:
            writer.add_scalar("train_loss", train_loss_avg, epoch + 1)
            writer.add_scalar("val_loss", val_loss, epoch + 1)
            writer.add_scalar("lr-Adam", current_lr, epoch + 1)

        # Scheduler step on validation loss
        scheduler.step(val_loss)

        # Save last checkpoint
        if save_model and last_ckpt_path is not None:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                },
                last_ckpt_path,
            )

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = _clone_state_dict(model)
            no_improve = 0

            if save_model and best_ckpt_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_model_state,
                        "scaler_mean": scaler.mean_,
                        "scaler_scale": scaler.scale_,
                    },
                    best_ckpt_path,
                )
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    if writer is not None:
        writer.close()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


@torch.no_grad()
def evaluate(
        model,
        loader,
        test_smi,
        preds_path,
        scaler,
        device,
        save_test_preds=False,
        target_cols=("Tc", "Pc", "Vc", "omega"),
        return_metrics=False,
):
    """
    Evaluate the model on a given loader.
    Returns MSE on the original (unscaled) target space.
    Optionally saves test predictions to CSV.
    """
    model.eval()
    all_pred = []
    all_target = []

    for x, y, mask in loader:
        x = x.to(device)
        mask = mask.to(device)

        pred = model(x, mask=mask)
        all_pred.append(pred.cpu())
        all_target.append(y)

    if len(all_pred) == 0:
        print("Warning: empty loader for evaluation.")
        if return_metrics:
            return float("nan"), {}
        return float("nan")

    preds = torch.cat(all_pred, dim=0)  # scaled predictions
    targets = torch.cat(all_target, dim=0)  # original targets

    # Inverse-transform predictions back to physical scale
    preds_out = scaler.inverse_transform(preds.numpy())
    targets_out = targets.numpy()

    if save_test_preds:
        save_fold_outputs(
            fold_dir=Path(preds_path).parent,
            smiles=test_smi,
            preds=preds_out,
            target_cols=target_cols,
        )

    mse = np.nanmean((preds_out - targets_out) ** 2)
    if return_metrics:
        metrics = compute_multitask_array_metrics(targets_out, preds_out, target_cols)
        return mse, metrics
    return mse


