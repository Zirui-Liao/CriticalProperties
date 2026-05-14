from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seed Python, NumPy, Torch, CUDA, and return a DataLoader generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs inside DataLoader workers from Torch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute R2, MSE, RMSE, and MAE for a regression prediction vector."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_samples = len(y_true_arr)
    if n_samples == 0:
        return {
            "n_samples": 0,
            "r2": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }

    mse = mean_squared_error(y_true_arr, y_pred_arr)
    return {
        "n_samples": n_samples,
        "r2": float("nan") if n_samples < 2 else float(r2_score(y_true_arr, y_pred_arr)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
    }


def masked_mse_loss(pred, target):
    """
    MSE loss that ignores target positions with NaN.

    pred, target: tensors with the same shape, typically (batch_size, num_tasks).
    """
    mask = ~torch.isnan(target)
    diff = (pred - target)[mask]
    if diff.numel() == 0:
        return pred.sum() * 0.0
    return (diff ** 2).mean()


def load_split_data(split_file, target_cols=("Tc", "Pc", "Vc", "omega")):
    """
    Load a split CSV and return (train/val/test) SMILES and targets.
    """
    import pandas as pd

    df = pd.read_csv(split_file)

    cols = ["SMILES", "split", *target_cols]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {split_file}: {missing}")

    smiles = df["SMILES"]
    splits = df["split"]
    y_all = df[list(target_cols)].to_numpy(dtype=np.float32)

    def select(tag: str):
        mask = splits == tag
        return smiles[mask].tolist(), y_all[mask]

    train_smi, train_y = select("train")
    val_smi, val_y = select("val")
    test_smi, test_y = select("test")

    return train_smi, train_y, val_smi, val_y, test_smi, test_y


def save_fold_outputs(
    fold_dir: Path,
    smiles: Sequence[str],
    preds: np.ndarray,
    target_cols: Sequence[str],
    targets: np.ndarray | None = None,
    pred_suffix: str | None = None,
    target_suffix: str = "_target",
) -> Path:
    """Save per-fold test_predictions.csv."""
    import pandas as pd

    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    preds = np.asarray(preds)
    preds_df = pd.DataFrame({"SMILES": list(smiles)})
    for i, col in enumerate(target_cols):
        pred_col = f"{col}{pred_suffix}" if pred_suffix else col
        preds_df[pred_col] = preds[:, i] if preds.ndim > 1 else preds.reshape(-1)

    if targets is not None:
        targets = np.asarray(targets)
        for i, col in enumerate(target_cols):
            target_col = f"{col}{target_suffix}"
            preds_df[target_col] = targets[:, i] if targets.ndim > 1 else targets.reshape(-1)

    preds_path = fold_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    return preds_path


def _json_safe(value: Any) -> Any:
    """Convert numpy scalar containers to plain Python objects for json.dump."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def compute_multitask_array_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    target_cols: Sequence[str],
) -> Dict[str, float]:
    """Compute per-task and mean R2/MAE/RMSE/MSE from target and prediction arrays."""
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)

    metrics: Dict[str, float] = {}
    r2_list = []
    mae_list = []
    rmse_list = []
    mse_list = []

    for i, col in enumerate(target_cols):
        yt = y_true[:, i]
        yp = preds[:, i]
        mask = np.isfinite(yt)
        if mask.sum() == 0:
            metrics[f"{col}_r2"] = float("nan")
            metrics[f"{col}_mae"] = float("nan")
            metrics[f"{col}_rmse"] = float("nan")
            metrics[f"{col}_mse"] = float("nan")
            continue

        values = regression_metrics(yt[mask], yp[mask])
        metrics[f"{col}_r2"] = values["r2"]
        metrics[f"{col}_mae"] = values["mae"]
        metrics[f"{col}_rmse"] = values["rmse"]
        metrics[f"{col}_mse"] = values["mse"]

        if np.isfinite(values["r2"]):
            r2_list.append(values["r2"])
        mae_list.append(values["mae"])
        rmse_list.append(values["rmse"])
        mse_list.append(values["mse"])

    if r2_list:
        metrics["mean_r2"] = float(np.mean(r2_list))
    else:
        metrics["mean_r2"] = float("nan")

    if mae_list:
        metrics["mean_mae"] = float(np.mean(mae_list))
        metrics["mean_rmse"] = float(np.mean(rmse_list))
        metrics["mean_mse"] = float(np.mean(mse_list))
    else:
        metrics["mean_mae"] = float("nan")
        metrics["mean_rmse"] = float("nan")
        metrics["mean_mse"] = float("nan")

    return metrics


def compute_single_target_array_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
) -> Dict[str, float]:
    """Compute R2/MAE/RMSE for one target array, ignoring NaN targets."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(preds, dtype=float).reshape(-1)
    mask = np.isfinite(y_true)
    if mask.sum() < 2:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    return regression_metrics(y_true[mask], y_pred[mask])


def save_cv_metrics(
    save_root: Path,
    fold_metrics: Sequence[Dict[str, float]],
    metric_keys: Sequence[str],
) -> Path:
    """Save cross-validation metric summary in the existing JSON format."""
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    safe_fold_metrics = [_json_safe(metrics) for metrics in fold_metrics]
    agg: Dict[str, Any] = {"fold_metrics": safe_fold_metrics}
    for key in metric_keys:
        values = [
            metrics[key]
            for metrics in safe_fold_metrics
            if key in metrics and np.isfinite(metrics[key])
        ]
        if values:
            agg[f"cv_{key}_mean"] = float(np.mean(values))
            agg[f"cv_{key}_std"] = float(np.std(values))
        else:
            agg[f"cv_{key}_mean"] = float("nan")
            agg[f"cv_{key}_std"] = float("nan")

    cv_metrics_path = save_root / "cv_metrics.json"
    with open(cv_metrics_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return cv_metrics_path


def load_pretrained_weights_into_model(model, ckpt_path: str | Path) -> None:
    """Load pretrained weights while keeping fold-specific output transforms intact."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if "predictor.output_transform" not in key
    }

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"[Pretrain] Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[Pretrain] Unexpected keys when loading state_dict: {unexpected}")
