from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from trainer_utils import masked_mse_loss


# ========== Fingerprint / feature utilities ==========


class MolDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y


def collate_fn(batch):
    features = np.stack([b[0] for b in batch], axis=0)
    targets = np.stack([b[1] for b in batch], axis=0)
    return torch.from_numpy(features), torch.from_numpy(targets)


def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Convert SMILES to Morgan bit vector (np.ndarray, shape=(n_bits,))."""
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_fold_arrays(input_dir: Path, fold: int, prop: str,
                      radius: int = 2, n_bits: int = 1024) -> Tuple[
    pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load one fold of data, generate X (fingerprints), y (target), and boolean masks for train/val/test.
    """
    df = pd.read_csv(input_dir / f'fold_{fold}.csv')
    df_valid = df.reset_index(drop=True)
    fps = df_valid['SMILES'].apply(lambda s: compute_morgan_fp(s, radius, n_bits)).tolist()
    X = np.array(fps)
    y = df_valid[prop].values

    masks = {
        'train': (df_valid['split'] == 'train').to_numpy(),
        'val': (df_valid['split'] == 'val').to_numpy(),
        'test': (df_valid['split'] == 'test').to_numpy(),
    }
    return df_valid, X, y, masks


# ========== Scaling utilities (for MLP / SVM only) ==========

def fit_transform_scalers(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray):
    """
    Fit X/y scalers and return scaled train/val sets.
    Returns (X_train_s, y_train_s, X_val_s, y_val_s, X_scaler, y_scaler).
    """
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = X_scaler.fit_transform(X_train)
    X_val_s = X_scaler.transform(X_val)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    return X_train_s, y_train_s, X_val_s, y_val_s, X_scaler, y_scaler


def _clone_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_with_earlystop(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        n_epochs: int = 200,
        patience: int = 30,
        save_logs: bool = False,
        log_path=None,
        save_model: bool = False,
        model_path=None,
        feature_scaler=None,
        target_scalers=None,
):
    """
    Train multitask fingerprint MLP with masked-MSE early stopping.

    When save_model=True, write last_model.pt every epoch and best_model.pt
    whenever validation loss improves, matching the Transformer checkpoint flow.
    """
    if save_model:
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = model_path / "best_model.pt"
        last_ckpt_path = model_path / "last_model.pt"
    else:
        best_ckpt_path = None
        last_ckpt_path = None

    if save_logs:
        from torch.utils.tensorboard import SummaryWriter

        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_path))
    else:
        writer = None

    best_val_mse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = masked_mse_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_targets, val_preds = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)

                val_targets.append(y_batch.cpu().numpy())
                val_preds.append(preds.cpu().numpy())

        if not val_targets:
            break

        val_targets = np.vstack(val_targets)
        val_preds = np.vstack(val_preds)
        val_mse = masked_mse_loss(
            torch.as_tensor(val_preds, dtype=torch.float32),
            torch.as_tensor(val_targets, dtype=torch.float32),
        ).item()
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_num = epoch + 1

        print(
            f"Epoch {epoch_num} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_mse:.6f} | "
            f"LR: {current_lr:.4g}"
        )

        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch_num)
            writer.add_scalar("val_loss", val_mse, epoch_num)
            writer.add_scalar("lr-Adam", current_lr, epoch_num)

        improved = val_mse + 1e-6 < best_val_mse
        if improved:
            best_val_mse = val_mse
            wait = 0
            best_state = _clone_state_dict(model)

            if save_model and best_ckpt_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "feature_scaler": feature_scaler,
                        "target_scalers": target_scalers,
                    },
                    best_ckpt_path,
                )
        else:
            wait += 1

        if save_model and last_ckpt_path is not None:
            torch.save(
                {
                    "model_state_dict": _clone_state_dict(model),
                    "feature_scaler": feature_scaler,
                    "target_scalers": target_scalers,
                },
                last_ckpt_path,
            )

        if not improved and wait >= patience:
            print("Early stopping triggered.")
            break

    if writer is not None:
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_mse


# ========== Early stopping for sklearn MLP ==========

def fit_mlp_with_val(
        model: MLPRegressor,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        patience: int = 20,
        max_epochs: int = 1000,
        tol: float = 1e-6,
) -> MLPRegressor:
    """
    Train MLP with validation-based early stopping using partial_fit().
    Use validation MSE as the criterion (lower is better).
    Save the best weights (based on val MSE) and restore them after patience epochs.
    """
    model = clone(model)
    model.warm_start = True
    model.max_iter = 1

    best_val_loss = np.inf
    wait = 0
    best_coefs_ = None
    best_intercepts_ = None

    for epoch in range(max_epochs):
        model.partial_fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)

        if val_mse < best_val_loss - tol:
            best_val_loss = val_mse
            wait = 0
            best_coefs_ = [c.copy() for c in model.coefs_]
            best_intercepts_ = [b.copy() for b in model.intercepts_]
        else:
            wait += 1
            if wait >= patience:
                break

    if best_coefs_ is not None and best_intercepts_ is not None:
        model.coefs_ = best_coefs_
        model.intercepts_ = best_intercepts_

    model.n_iter_ = max(1, epoch - patience + 1)
    return model
