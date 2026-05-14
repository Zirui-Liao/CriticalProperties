from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def create_trainer(
    max_epochs: int,
    patience: int,
    checkpoint_dir: Path | None = None,
    default_root_dir: Path | None = None,
    enable_progress_bar: bool = False,
    precision: str | None = None,
    log_dir: Path | None = None,
) -> pl.Trainer:
    """Create the Lightning trainer used by MPNN HPO/final training."""
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            min_delta=0.0,
            verbose=False,
        )
    ]
    enable_checkpointing = checkpoint_dir is not None
    if checkpoint_dir is not None:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        )
    logger = False
    if log_dir is not None:
        logger = TensorBoardLogger(
            save_dir=str(log_dir),
            name="",
            version="",
            default_hp_metric=False,
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer_kwargs: Dict[str, Any] = {
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": enable_checkpointing,
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": 1,
        "enable_progress_bar": enable_progress_bar,
    }
    if default_root_dir is not None:
        trainer_kwargs["default_root_dir"] = str(default_root_dir)
    if precision is not None:
        trainer_kwargs["precision"] = precision
    return pl.Trainer(**trainer_kwargs)


def predict_array(
    trainer: pl.Trainer,
    model,
    loader,
    ckpt_path: str | Path | None = None,
) -> np.ndarray:
    """Run Chemprop prediction and return a single numpy array."""
    if ckpt_path is not None:
        ckpt_path = str(ckpt_path)
    preds_batches = trainer.predict(model, loader, ckpt_path=ckpt_path)
    return np.concatenate([p.detach().cpu().numpy() for p in preds_batches], axis=0)


def save_last_and_best_model_aliases(
    trainer: pl.Trainer,
    checkpoint_dir: Path,
) -> tuple[Path, Path | None]:
    """Save last_model.pt and a best_model.pt alias without changing checkpoint selection."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_model_path = checkpoint_dir / "last_model.pt"
    trainer.save_checkpoint(str(last_model_path))

    best_model_alias_path = checkpoint_dir / "best_model.pt"
    best_model_path = None
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is not None:
        best_model_path = getattr(checkpoint_callback, "best_model_path", None)
    if best_model_path:
        shutil.copy2(best_model_path, best_model_alias_path)
        best_checkpoint_path = Path(best_model_path)
        if (
            best_checkpoint_path.resolve() != best_model_alias_path.resolve()
            and best_checkpoint_path.parent.resolve() == checkpoint_dir.resolve()
        ):
            best_checkpoint_path.unlink(missing_ok=True)
        return last_model_path, best_model_alias_path

    return last_model_path, None


def load_structure_params(best_params_path: Path, structure_keys: Sequence[str]) -> Dict[str, Any]:
    """Load and keep only fixed structure parameters from a reference Optuna run."""
    if not best_params_path.is_file():
        old_best_params_path = best_params_path.with_name("best_params.json")
        if best_params_path.name != "best_hyperparams.json" or not old_best_params_path.is_file():
            raise FileNotFoundError(f"Reference best params not found: {best_params_path}")
        best_params_path = old_best_params_path

    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    missing = [key for key in structure_keys if key not in best_params]
    if missing:
        raise KeyError(f"Reference best params are missing structural keys: {missing}")
    return {key: best_params[key] for key in structure_keys}
