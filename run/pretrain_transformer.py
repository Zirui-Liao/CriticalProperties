import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
from utils.trainer_utils import seed_worker, set_seed
from utils.transformer_utils import load_vocab_and_max_len, train_model


def pretrain_on_est(
    est_file: Path,
    char_to_idx: dict,
    max_len: int,
    base_struct_hparams: dict,
    device: torch.device,
) -> Path:
    if not est_file.exists():
        raise FileNotFoundError(f"Pretraining file not found: {est_file}")

    df = pd.read_csv(est_file)
    required_cols = ["SMILES", *config.mtl_target_cols]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in pretraining file {est_file}: {missing}")

    smiles_all = df["SMILES"].tolist()
    targets_all = df[list(config.mtl_target_cols)].to_numpy(dtype=np.float32)

    train_smi, val_smi, train_y, val_y = train_test_split(
        smiles_all,
        targets_all,
        test_size=0.1,
        random_state=config.random_seed,
    )

    pretrain_save_dir = config.project_root / "transformer_models" / config.pretrain_tag
    pretrain_save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = pretrain_save_dir / "trainer_logs"
    ckpt_dir = pretrain_save_dir / "checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = SmilesDataset(train_smi, train_y, char_to_idx, max_len=max_len)
    val_ds = SmilesDataset(val_smi, val_y, char_to_idx, max_len=max_len)

    generator = set_seed(config.random_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.pretrain_batch_size,
        shuffle=True,
        collate_fn=lambda batch: tuple(torch.stack(items) for items in zip(*batch)),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.pretrain_batch_size,
        shuffle=False,
        collate_fn=lambda batch: tuple(torch.stack(items) for items in zip(*batch)),
        worker_init_fn=seed_worker,
    )

    y_train_list = [y for _, y, _ in train_loader]
    if not y_train_list:
        raise ValueError("No training data found for pretraining.")
    y_train = torch.cat(y_train_list, dim=0).numpy()
    label_scaler = StandardScaler().fit(y_train)

    model = TransformerEncoderModel(
        vocab_size=len(char_to_idx),
        max_len=max_len,
        output_dim=len(config.mtl_target_cols),
        **config.select_params(base_struct_hparams, config.structure_keys),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scaler=label_scaler,
        save_logs=True,
        log_path=log_dir,
        save_model=True,
        model_path=ckpt_dir,
        epochs=config.pretrain_epochs,
        patience=config.pretrain_patience,
        base_lr=config.pretrain_lr,
        warmup_epochs=config.pretrain_warmup_epochs,
    )

    best_ckpt_path = ckpt_dir / "best_model.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Expected best_model.pt not found in {ckpt_dir}.")

    print(f"[Pretrain] Finished. Best checkpoint at: {best_ckpt_path}")
    return best_ckpt_path




if __name__ == "__main__":
    set_seed(config.random_seed)

    char_to_idx, max_len = load_vocab_and_max_len(
        vocab_path=config.get_vocab_path(),
        max_len_path=config.get_max_len_path(),
    )
    best_path = config.get_reference_best_params_path()
    if not best_path.exists():
        raise FileNotFoundError(
            f"Cannot find best_hyperparams.json at {best_path}. "
            "Please run search_transformer_reference_optuna.py first."
        )
    base_struct_hparams = config.load_structure_params(best_path)
    pretrain_ckpt_path = pretrain_on_est(
        est_file=config.pretrain_file,
        char_to_idx=char_to_idx,
        max_len=max_len,
        base_struct_hparams=base_struct_hparams,
        device=config.device,
    )
    print(f"[Pretrain] Checkpoint saved to: {pretrain_ckpt_path}")
