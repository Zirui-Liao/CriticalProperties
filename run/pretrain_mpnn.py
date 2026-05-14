from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from chemprop import data as cp_data
from chemprop import featurizers
from sklearn.model_selection import train_test_split

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import mpnn_config as config
from models.mpnn_model import build_mpnn_from_params
from utils.mpnn_utils import (
    create_trainer,
    load_structure_params,
    save_last_and_best_model_aliases,
)
from utils.trainer_utils import set_seed


def pretrain_on_est(struct_params: Dict[str, Any]) -> str:
    set_seed(config.pretrain_random_seed)

    print("[Pretrain] Loading pretraining data from:", config.pretrain_file)
    df = pd.read_csv(config.pretrain_file)

    required_cols = ["SMILES", *config.mtl_target_cols]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Pretraining file is missing columns: {missing}")

    train_df, val_df = train_test_split(
        df,
        test_size=config.pretrain_val_fraction,
        random_state=config.pretrain_random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(
        f"[Pretrain] Train size: {len(train_df)}, "
        f"Val size: {len(val_df)} (val fraction={config.pretrain_val_fraction:.2f})"
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_smi = train_df["SMILES"].astype(str).tolist()
    train_y = train_df[list(config.mtl_target_cols)].to_numpy(dtype=float)
    val_smi = val_df["SMILES"].astype(str).tolist()
    val_y = val_df[list(config.mtl_target_cols)].to_numpy(dtype=float)
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
    train_loader = cp_data.build_dataloader(train_ds, batch_size=config.pretrain_batch_size, num_workers=config.num_workers, shuffle=True)
    val_loader = cp_data.build_dataloader(val_ds, batch_size=config.pretrain_batch_size, num_workers=config.num_workers, shuffle=False)

    train_params = {
        "init_lr": config.pretrain_init_lr,
        "max_lr": config.pretrain_max_lr,
        "final_lr": config.pretrain_final_lr,
        "warmup_epochs": config.pretrain_warmup_epochs,
    }
    model = build_mpnn_from_params(n_tasks=len(config.mtl_target_cols), scaler=scaler,
                                   hparams={**struct_params, **train_params})

    pretrain_root = config.get_pretrain_root()
    ckpt_dir = pretrain_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_trainer(
        max_epochs=config.pretrain_max_epochs,
        patience=config.pretrain_patience,
        checkpoint_dir=ckpt_dir,
        default_root_dir=pretrain_root,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    last_model_path, best_model_path = save_last_and_best_model_aliases(
        trainer,
        ckpt_dir,
    )
    if best_model_path is None:
        best_model_path = config.get_pretrained_checkpoint_path()

    print(f"[Pretrain] Finished. Best checkpoint: {best_model_path}")
    print(f"[Pretrain] Finished. Last checkpoint: {last_model_path}")
    return str(best_model_path)




if __name__ == "__main__":
    set_seed(config.pretrain_random_seed)

    struct_params = load_structure_params(
        config.get_reference_best_params_path(),
        config.structure_keys,
    )
    print("[INFO] Loaded structural params from multi-task mpnn_optuna:", struct_params)

    print("=" * 80)
    print("[Pretrain] Pretraining on pretraining_data_035.csv")
    print("=" * 80)
    pretrained_ckpt = pretrain_on_est(struct_params)
    print(f"[Pretrain] Checkpoint saved to: {pretrained_ckpt}")
