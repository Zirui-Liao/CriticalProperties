import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import torch

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import transformer_config as config

from search_transformer_reference_optuna import (
    compute_fold_val_mse,
    run_final_folds,
)
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    compute_single_target_array_metrics,
    save_fold_outputs,
    set_seed,
)
from utils.transformer_utils import load_vocab_and_max_len

@torch.no_grad()
def evaluate_single_prop(
        model,
        loader,
        test_smi,
        preds_path,
        scaler,
        device,
        prop_name: str,
        save_test_preds: bool = False,
        return_metrics: bool = False,
):
    """
    Evaluate single-output model on original (unscaled) space and (optionally) save predictions.
    """
    model.eval()
    preds_all, targets_all = [], []
    for x, y, mask in loader:
        x, mask = x.to(device), mask.to(device)
        pred = model(x, mask=mask)  # (B, 1) scaled
        preds_all.append(pred.cpu())
        targets_all.append(y)  # (B, 1) original

    if not preds_all:
        if return_metrics:
            return float("nan"), {}
        return float("nan")

    preds_scaled = torch.cat(preds_all, dim=0).numpy()
    targets_orig = torch.cat(targets_all, dim=0).numpy()

    preds_orig = scaler.inverse_transform(preds_scaled)

    if save_test_preds:
        save_fold_outputs(
            fold_dir=Path(preds_path).parent,
            smiles=test_smi,
            preds=preds_orig,
            target_cols=(prop_name,),
            targets=targets_orig,
            pred_suffix="_pred",
        )

    mse = float(np.nanmean((preds_orig - targets_orig) ** 2))
    if return_metrics:
        metrics = compute_single_target_array_metrics(targets_orig, preds_orig)
        return mse, metrics
    return mse


# ===============================
# 2. Per-property Optuna objective (5-fold mean val MSE)
# ===============================
def make_objective_for_property(
        prop_name: str,
        data_dir: Path,
        char_to_idx: dict,
        max_len: int,
        base_struct_hparams: dict,
        ctx: SimpleNamespace,
):
    """
    Returns an Optuna objective(trial) that:
      - Samples hyperparameters (same space as transformer_optuna.py)
      - For k in folds: trains & validates; collects val MSE (scaled space)
      - Returns mean val MSE across folds
    """

    def objective(trial: optuna.Trial):
        hparams = {**base_struct_hparams, **config.suggest_training_params(trial)}

        fold_mses = []
        for k in range(ctx.fold_num):
            split_file = data_dir / f"fold_{k}.csv"
            fold_mse = compute_fold_val_mse(
                hparams=hparams,
                split_file=split_file,
                char_to_idx=char_to_idx,
                max_len=max_len,
                device=ctx.device,
                max_epochs=ctx.max_epochs,
                patience=ctx.patience,
                target_cols=(prop_name,),
                seed=ctx.random_seed + k,
            )
            fold_mses.append(fold_mse)

        mean_val_mse = float(np.mean(fold_mses))
        trial.set_user_attr("fold_rmse_list", fold_mses)

        print(
            f"[Trial {trial.number}][{prop_name}] "
            f"{config.format_params(hparams, config.structure_keys + config.training_keys)} | "
            f"mean_val_mse={mean_val_mse:.6f}"
        )

        return mean_val_mse

    return objective


# ===============================
# 3. Main: per-property search + per-fold final training/eval
# ===============================


if __name__ == "__main__":
    set_seed(config.random_seed)

    properties = config.stl_properties
    fold_num = config.fold_num
    max_epochs = config.default_max_epochs
    patience = config.default_patience
    device = config.device
    base_struct_hparams = config.load_structure_params(config.get_reference_best_params_path())
    ctx = SimpleNamespace(
        fold_num=fold_num,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        random_seed=config.random_seed,
    )

    for prop in properties:
        print(f"\n===== Property: {prop} =====")
        split_settings = [
            {
                "split_name": f"kmeans_{prop}_no_outliers_035",
                "data_dir": config.project_root / "data_folds" / f"kmeans_{prop}_no_outliers_035",
                "save_root": config.project_root / "transformer_models" / f"kmeans_{prop}_no_outliers_035_optuna",
            },
            {
                "split_name": f"random_{prop}_no_outliers_035",
                "data_dir": config.project_root / "data_folds" / f"random_{prop}_no_outliers_035",
                "save_root": config.project_root / "transformer_models" / f"random_{prop}_no_outliers_035_optuna",
            },
        ]

        char_to_idx, max_len = load_vocab_and_max_len(
            vocab_path=config.get_vocab_path(),
            max_len_path=config.get_max_len_path(),
        )

        for split_setting in split_settings:
            split_name = split_setting["split_name"]
            data_dir = split_setting["data_dir"]
            save_root = split_setting["save_root"]
            print(f"\n===== Property: {prop} | Split: {split_name} =====")
            os.makedirs(save_root, exist_ok=True)

            # ---- Optuna over 5 folds (minimize mean val MSE) ----
            sampler = optuna.samplers.TPESampler(seed=config.random_seed)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                make_objective_for_property(prop, data_dir, char_to_idx, max_len, base_struct_hparams, ctx),
                n_trials=config.n_trials,
            )

            best = study.best_trial
            best_params = {**base_struct_hparams, **best.params}

            print(f"[{prop}][{split_name}] Best value: {best.value:.6f}")
            print(f"[{prop}][{split_name}] Best params (struct + training): {best_params}")

            save_study_results(
                study,
                save_root,
                best_params,
                params_label="Best (struct+train) params",
            )

            final_ctx = SimpleNamespace(
                data_dir=data_dir,
                final_root=save_root,
                fold_num=fold_num,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
                target_cols=(prop,),
                char_to_idx=char_to_idx,
                max_len=max_len,
                split_name=f"{prop} | {split_name}",
                eval_fn=evaluate_single_prop,
                eval_extra_kwargs={"prop_name": prop},
            )
            run_final_folds(
                best_params=best_params,
                ctx=final_ctx,
                metric_keys=("r2", "mae", "rmse"),
            )
