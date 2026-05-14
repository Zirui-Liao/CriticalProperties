import os
import sys
from pathlib import Path
import optuna
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import fp_config as config
from models.fingerprint_models import MLPMultiTask
from utils.fp_utils import MolDataset, build_fold_arrays, collate_fn, train_with_earlystop
from utils.optuna_utils import save_study_results
from utils.trainer_utils import (
    compute_multitask_array_metrics,
    masked_mse_loss,
    save_cv_metrics,
    seed_worker,
    set_seed,
)

# -------------------------
# Training / Optuna objective
# -------------------------
def make_objective(folds_data, tasks, device, n_epochs, patience, random_seed):
    def objective(trial):
        # hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 3)
        units = []
        for i in range(n_layers):
            units.append(trial.suggest_int(f"units_l{i + 1}", 32, 512 if i == 0 else units[-1], step=32))
        params = {
            "n_layers": n_layers,
            **{f"units_l{i + 1}": u for i, u in enumerate(units)},
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        n_layers = params["n_layers"]
        units = [params[f"units_l{i + 1}"] for i in range(n_layers)]
        dropout = params["dropout"]
        lr = params["lr"]
        weight_decay = params["weight_decay"]
        batch_size = params["batch_size"]

        fold_mses = []

        for fd in folds_data:
            fold = fd["fold"]
            generator = set_seed(random_seed + fold)

            features_all = fd["features_all"]
            targets_all = fd["targets_all"]
            masks = fd["masks"]

            train_mask = masks['train']
            val_mask = masks['val']

            x_train = features_all[train_mask]
            x_val = features_all[val_mask]
            y_train = targets_all[train_mask]
            y_val = targets_all[val_mask]

            if x_train.shape[0] == 0 or x_val.shape[0] == 0:
                continue

            # --- X scaler (per fold) ---
            x_scaler = StandardScaler().fit(x_train)
            x_train_scaled = x_scaler.transform(x_train)
            x_val_scaled = x_scaler.transform(x_val)

            # --- Y scalers (per task, per fold) ---
            y_train_scaled = y_train.copy()
            y_val_scaled = y_val.copy()
            y_scalers = {}
            for t_idx, tname in enumerate(tasks):
                col = y_train[:, t_idx]
                mask = ~np.isnan(col)
                if mask.sum() > 0:
                    scaler = StandardScaler().fit(col[mask].reshape(-1, 1))
                    y_scalers[tname] = scaler
                    # apply to train/val
                    for src, dst in [(y_train, y_train_scaled), (y_val, y_val_scaled)]:
                        col_arr = src[:, t_idx]
                        m2 = ~np.isnan(col_arr)
                        if m2.sum() > 0:
                            dst[m2, t_idx] = scaler.transform(col_arr[m2].reshape(-1, 1)).ravel()
                else:
                    y_scalers[tname] = None

            # --- build loaders ---
            train_ds = MolDataset(x_train_scaled, y_train_scaled)
            val_ds = MolDataset(x_val_scaled, y_val_scaled)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                generator=generator,
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                worker_init_fn=seed_worker,
            )

            # --- build & train model ---
            input_dim = x_train_scaled.shape[1]
            model = MLPMultiTask(
                input_dim=input_dim,
                output_dim=len(tasks),
                hidden_sizes=tuple(units),
                dropout=dropout,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            model, _ = train_with_earlystop(
                model, optimizer, train_loader, val_loader,
                device=device,
                n_epochs=n_epochs, patience=patience
            )

            # --- compute val predictions & MSE in original scale ---
            model.eval()
            all_preds_s, all_targets_s = [], []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(x_batch)
                    all_preds_s.append(preds.cpu().numpy())
                    all_targets_s.append(y_batch.cpu().numpy())

            if not all_preds_s:
                continue

            preds_s = np.vstack(all_preds_s)  # scaled preds
            y_val_scaled_batch = np.vstack(all_targets_s)
            fold_mse = masked_mse_loss(
                torch.as_tensor(preds_s, dtype=torch.float32),
                torch.as_tensor(y_val_scaled_batch, dtype=torch.float32),
            ).item()
            if not np.isnan(fold_mse):
                fold_mses.append(fold_mse)

        if not fold_mses:
            return float("inf")

        mean_val_mse = float(np.mean(fold_mses))
        trial.set_user_attr("fold_rmse_list", fold_mses)
        print(f"[Trial {trial.number}] mean_val_mse={mean_val_mse:.6f}")
        return mean_val_mse
    return objective





# -------------------------
# Main flow: loop folds, optuna, train final on train+val, test evaluate
# -------------------------


if __name__ == "__main__":
    random_seed = config.random_seed
    tasks = config.target_columns
    fold_num = config.fold_num
    max_trials = config.mlp_mtl_max_trials
    n_epochs = config.mlp_mtl_epochs
    patience = config.mlp_mtl_patience
    default_batch_size = config.mlp_mtl_batch_size
    device = config.mlp_mtl_device

    split_settings = [
        {
            "split_name": "kmeans_no_outliers_035",
            "data_dir": config.project_root / "data_folds" / "kmeans_no_outliers_035",
            "save_root": config.mlp_mtl_output_root.parent
            / config.mlp_mtl_output_root.name.replace("random", "kmeans"),
        },
        {
            "split_name": "random_no_outliers_035",
            "data_dir": config.project_root / "data_folds" / "random_no_outliers_035",
            "save_root": config.mlp_mtl_output_root,
        },
    ]

    for split_setting in split_settings:
        split_name = split_setting["split_name"]
        data_dir = split_setting["data_dir"]
        save_root = split_setting["save_root"]

        print(f"\n==================== Split: {split_name} ====================")

        set_seed(random_seed)
        os.makedirs(save_root, exist_ok=True)
        fold_metrics = []

        # -------- Stage 1: collect data for all folds --------
        folds_data = []

        for fold in range(fold_num):
            df_valid, features_all, _, masks = build_fold_arrays(data_dir, fold, prop='Tc')

            targets_all = np.zeros((len(df_valid), len(tasks)), dtype=np.float32)
            targets_all[:] = np.nan
            for j, t in enumerate(tasks):
                if t in df_valid.columns:
                    col = pd.to_numeric(df_valid[t].values, errors='coerce')
                    targets_all[:, j] = col.astype(np.float32)

            folds_data.append({
                "fold": fold,
                "df_valid": df_valid,
                "features_all": features_all,
                "targets_all": targets_all,
                "masks": masks,
            })

        # -------- Stage 2: Optuna over all folds (mean val MSE) --------
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)

        study.optimize(
            make_objective(folds_data, tasks, device, n_epochs, patience, random_seed),
            n_trials=max_trials,
            show_progress_bar=True,
        )
        best_params = study.best_params
        print(f"[{split_name}] Best hyperparams:", best_params)
        save_study_results(
            study,
            save_root,
            best_params,
            params_label=f"{split_name} best hyperparams",
        )

        # -------- Stage 3: final training per fold with best_params --------
        for fd in folds_data:
            fold = fd["fold"]
            print(f"\n=== {split_name} | Fold {fold} final training ===")

            df_valid = fd["df_valid"]
            features_all = fd["features_all"]
            targets_all = fd["targets_all"]
            masks = fd["masks"]

            train_mask = masks['train']
            val_mask = masks['val']
            test_mask = masks['test']

            # train+val as training data
            train_val_mask = train_mask | val_mask
            x_trainval = features_all[train_val_mask]
            y_trainval = targets_all[train_val_mask]
            x_val = features_all[val_mask]
            y_val = targets_all[val_mask]
            x_test = features_all[test_mask]
            y_test = targets_all[test_mask]

            fold_out = save_root / f'fold_{fold}'
            checkpoints_dir = fold_out / 'checkpoints'
            log_dir = fold_out / 'trainer_logs'
            os.makedirs(checkpoints_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            # X scaler on train+val
            x_scaler_final = StandardScaler().fit(x_trainval)
            x_trainval_scaled = x_scaler_final.transform(x_trainval)
            x_val_scaled = x_scaler_final.transform(x_val)

            # Y scalers per task on train+val
            y_scalers_final = {}
            y_trainval_scaled = y_trainval.copy()
            y_val_scaled = y_val.copy()
            for t_idx, tname in enumerate(tasks):
                col = y_trainval[:, t_idx]
                mask = ~np.isnan(col)
                if mask.sum() > 0:
                    scaler = StandardScaler().fit(col[mask].reshape(-1, 1))
                    y_scalers_final[tname] = scaler
                    for src, dst in [(y_trainval, y_trainval_scaled), (y_val, y_val_scaled)]:
                        col_arr = src[:, t_idx]
                        m2 = ~np.isnan(col_arr)
                        if m2.sum() > 0:
                            dst[m2, t_idx] = scaler.transform(col_arr[m2].reshape(-1, 1)).ravel()
                else:
                    y_scalers_final[tname] = None

            # unpack best hyperparameters
            n_layers = best_params.get("n_layers", 1)
            units = [best_params[f"units_l{i + 1}"] for i in range(n_layers)]
            dropout = best_params.get("dropout", 0.0)
            lr = best_params.get("lr", 1e-3)
            weight_decay = best_params.get("weight_decay", 1e-6)
            batch_size = best_params.get("batch_size", default_batch_size)

            input_dim = x_trainval_scaled.shape[1]
            generator = set_seed(random_seed + fold)
            final_model = MLPMultiTask(
                input_dim=input_dim,
                output_dim=len(tasks),
                hidden_sizes=tuple(units),
                dropout=dropout,
            ).to(device)
            optimizer = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)

            train_ds = MolDataset(x_trainval_scaled, y_trainval_scaled)
            val_ds = MolDataset(x_val_scaled, y_val_scaled)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                generator=generator,
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                worker_init_fn=seed_worker,
            )

            # we still use val_loader from train data for early stopping
            final_model, best_val_score = train_with_earlystop(
                final_model, optimizer, train_loader, val_loader,
                device=device,
                n_epochs=n_epochs, patience=patience,
                save_logs=True,
                log_path=log_dir,
                save_model=True,
                model_path=checkpoints_dir,
                feature_scaler=x_scaler_final,
                target_scalers=y_scalers_final,
            )

            best_model_path = checkpoints_dir / 'best_model.pt'
            try:
                best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            except TypeError:
                best_checkpoint = torch.load(best_model_path, map_location=device)
            final_model.load_state_dict(best_checkpoint["model_state_dict"])
            feature_scaler = best_checkpoint["feature_scaler"]
            target_scalers = best_checkpoint["target_scalers"]

            # Test evaluation uses the best checkpoint selected by early stopping.
            final_model.eval()
            with torch.no_grad():
                x_test_scaled = feature_scaler.transform(x_test)
                x_test_tensor = torch.from_numpy(x_test_scaled.astype(np.float32)).to(device)
                preds_test_s = final_model(x_test_tensor).cpu().numpy()

            preds_test_inv = preds_test_s.copy()
            for t_idx, tname in enumerate(tasks):
                scaler = target_scalers.get(tname)
                mask = ~np.isnan(y_test[:, t_idx])
                if not mask.any():
                    preds_test_inv[:, t_idx] = preds_test_s[:, t_idx]
                    continue
                if scaler is not None:
                    preds_test_inv[mask, t_idx] = scaler.inverse_transform(
                        preds_test_s[mask, t_idx].reshape(-1, 1)
                    ).ravel()
                else:
                    preds_test_inv[mask, t_idx] = preds_test_s[mask, t_idx]

            metrics = compute_multitask_array_metrics(y_test, preds_test_inv, tasks)

            smiles_test = df_valid.loc[test_mask, 'SMILES'].reset_index(drop=True)
            preds_df = pd.DataFrame(preds_test_inv, columns=tasks)
            preds_df.insert(0, 'SMILES', smiles_test)
            preds_df.to_csv(fold_out / 'test_predictions.csv', index=False)

            summary = {
                'fold': fold,
                'val_score_at_best': float(best_val_score) if not np.isnan(best_val_score) else None,
                'device': str(device),
            }
            summary.update(metrics)

            pd.DataFrame([summary]).to_csv(fold_out / 'metrics.csv', index=False)
            fold_metrics.append(summary)
            print(f"Fold {fold} test metrics: {metrics}")

        metric_keys = []
        for t in tasks:
            metric_keys.extend([f"{t}_r2", f"{t}_mae", f"{t}_rmse", f"{t}_mse"])
        metric_keys.extend(["mean_r2", "mean_mae", "mean_rmse", "mean_mse"])
        cv_metrics_path = save_cv_metrics(save_root, fold_metrics, metric_keys)
        print(f"[Final][{split_name}] Cross-validation metrics saved to: {cv_metrics_path}")
        print("\nSplit finished. Results saved to:", save_root)
