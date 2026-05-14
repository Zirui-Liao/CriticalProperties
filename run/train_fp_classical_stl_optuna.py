import os
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

run_dir = Path(__file__).resolve().parent
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

from configs import fp_config as config
from utils.fp_utils import (
    build_fold_arrays,
    fit_mlp_with_val,
    fit_transform_scalers,
)
from utils.trainer_utils import compute_single_target_array_metrics, save_cv_metrics, set_seed
from utils.optuna_utils import save_study_results


def multi_fold_optimize_optuna(model_name: str, folds_data, n_trials: int = 30, seed: int = 42, output_dir=None):
    """
    Run Optuna once per property/model using all folds.

    Objective remains the original mean validation MSE over folds for one
    shared hyperparameter set.
    """
    set_seed(seed)

    def build_mlp_params(params):
        n_layers = params["mlp_n_layers"]
        units = [params[f"mlp_n_units_l{i}"] for i in range(1, n_layers + 1)]
        return {
            "hidden_layer_sizes": tuple(units),
            "activation": params["activation"],
            "alpha": params["alpha"],
            "learning_rate_init": params["learning_rate_init"],
            "random_state": seed,
        }

    def make_model_and_fit_one_fold(params, x_train, y_train, x_val, y_val):
        if model_name == "mlp":
            base_model = MLPRegressor(**build_mlp_params(params))
            x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_scaler, y_scaler = fit_transform_scalers(
                x_train, y_train, x_val, y_val
            )
            model = fit_mlp_with_val(
                base_model,
                x_train_scaled,
                y_train_scaled,
                x_val_scaled,
                y_val_scaled,
                patience=20,
            )
            y_pred_scaled = model.predict(x_val_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            return model, y_pred, x_scaler, y_scaler

        if model_name == "svm":
            model = SVR(**params)
            x_train_scaled, y_train_scaled, x_val_scaled, _, x_scaler, y_scaler = fit_transform_scalers(
                x_train, y_train, x_val, y_val
            )
            model.fit(x_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(x_val_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            return model, y_pred, x_scaler, y_scaler

        if model_name == "rf":
            model = RandomForestRegressor(**params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            return model, y_pred, None, None

        if model_name == "xgboost":
            model = XGBRegressor(**params)
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            y_pred = model.predict(x_val)
            return model, y_pred, None, None

        raise ValueError(f"Unknown model_name: {model_name}")

    def objective(trial):
        if model_name == "mlp":
            n_layers = trial.suggest_int("mlp_n_layers", 1, 3)
            units = []
            for i in range(1, n_layers + 1):
                units.append(trial.suggest_int(f"mlp_n_units_l{i}", 32, 256 if i == 1 else units[-1], step=32))
            params = {
                "mlp_n_layers": n_layers,
                **{f"mlp_n_units_l{i + 1}": u for i, u in enumerate(units)},
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            }
        elif model_name == "svm":
            params = {
                "C": trial.suggest_float("C", 0.1, 100, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "kernel": "rbf",
            }
        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "n_jobs": -1,
                "random_state": seed,
            }
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "n_jobs": -1,
                "objective": "reg:squarederror",
                "verbosity": 0,
                "use_label_encoder": False,
                "eval_metric": "rmse",
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "random_state": seed,
            }
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        fold_mses = []

        for fold_data in folds_data:
            x_train, y_train = fold_data["x_train"], fold_data["y_train"]
            x_val, y_val = fold_data["x_val"], fold_data["y_val"]

            if x_train.size == 0 or x_val.size == 0:
                return float(np.inf)

            _, y_pred, _, _ = make_model_and_fit_one_fold(params, x_train, y_train, x_val, y_val)
            fold_mses.append(mean_squared_error(y_val, y_pred))

        if not fold_mses:
            return float(np.inf)

        mean_val_mse = float(np.mean(fold_mses))
        trial.set_user_attr("fold_rmse_list", fold_mses)
        print(f"[Trial {trial.number}][{model_name}] mean_val_mse={mean_val_mse:.6f}")
        return mean_val_mse

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"[Optuna][{model_name}] Best params: {best_params}")
    if output_dir is not None:
        save_study_results(
            study,
            Path(output_dir),
            best_params,
            params_label=f"{model_name} best params",
        )

    final_models = []
    scaler_list = []

    for fold_data in folds_data:
        x_train, y_train = fold_data["x_train"], fold_data["y_train"]
        x_val, y_val = fold_data["x_val"], fold_data["y_val"]
        x_all = np.vstack((x_train, x_val))
        y_all = np.concatenate((y_train, y_val))

        if model_name == "mlp":
            base_model = MLPRegressor(**build_mlp_params(best_params))
            x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_scaler, y_scaler = fit_transform_scalers(
                x_all, y_all, x_val, y_val
            )
            model = fit_mlp_with_val(
                base_model,
                x_train_scaled,
                y_train_scaled,
                x_val_scaled,
                y_val_scaled,
                patience=20,
            )
            model._best_params = best_params
            final_models.append(model)
            scaler_list.append((x_scaler, y_scaler))

        elif model_name == "svm":
            model = SVR(**best_params.copy())
            x_train_scaled, y_train_scaled, _, _, x_scaler, y_scaler = fit_transform_scalers(
                x_all, y_all, x_val, y_val
            )
            model.fit(x_train_scaled, y_train_scaled)
            model._best_params = best_params
            final_models.append(model)
            scaler_list.append((x_scaler, y_scaler))

        elif model_name == "rf":
            model = RandomForestRegressor(**best_params.copy())
            model.fit(x_all, y_all)
            model._best_params = best_params
            final_models.append(model)
            scaler_list.append((None, None))

        elif model_name == "xgboost":
            model = XGBRegressor(**best_params.copy())
            model.fit(x_all, y_all, eval_set=[(x_val, y_val)], verbose=False)
            model._best_params = best_params
            final_models.append(model)
            scaler_list.append((None, None))

    return best_params, final_models, scaler_list




if __name__ == "__main__":
    set_seed(config.random_seed)

    for prop in config.property_columns:
        split_settings = [
            {
                "split_name": f"kmeans_{prop}_no_outliers_035",
                "data_dir": config.project_root / "data_folds" / f"kmeans_{prop}_no_outliers_035",
                "save_root": config.classical_output_root.parent
                / config.classical_output_root.name.replace("random", "kmeans")
                / prop,
            },
            {
                "split_name": f"random_{prop}_no_outliers_035",
                "data_dir": config.project_root / "data_folds" / f"random_{prop}_no_outliers_035",
                "save_root": config.classical_output_root / prop,
            },
        ]

        for split_setting in split_settings:
            split_name = split_setting["split_name"]
            data_dir = split_setting["data_dir"]
            save_root = split_setting["save_root"]
            os.makedirs(save_root, exist_ok=True)

            all_results = []

            print(f"\n===== Property: {prop} | Split: {split_name} =====")

            for model_name in config.classical_model_names:
                model_output_dir = save_root / model_name
                os.makedirs(model_output_dir, exist_ok=True)

                folds_data = []
                for fold in range(config.fold_num):
                    df_valid, features, targets, masks = build_fold_arrays(data_dir, fold, prop)
                    train_mask, val_mask, test_mask = masks["train"], masks["val"], masks["test"]
                    folds_data.append(
                        {
                            "fold": fold,
                            "df_valid": df_valid,
                            "masks": masks,
                            "x_train": features[train_mask],
                            "y_train": targets[train_mask],
                            "x_val": features[val_mask],
                            "y_val": targets[val_mask],
                            "x_test": features[test_mask],
                            "y_test": targets[test_mask],
                        }
                    )
                best_params, final_models, scaler_list = multi_fold_optimize_optuna(
                    model_name,
                    folds_data,
                    n_trials=config.classical_n_trials,
                    seed=config.random_seed,
                    output_dir=model_output_dir,
                )
                fold_metrics = []
                for fold_data, model, scalers in zip(folds_data, final_models, scaler_list):
                    fold = fold_data["fold"]
                    df_valid = fold_data["df_valid"]
                    masks = fold_data["masks"]
                    x_test = fold_data["x_test"]
                    y_test = fold_data["y_test"]

                    if model_name in {"mlp", "svm"}:
                        x_scaler, y_scaler = scalers
                        x_test_scaled = x_scaler.transform(x_test)
                        y_test_pred_scaled = model.predict(x_test_scaled)
                        y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
                    else:
                        y_test_pred = model.predict(x_test)

                    r2 = r2_score(y_test, y_test_pred)
                    mse = mean_squared_error(y_test, y_test_pred)
                    result = {
                        'model': model_name,
                        'fold': fold,
                        'best_params': getattr(model, "_best_params", None),
                        'r2': r2,
                        'mse': mse,
                    }

                    fold_dir = model_output_dir / f"fold_{fold}"
                    os.makedirs(fold_dir, exist_ok=True)

                    x_scaler, y_scaler = scalers
                    model_bundle = {
                        "model": model,
                        "feature_scaler": x_scaler,
                        "target_scaler": y_scaler,
                        "target_col": prop,
                        "model_name": model_name,
                        "best_params": best_params,
                    }
                    joblib.dump(model_bundle, fold_dir / "model.pkl")

                    metrics = compute_single_target_array_metrics(y_test, y_test_pred)
                    metrics = {
                        "fold": fold,
                        "r2": metrics["r2"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                    }

                    print(f"Prop: {prop}, Model: {model_name}, Fold: {fold}, R2: {r2:.4f}, MSE={mse:.4f}")
                    all_results.append(result)
                    fold_metrics.append(metrics)

                    test_pred_df = pd.DataFrame({
                        "SMILES": df_valid.loc[masks["test"], "SMILES"].values,
                        f"{prop}_pred": y_test_pred,
                    })
                    test_pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)

                cv_metrics_path = save_cv_metrics(model_output_dir, fold_metrics, ("r2", "mae", "rmse"))
                print(f"[Final][{model_name}] Cross-validation metrics saved to: {cv_metrics_path}")

            pd.DataFrame(all_results).to_csv(save_root / "all_models_summary_metrics.csv", index=False)
