from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna


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


def save_params(params: Dict[str, Any], path: Path) -> Path:
    """Save a hyperparameter dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(params), f, indent=2)
    return path


def save_trials(study: optuna.Study, path: Path) -> Path:
    """Save the trial summary format consumed by existing result scripts."""
    trials_data = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "fold_rmse_list": trial.user_attrs.get("fold_rmse_list", None),
            "state": str(trial.state),
        }
        for trial in study.trials
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=2)
    return path


def save_study_results(
    study: optuna.Study,
    hpo_root: Path,
    params: Dict[str, Any],
    params_label: str = "Best params",
) -> None:
    """Save Optuna best parameters and trial summaries in the standard layout."""
    hpo_root = Path(hpo_root)
    best_params_path = save_params(params, hpo_root / "best_hyperparams.json")
    trials_summary_path = save_trials(study, hpo_root / "trials_summary.json")
    print(f"[HPO] {params_label} saved to: {best_params_path}")
    print(f"[HPO] All trials summary saved to: {trials_summary_path}")
