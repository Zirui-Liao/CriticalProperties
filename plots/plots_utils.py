from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
plots_root = project_root / "plots"
experimental_data_file = project_root / "training_data" / "exp_data_no_outliers_035.csv"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


properties = ["Tc", "Pc", "Vc", "omega"]
property_positions = {"Tc": (0, 0), "Pc": (0, 1), "Vc": (1, 0), "omega": (1, 1)}
display_labels = {"Tc": "Tc", "Pc": "Pc", "Vc": "Vc", "omega": r"$\omega$"}
paper_plot_style = {
    "font.family": "Times New Roman",
    "font.sans-serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
}


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    n_samples = len(y_true_arr)
    if n_samples == 0:
        return {"n_samples": 0, "r2": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan}

    diff = y_true_arr - y_pred_arr
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    total = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = np.nan if n_samples < 2 or total == 0.0 else 1.0 - float(np.sum(diff ** 2)) / total
    return {"n_samples": n_samples, "r2": float(r2), "mse": mse, "rmse": float(mse ** 0.5), "mae": mae}


def apply_paper_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(paper_plot_style)


def make_output_dir(name: str) -> Path:
    out_dir = plots_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig, save_path: Path, *, dpi: int = 300, save_svg: bool = True) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if save_svg:
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")


def split_file(split_name: str, *, prop: str | None = None, fold: int = 0) -> Path:
    if prop is None:
        folder = f"{split_name}_no_outliers_035"
    else:
        folder = f"{split_name}_{prop}_no_outliers_035"
    return project_root / "data_folds" / folder / f"fold_{fold}.csv"


def read_table(path: Path, required_cols: Iterable[str] = ()) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    return df


def read_experimental(path: Path, props: Sequence[str]) -> pd.DataFrame:
    df = read_table(path, ["SMILES"])
    if "split" in df.columns:
        df = df.drop(columns=["split"])
    return df.rename(columns={prop: f"{prop}_exp" for prop in props if prop in df.columns})


def mtl_experiments(model_family: str, *, finetuned: bool) -> list[dict]:
    roots = {
        "mpnn": project_root / "mpnn_models",
        "transformer": project_root / "transformer_models",
    }
    root = roots[model_family]
    if model_family == "mpnn" and not finetuned:
        names = {
            "kmeans": "kmeans_no_outliers_035_optuna",
            "random": "random_no_outliers_035_optuna_trainopt",
        }
    elif model_family == "mpnn":
        names = {
            "kmeans": "kmeans_no_outliers_035_optuna_ft_from_pretrain",
            "random": "random_no_outliers_035_optuna_ft_from_pretrain",
        }
    elif model_family == "transformer" and not finetuned:
        names = {
            "kmeans": "kmeans_no_outliers_035_optuna",
            "random": "random_no_outliers_035_optuna_trainopt",
        }
    elif model_family == "transformer":
        names = {
            "kmeans": "kmeans_no_outliers_035_mtl_ft_from_pretrain",
            "random": "random_no_outliers_035_mtl_ft_from_pretrain",
        }
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return [
        {
            "experiment": folder,
            "split_type": split,
            "split_name": f"{split}_no_outliers_035",
            "root": root / folder,
            "finetune_from_pretrain": finetuned,
        }
        for split, folder in names.items()
    ]


def stl_prediction_root(model_family: str, split_name: str, prop: str) -> Path:
    model_roots = {
        "mpnn": project_root / "mpnn_models",
        "transformer": project_root / "transformer_models",
    }
    return model_roots[model_family] / f"{split_name}_{prop}_no_outliers_035_optuna"


def classical_stl_root(split_name: str) -> Path:
    return project_root / "clscl_models" / f"outputs_clscl_ml_optuna_{split_name}"


def classical_mtl_root(split_name: str) -> Path:
    return project_root / "clscl_models" / f"outputs_multitask_mlp_optuna_{split_name}"


def fold_prediction_path(pred_root: Path, fold: int) -> Path:
    return pred_root / f"fold_{fold}" / "test_predictions.csv"


def prediction_column(df: pd.DataFrame, prop: str) -> str:
    for col in (prop, f"{prop}_pred", "prediction", "pred", "y_pred"):
        if col in df.columns:
            return col
    raise KeyError(f"Prediction column for {prop} not found. Columns: {df.columns.tolist()}")


def true_column(df: pd.DataFrame, exp_df: pd.DataFrame, prop: str) -> tuple[pd.DataFrame, str]:
    col = f"{prop}_target"
    if col in df.columns:
        return df, col
    if prop not in exp_df.columns:
        raise KeyError(f"Experimental column for {prop} not found.")
    true_col = f"{prop}_true"
    true_df = exp_df[["SMILES", prop]].rename(columns={prop: true_col})
    merged = true_df.merge(df, on="SMILES", how="inner")
    return merged, true_col


def metric_rows_from_predictions(
    exp_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    targets: Sequence[str],
) -> list[dict]:
    rows = []
    for target in targets:
        try:
            pred_col = prediction_column(pred_df, target)
            data, y_col = true_column(pred_df, exp_df, target)
        except KeyError as exc:
            print(f"[WARN] {exc}")
            continue

        stat_df = data[["SMILES", y_col, pred_col]].dropna()
        if stat_df.empty:
            values = {"r2": np.nan, "mse": np.nan, "mae": np.nan, "rmse": np.nan}
        else:
            values = regression_metrics(stat_df[y_col].to_numpy(), stat_df[pred_col].to_numpy())

        rows.append(
            {
                "target": target,
                "n_samples": len(stat_df),
                "r2": values["r2"],
                "mse": values["mse"],
                "mae": values["mae"],
                "rmse": values["rmse"],
            }
        )
    return rows


def collect_mtl_metrics(
    experiments: Sequence[dict],
    targets: Sequence[str],
    *,
    n_folds: int,
) -> pd.DataFrame:
    exp_df = read_table(experimental_data_file, ["SMILES", *targets])
    rows = []
    for experiment in experiments:
        for fold in range(n_folds):
            pred_path = fold_prediction_path(experiment["root"], fold)
            if not pred_path.exists():
                print(f"[WARN] Missing predictions: {pred_path}")
                continue
            pred_df = read_table(pred_path, ["SMILES"])
            for metrics in metric_rows_from_predictions(exp_df, pred_df, targets):
                rows.append(
                    {
                        "experiment": experiment["experiment"],
                        "split_type": experiment["split_type"],
                        "fold": fold,
                        "finetune_from_pretrain": experiment["finetune_from_pretrain"],
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def collect_stl_metrics(
    model_family: str,
    split_name: str,
    targets: Sequence[str],
    *,
    n_folds: int,
) -> pd.DataFrame:
    rows = []
    for prop in targets:
        exp_df = read_table(split_file(split_name, prop=prop), ["SMILES", prop])
        pred_root = stl_prediction_root(model_family, split_name, prop)
        for fold in range(n_folds):
            pred_path = fold_prediction_path(pred_root, fold)
            if not pred_path.exists():
                print(f"[WARN] Missing predictions: {pred_path}")
                continue
            pred_df = read_table(pred_path, ["SMILES"])
            for metrics in metric_rows_from_predictions(exp_df, pred_df, [prop]):
                rows.append({"property": prop, "fold": fold, "R2": metrics["r2"], "MSE": metrics["mse"], "MAE": metrics["mae"]})
    columns = ["property", "fold", "R2", "MSE", "MAE"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["property", "fold"]).reset_index(drop=True)


def load_all_folds_predictions(
    pred_root: Path,
    props: Sequence[str],
    *,
    n_folds: int,
) -> pd.DataFrame:
    frames = []
    for fold in range(n_folds):
        path = fold_prediction_path(pred_root, fold)
        if not path.exists():
            print(f"[WARN] Missing predictions: {path}")
            continue
        df = read_table(path, ["SMILES"]).copy()
        rename_map = {f"{prop}_pred": prop for prop in props if f"{prop}_pred" in df.columns and prop not in df.columns}
        df = df.rename(columns=rename_map)
        df["replicate"] = fold
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No fold predictions under {pred_root}")
    df_all = pd.concat(frames, ignore_index=True)
    keep_cols = ["SMILES", "replicate"] + [prop for prop in props if prop in df_all.columns]
    return df_all[keep_cols]


def parity_metrics(df_all: pd.DataFrame, exp_df: pd.DataFrame, props: Sequence[str]) -> dict:
    available = [prop for prop in props if prop in df_all.columns]
    pred_avg = df_all[["SMILES", *available]].groupby("SMILES", as_index=False).mean()
    merged = pred_avg.merge(exp_df, on="SMILES", how="inner")
    out = {}
    for prop in props:
        exp_col = f"{prop}_exp"
        if prop not in merged.columns or exp_col not in merged.columns:
            continue
        stat_df = merged[[prop, exp_col]].dropna()
        if stat_df.empty:
            continue
        values = regression_metrics(stat_df[exp_col], stat_df[prop])
        out[prop] = {
            "R2": f"{values['r2']:.4f}",
            "RMSE": f"{values['rmse']:.4f}",
            "MAE": f"{values['mae']:.2f}",
        }
    return out


def plot_parity_grid(
    df_all: pd.DataFrame,
    exp_df: pd.DataFrame,
    props: Sequence[str],
    save_path: Path,
    *,
    style: dict | None = None,
) -> None:
    import matplotlib.pyplot as plt

    style = style or {}
    metrics = parity_metrics(df_all, exp_df, props)
    fig, axes = plt.subplots(2, 2, figsize=style.get("figsize", (12, 12)))
    cmap = plt.get_cmap("tab10")

    for prop in props:
        row, col = property_positions[prop]
        ax = axes[row, col]
        exp_col = f"{prop}_exp"
        if prop not in df_all.columns or exp_col not in exp_df.columns:
            ax.axis("off")
            continue

        for fold in sorted(df_all["replicate"].unique()):
            df_fold = df_all[df_all["replicate"] == fold]
            df_plot = df_fold[["SMILES", prop]].merge(exp_df[["SMILES", exp_col]], on="SMILES", how="inner").dropna()
            ax.scatter(
                df_plot[exp_col],
                df_plot[prop],
                label=f"Fold {fold}",
                alpha=style.get("scatter_alpha", 0.6),
                s=style.get("scatter_size", 20),
                color=cmap(int(fold)),
            )

        pred_avg = df_all[["SMILES", prop]].groupby("SMILES", as_index=False).mean()
        merged = pred_avg.merge(exp_df[["SMILES", exp_col]], on="SMILES", how="inner").dropna()
        if not merged.empty:
            mn = min(merged[exp_col].min(), merged[prop].min())
            mx = max(merged[exp_col].max(), merged[prop].max())
            ax.plot([mn, mx], [mn, mx], ls="--", lw=style.get("diag_line_width", 1.0), c="gray")

        ax.set_ylabel(style.get("ylabel_map", {}).get(prop, f"Predicted {display_labels[prop]}"), fontsize=style.get("axis_label_fontsize", None))
        ax.set_xlabel("Experimental data" if row == 1 else "", fontsize=style.get("axis_label_fontsize", None))
        ax.tick_params(axis="both", labelsize=style.get("tick_fontsize", None))

        if prop in metrics:
            text = "\n".join(
                [
                    f"R2 = {metrics[prop]['R2']}",
                    f"RMSE = {metrics[prop]['RMSE']}",
                    f"MAE = {metrics[prop]['MAE']}",
                ]
            )
            ax.text(
                0.95,
                0.05,
                text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=style.get("metrics_fontsize", None),
            )
        ax.legend(fontsize=style.get("legend_fontsize", "small"), loc="upper left", ncol=style.get("legend_ncol", 1), frameon=False)
        ax.set_box_aspect(1)

    fig.tight_layout()
    save_figure(fig, save_path, dpi=style.get("dpi", 300))
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_stl_parity_grid(
    model_family: str,
    split_name: str,
    props: Sequence[str],
    save_path: Path,
    *,
    n_folds: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    cmap = plt.get_cmap("tab10")
    for prop in props:
        ax = axes[property_positions[prop]]
        exp_df = read_experimental(split_file(split_name, prop=prop), [prop])
        df_all = load_all_folds_predictions(stl_prediction_root(model_family, split_name, prop), [prop], n_folds=n_folds)
        exp_col = f"{prop}_exp"
        all_true = []
        all_pred = []

        for fold in sorted(df_all["replicate"].unique()):
            df_fold = df_all[df_all["replicate"] == fold]
            df_plot = df_fold[["SMILES", prop]].merge(exp_df[["SMILES", exp_col]], on="SMILES", how="inner").dropna()
            ax.scatter(df_plot[exp_col], df_plot[prop], color=cmap(int(fold)), label=f"Fold {fold}", alpha=0.7, s=18)
            all_true.extend(df_plot[exp_col].tolist())
            all_pred.extend(df_plot[prop].tolist())

        values = regression_metrics(all_true, all_pred)
        mn = min(min(all_true), min(all_pred))
        mx = max(max(all_true), max(all_pred))
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        ax.set_xlabel("Experimental Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"{prop}: Experimental vs. Predicted")
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.95,
            0.05,
            f"R2 = {values['r2']:.3f}\nMSE = {values['mse']:.3f}\nMAE = {values['mae']:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"),
        )
        if prop == props[0]:
            ax.legend(fontsize="small", loc="upper left", ncol=2)

    fig.tight_layout()
    save_figure(fig, save_path, dpi=300, save_svg=False)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_classical_stl_parity(
    model_name: str,
    split_name: str,
    props: Sequence[str],
    save_path: Path,
    *,
    n_folds: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    cmap = plt.get_cmap("tab10")
    for prop in props:
        ax = axes[property_positions[prop]]
        exp_df = read_experimental(split_file(split_name, prop=prop), [prop])
        pred_root = classical_stl_root(split_name) / prop / model_name
        df_all = load_all_folds_predictions(pred_root, [prop], n_folds=n_folds)
        exp_col = f"{prop}_exp"
        all_true = []
        all_pred = []
        for fold in sorted(df_all["replicate"].unique()):
            df_fold = df_all[df_all["replicate"] == fold]
            df_plot = df_fold[["SMILES", prop]].merge(exp_df[["SMILES", exp_col]], on="SMILES", how="inner").dropna()
            ax.scatter(df_plot[exp_col], df_plot[prop], color=cmap(int(fold)), label=f"Fold {fold}", alpha=0.6, s=20)
            all_true.extend(df_plot[exp_col].tolist())
            all_pred.extend(df_plot[prop].tolist())

        values = regression_metrics(all_true, all_pred)
        mn = min(min(all_true), min(all_pred))
        mx = max(max(all_true), max(all_pred))
        ax.plot([mn, mx], [mn, mx], ls="--", lw=1, c="gray")
        ax.set_title(f"{prop} ({model_name})")
        ax.set_xlabel("Experimental")
        ax.set_ylabel("Predicted")
        ax.text(
            0.95,
            0.05,
            f"R2 = {values['r2']:.4f}\nMSE = {values['mse']:.4f}\nMAE = {values['mae']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.5),
        )
        ax.legend(fontsize="small", loc="upper left", ncol=2)

    fig.tight_layout()
    save_figure(fig, save_path, dpi=300, save_svg=False)
    plt.close(fig)
    print(f"Saved: {save_path}")


def aggregate_r2(df: pd.DataFrame, *, target_col: str, r2_col: str, split_type: str | None = None) -> dict:
    data = df.copy()
    if split_type is not None and "split_type" in data.columns:
        data = data[data["split_type"].astype(str).str.lower() == split_type.lower()]
    return {
        prop: (
            data.loc[data[target_col] == prop, r2_col].mean(),
            data.loc[data[target_col] == prop, r2_col].std(ddof=1),
        )
        for prop in properties
    }


def plot_grouped_r2_panels(
    random_metrics: dict,
    kmeans_metrics: dict,
    methods: Sequence[str],
    save_path: Path,
    *,
    style: dict,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2,
        figsize=style["figsize"],
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    for ax, metrics in zip(axes, [random_metrics, kmeans_metrics]):
        x = np.arange(len(properties))
        width = style["bar_width"]
        offsets = np.linspace(-(len(methods) - 1) / 2 * width, (len(methods) - 1) / 2 * width, len(methods))
        for idx, method in enumerate(methods):
            means = [metrics[method][prop][0] for prop in properties]
            stds = [metrics[method][prop][1] for prop in properties]
            ax.bar(
                x + offsets[idx],
                means,
                width=width,
                yerr=stds,
                capsize=style["capsize"],
                error_kw={"elinewidth": style["elinewidth"]},
                color=style["bar_colors"][idx],
                label=method,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([display_labels[prop] for prop in properties], fontsize=style["xtick_fontsize"])
        ax.tick_params(axis="y", labelsize=style["ytick_fontsize"])
        ax.grid(axis="y", linestyle=style["grid_linestyle"], alpha=style["grid_alpha"])
        ax.set_ylim(*style["ylim"])

    axes[0].set_ylabel("Mean R2 (5-fold)", fontsize=style["ylabel_fontsize"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=style["legend_ncol"],
        frameon=False,
        fontsize=style["legend_fontsize"],
        bbox_to_anchor=style["legend_bbox"],
    )
    save_figure(fig, save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")
