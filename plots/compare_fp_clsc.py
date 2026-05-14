import numpy as np
import pandas as pd

from plots_utils import (
    apply_paper_plot_style,
    classical_mtl_root,
    classical_stl_root,
    make_output_dir,
    plot_grouped_r2_panels,
    properties,
)


def load_stl_stats(stl_root, props, model_names):
    means = {prop: {} for prop in props}
    stds = {prop: {} for prop in props}
    for prop in props:
        df = pd.read_csv(stl_root / prop / "all_models_summary_metrics.csv")
        grouped = df.groupby("model")["r2"]
        for model_name in model_names:
            values = grouped.get_group(model_name) if model_name in grouped.groups else pd.Series(dtype=float)
            means[prop][model_name] = values.mean() if not values.empty else np.nan
            stds[prop][model_name] = values.std(ddof=1) if not values.empty else np.nan
    return means, stds


def load_mtl_stats(mtl_root, props):
    df = pd.read_csv(mtl_root / "all_folds_metrics.csv")
    return (
        {prop: df[f"{prop}_r2"].mean() for prop in props},
        {prop: df[f"{prop}_r2"].std(ddof=1) for prop in props},
    )


def save_split_summary(split_name, stl_mean, stl_std, mtl_mean, mtl_std, save_dir, props, model_names):
    rows = []
    for prop in props:
        for model_name in model_names:
            rows.append({"property": prop, "method": model_name, "mean_r2": stl_mean[prop][model_name], "std_r2": stl_std[prop][model_name]})
        rows.append({"property": prop, "method": "mtl", "mean_r2": mtl_mean[prop], "std_r2": mtl_std[prop]})
    save_path = save_dir / f"clscl_{split_name}_summary.csv"
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


def main() -> None:
    apply_paper_plot_style()
    props = properties
    model_names = ("rf", "xgboost", "svm", "mlp")
    method_labels = ["RF", "XGBoost", "SVR", "STL-MLP", "MTL-MLP"]
    style = {
        "figsize": (15.5, 5.2),
        "bar_width": 0.15,
        "ylabel_fontsize": 20,
        "xtick_fontsize": 20,
        "ytick_fontsize": 20,
        "legend_fontsize": 19,
        "capsize": 3,
        "elinewidth": 1.0,
        "bar_colors": ["#206FB6", "#6BADD7", "#C5DAEE", "#FC9171", "#EE3B2A"],
        "grid_alpha": 0.3,
        "grid_linestyle": "--",
        "ylim": (-1.25, 0.9),
        "legend_ncol": 5,
        "legend_bbox": (0.5, 1.1),
    }

    metrics_by_split = {}
    csv_dir = make_output_dir("clscl_metrics")
    fig_dir = make_output_dir("clscl_vs_mtl_r2_combined_horizontal")

    for split_name in ("random", "kmeans"):
        stl_mean, stl_std = load_stl_stats(classical_stl_root(split_name), props, model_names)
        mtl_mean, mtl_std = load_mtl_stats(classical_mtl_root(split_name), props)
        save_split_summary(split_name, stl_mean, stl_std, mtl_mean, mtl_std, csv_dir, props, model_names)
        metrics_by_split[split_name] = {
            label: {prop: (stl_mean[prop][model], stl_std[prop][model]) for prop in props}
            for label, model in zip(method_labels[:-1], model_names)
        }
        metrics_by_split[split_name]["MTL-MLP"] = {
            prop: (mtl_mean[prop], mtl_std[prop]) for prop in props
        }

    plot_grouped_r2_panels(
        metrics_by_split["random"],
        metrics_by_split["kmeans"],
        method_labels,
        fig_dir / "clscl_vs_mtl_r2_combined_horizontal.png",
        style=style,
    )


if __name__ == "__main__":
    main()
