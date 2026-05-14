import pandas as pd

from plots_utils import (
    aggregate_r2,
    apply_paper_plot_style,
    make_output_dir,
    plot_grouped_r2_panels,
    plots_root,
)


def load_metrics_for_split(split_name: str) -> dict:
    mpnn_metrics_dir = plots_root / "mpnn_metrics"
    transformer_metrics_dir = plots_root / "transformer_metrics"

    mpnn_stl = pd.read_csv(mpnn_metrics_dir / f"mpnn_{split_name}_indiv_prop_metrics.csv")
    mpnn_mtl = pd.read_csv(mpnn_metrics_dir / "mpnn_mtl_no_finetune_metrics.csv")
    mpnn_ft = pd.read_csv(mpnn_metrics_dir / "mpnn_mtl_finetune_from_pretrain_metrics.csv")

    transformer_stl = pd.read_csv(transformer_metrics_dir / f"transformer_{split_name}_indiv_prop_metrics.csv")
    transformer_mtl = pd.read_csv(transformer_metrics_dir / "transformer_mtl_no_finetune_metrics.csv")
    transformer_ft = pd.read_csv(transformer_metrics_dir / "transformer_mtl_finetune_from_pretrain_metrics.csv")

    return {
        "MPNN STL": aggregate_r2(mpnn_stl, target_col="property", r2_col="R2"),
        "MPNN MTL": aggregate_r2(mpnn_mtl, target_col="target", r2_col="r2", split_type=split_name),
        "MPNN pretraining MTL": aggregate_r2(mpnn_ft, target_col="target", r2_col="r2", split_type=split_name),
        "Transformer STL": aggregate_r2(transformer_stl, target_col="property", r2_col="R2"),
        "Transformer MTL": aggregate_r2(transformer_mtl, target_col="target", r2_col="r2", split_type=split_name),
        "Transformer pretraining MTL": aggregate_r2(transformer_ft, target_col="target", r2_col="r2", split_type=split_name),
    }


def main() -> None:
    apply_paper_plot_style()
    methods = [
        "MPNN STL",
        "MPNN MTL",
        "MPNN pretraining MTL",
        "Transformer STL",
        "Transformer MTL",
        "Transformer pretraining MTL",
    ]
    style = {
        "figsize": (16, 5.2),
        "bar_width": 0.12,
        "ylabel_fontsize": 20,
        "legend_fontsize": 20,
        "ytick_fontsize": 20,
        "xtick_fontsize": 20,
        "capsize": 3,
        "elinewidth": 1.0,
        "ylim": (-0.05, 1.1),
        "grid_alpha": 0.3,
        "grid_linestyle": "--",
        "bar_colors": ["#206FB6", "#6BADD7", "#C5DAEE", "#EE3B2A", "#FC9171", "#FDDFD0"],
        "legend_ncol": 3,
        "legend_bbox": (0.5, 1.2),
    }
    out_dir = make_output_dir("mpnn_transformer_r2_combined")
    plot_grouped_r2_panels(
        load_metrics_for_split("random"),
        load_metrics_for_split("kmeans"),
        methods,
        out_dir / "mpnn_transformer_r2_combined.png",
        style=style,
    )


if __name__ == "__main__":
    main()
