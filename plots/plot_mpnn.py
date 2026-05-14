import argparse

from plots_utils import (
    collect_mtl_metrics,
    collect_stl_metrics,
    load_all_folds_predictions,
    make_output_dir,
    mtl_experiments,
    plot_parity_grid,
    plot_stl_parity_grid,
    properties,
    read_experimental,
    split_file,
)


def write_mtl_metrics(props, n_folds):
    out_dir = make_output_dir("mpnn_metrics")
    for finetuned, filename in (
        (False, "mpnn_mtl_no_finetune_metrics.csv"),
        (True, "mpnn_mtl_finetune_from_pretrain_metrics.csv"),
    ):
        metrics_df = collect_mtl_metrics(
            mtl_experiments("mpnn", finetuned=finetuned),
            props,
            n_folds=n_folds,
        )
        save_path = out_dir / filename
        metrics_df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")


def write_stl_metrics(props, n_folds):
    out_dir = make_output_dir("mpnn_metrics")
    for split_name in ("kmeans", "random"):
        metrics_df = collect_stl_metrics("mpnn", split_name, props, n_folds=n_folds)
        save_path = out_dir / f"mpnn_{split_name}_indiv_prop_metrics.csv"
        metrics_df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")


def write_mtl_parity(props, n_folds):
    out_dir = make_output_dir("mpnn_parity_plots")
    style = {
        "figsize": (12, 12),
        "dpi": 300,
        "scatter_size": 20,
        "scatter_alpha": 0.6,
        "diag_line_width": 1.0,
        "tick_fontsize": 20,
        "axis_label_fontsize": 20,
        "legend_fontsize": 16,
        "metrics_fontsize": 20,
        "ylabel_map": {
            "Tc": "Predicted $T_c$ / K",
            "Pc": "Predicted $P_c$ / bar",
            "Vc": "Predicted $V_c$ / mL/mol",
            "omega": r"Predicted $\omega$",
        },
    }
    for experiment in [
        *mtl_experiments("mpnn", finetuned=False),
        *mtl_experiments("mpnn", finetuned=True),
    ]:
        df_all = load_all_folds_predictions(experiment["root"], props, n_folds=n_folds)
        exp_df = read_experimental(split_file(experiment["split_type"]), props)
        finetune_tag = "finetune" if experiment["finetune_from_pretrain"] else "nofinetune"
        save_path = out_dir / f"parity_mpnn_{experiment['split_type']}_{finetune_tag}.png"
        plot_parity_grid(df_all, exp_df, props, save_path, style=style)


def write_stl_parity(props, n_folds):
    out_dir = make_output_dir("mpnn_parity_plots")
    for split_name in ("kmeans", "random"):
        save_path = out_dir / f"parity_mpnn_{split_name}_stl.png"
        plot_stl_parity_grid("mpnn", split_name, props, save_path, n_folds=n_folds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MPNN metrics and parity plots.")
    parser.add_argument(
        "--task",
        choices=("all", "metrics", "parity", "mtl_metrics", "stl_metrics", "mtl_parity", "stl_parity"),
        default="all",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    props = properties
    if args.task in {"all", "metrics", "mtl_metrics"}:
        write_mtl_metrics(props, args.n_folds)
    if args.task in {"all", "metrics", "stl_metrics"}:
        write_stl_metrics(props, args.n_folds)
    if args.task in {"all", "parity", "mtl_parity"}:
        write_mtl_parity(props, args.n_folds)
    if args.task in {"all", "parity", "stl_parity"}:
        write_stl_parity(props, args.n_folds)


if __name__ == "__main__":
    main()
