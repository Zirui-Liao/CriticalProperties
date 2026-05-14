import argparse

from plots_utils import (
    classical_mtl_root,
    load_all_folds_predictions,
    make_output_dir,
    plot_classical_stl_parity,
    plot_parity_grid,
    properties,
    read_experimental,
    split_file,
)


def write_mtl_parity(props, n_folds):
    out_dir = make_output_dir("clscl_parity_plots")
    for split_name in ("kmeans", "random"):
        df_all = load_all_folds_predictions(classical_mtl_root(split_name), props, n_folds=n_folds)
        exp_df = read_experimental(split_file(split_name), props)
        save_path = out_dir / f"parity_mlp_{split_name}_mtl.png"
        plot_parity_grid(df_all, exp_df, props, save_path)


def write_stl_parity(props, n_folds):
    models = ["mlp", "rf", "svm", "xgboost"]
    out_dir = make_output_dir("clscl_parity_plots")
    for split_name in ("kmeans", "random"):
        for model_name in models:
            save_path = out_dir / f"parity_{model_name}_{split_name}_stl.png"
            plot_classical_stl_parity(model_name, split_name, props, save_path, n_folds=n_folds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate classical/FP parity plots.")
    parser.add_argument("--task", choices=("all", "mtl_parity", "stl_parity"), default="all")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    props = properties
    if args.task in {"all", "mtl_parity"}:
        write_mtl_parity(props, args.n_folds)
    if args.task in {"all", "stl_parity"}:
        write_stl_parity(props, args.n_folds)


if __name__ == "__main__":
    main()
