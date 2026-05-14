import argparse

from sklearn.model_selection import KFold

from split_utils import *


# make splits (train/val/test) per fold
def random_split(df, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: Dict[int, Dict[str, np.ndarray]] = {}
    for fold_id, (train_index, test_index) in enumerate(kf.split(df)):
        # choose 1/(n_splits-1) of train as val
        rng = np.random.default_rng(random_state + fold_id)
        val_size = int(round(len(train_index) / (n_splits - 1)))
        val_index = rng.choice(train_index, size=val_size, replace=False)
        train_only = np.setdiff1d(train_index, val_index, assume_unique=False)
        splits[fold_id] = {"train": train_only, "val": val_index, "test": np.asarray(test_index)}
    return splits


def save_multitask_random_split(
        input_file: Path = default_training_data_file(),
        output_dir: Path = default_fold_output_dir("random"),
        n_splits: int = 5,
        random_state: int = 42,
) -> pd.DataFrame:
    """Generate and save the multitask random split."""
    df = pd.read_csv(input_file)
    splits = random_split(df, n_splits=n_splits, random_state=random_state)
    row_idx = np.arange(len(df))
    return attach_and_save_splits(df, row_idx, splits, output_dir)


def save_property_random_splits(
        input_file: Path = default_training_data_file(),
        n_splits: int = 5,
        random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate and save per-property random splits after dropping missing target values."""
    df = pd.read_csv(input_file)
    property_dfs = make_property_dataframes(df)
    output_dirs = make_property_output_dirs(repo_root(), "random")
    summaries: Dict[str, pd.DataFrame] = {}
    for prop, prop_df in property_dfs.items():
        splits = random_split(prop_df, n_splits=n_splits, random_state=random_state)
        summaries[prop] = attach_and_save_splits(prop_df, prop_df.index.to_numpy(), splits, output_dirs[prop])
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random five-fold data splits.")
    parser.add_argument("--mode", choices=("multitask", "properties", "both"), default="multitask")
    parser.add_argument("--input-file", type=Path, default=default_training_data_file())
    parser.add_argument("--output-dir", type=Path, default=default_fold_output_dir("random"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Also plot PCA/UMAP/t-SNE fold visualizations for multitask data.")
    args = parser.parse_args()

    if args.mode in ("multitask", "both"):
        summary_df = save_multitask_random_split(args.input_file, args.output_dir, args.n_splits, args.seed)
        print(f"Splits saved to: {args.output_dir}")
        print(summary_df)

        if args.plot:
            df = pd.read_csv(args.input_file)
            splits = random_split(df, n_splits=args.n_splits, random_state=args.seed)
            X, _ = featurize_smiles_column(df, smiles_col="SMILES", radius=2, n_bits=1024)
            plot_embeddings_one_figure_per_mode(
                X=X,
                splits=splits,
                out_dir=args.output_dir,
                method_name="Random",
                cfg=EmbeddingCfg(random_state=args.seed),
                centers=None,
                modes=("pca", "umap", "tsne"),
            )

    if args.mode in ("properties", "both"):
        summaries = save_property_random_splits(args.input_file, args.n_splits, args.seed)
        for prop, summary_df in summaries.items():
            print(f"{prop} splits saved")
            print(summary_df)
