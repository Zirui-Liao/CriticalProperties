import argparse

from sklearn.cluster import KMeans

from split_utils import *


def kmeans_split(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    # Convert SMILES to Morgan fingerprint features
    X, row_idx = featurize_smiles_column(df, smiles_col="SMILES", radius=2, n_bits=1024)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # Build CV splits based on cluster labels
    splits = make_cv_splits(labels, n_splits=n_splits, random_state=random_state)

    return splits, centers, row_idx


def save_multitask_kmeans_split(
        input_file: Path = default_training_data_file(),
        output_dir: Path = default_fold_output_dir("kmeans"),
        n_splits: int = 5,
        random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate and save the multitask K-means split."""
    df = pd.read_csv(input_file)
    splits, centers, row_idx = kmeans_split(df, n_splits=n_splits, random_state=random_state)
    summary_df = attach_and_save_splits(df, row_idx, splits, output_dir)
    return summary_df, centers


def save_property_kmeans_splits(
        input_file: Path = default_training_data_file(),
        n_splits: int = 5,
        random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate and save per-property K-means splits after dropping missing target values."""
    df = pd.read_csv(input_file)
    property_dfs = make_property_dataframes(df)
    output_dirs = make_property_output_dirs(repo_root(), "kmeans")
    summaries: Dict[str, pd.DataFrame] = {}
    for prop, prop_df in property_dfs.items():
        splits, _, row_idx = kmeans_split(prop_df, n_splits=n_splits, random_state=random_state)
        summaries[prop] = attach_and_save_splits(prop_df, row_idx, splits, output_dirs[prop])
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-means five-fold data splits.")
    parser.add_argument("--mode", choices=("multitask", "properties", "both"), default="multitask")
    parser.add_argument("--input-file", type=Path, default=default_training_data_file())
    parser.add_argument("--output-dir", type=Path, default=default_fold_output_dir("kmeans"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Also plot PCA/UMAP/t-SNE fold visualizations for multitask data.")
    args = parser.parse_args()

    if args.mode in ("multitask", "both"):
        summary_df, centers = save_multitask_kmeans_split(args.input_file, args.output_dir, args.n_splits, args.seed)
        print(f"Splits saved to: {args.output_dir}")
        print(summary_df)

        if args.plot:
            df = pd.read_csv(args.input_file)
            splits, centers, _ = kmeans_split(df, n_splits=args.n_splits, random_state=args.seed)
            X, _ = featurize_smiles_column(df, "SMILES")
            plot_embeddings_one_figure_per_mode(
                X=X,
                splits=splits,
                out_dir=args.output_dir,
                method_name="KMeans",
                cfg=EmbeddingCfg(random_state=args.seed),
                centers=centers,
                modes=("pca", "umap", "tsne"),
            )

    if args.mode in ("properties", "both"):
        summaries = save_property_kmeans_splits(args.input_file, args.n_splits, args.seed)
        for prop, summary_df in summaries.items():
            print(f"{prop} splits saved")
            print(summary_df)
