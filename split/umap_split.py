import argparse

from sklearn.cluster import AgglomerativeClustering

from split_utils import *


def run_umap(X: np.ndarray, cfg: EmbeddingCfg) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        n_components=cfg.umap_n_components,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    return reducer.fit_transform(X)


def run_agglomerative(embedding: np.ndarray, n_clusters: int, linkage: str) -> np.ndarray:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = clusterer.fit_predict(embedding)
    return labels


def umap_split(
        df: pd.DataFrame,
        n_splits: int = 5,
        random_state: int = 42,
        cfg: Optional[EmbeddingCfg] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    """Cluster a UMAP embedding with agglomerative clustering and build train/val/test splits."""
    X, row_idx = featurize_smiles_column(df, smiles_col="SMILES", radius=2, n_bits=1024)
    emb_cfg = cfg or EmbeddingCfg(random_state=random_state, umap_n_components=10)
    embedding = run_umap(X, emb_cfg)
    labels = run_agglomerative(embedding, n_clusters=n_splits, linkage=emb_cfg.linkage)
    splits = make_cv_splits(labels, n_splits=n_splits, random_state=random_state)
    return splits, row_idx, embedding


def save_multitask_umap_split(
        input_file: Path = default_training_data_file(),
        output_dir: Path = default_fold_output_dir("umap"),
        n_splits: int = 5,
        random_state: int = 42,
) -> pd.DataFrame:
    """Generate and save the multitask UMAP/Agglomerative split."""
    df = pd.read_csv(input_file)
    splits, row_idx, _ = umap_split(df, n_splits=n_splits, random_state=random_state)
    return attach_and_save_splits(df, row_idx, splits, output_dir)


def save_property_umap_splits(
        input_file: Path = default_training_data_file(),
        n_splits: int = 5,
        random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate and save per-property UMAP/Agglomerative splits after dropping missing target values."""
    df = pd.read_csv(input_file)
    property_dfs = make_property_dataframes(df)
    output_dirs = make_property_output_dirs(repo_root(), "umap")
    summaries: Dict[str, pd.DataFrame] = {}
    for prop, prop_df in property_dfs.items():
        splits, row_idx, _ = umap_split(prop_df, n_splits=n_splits, random_state=random_state)
        summaries[prop] = attach_and_save_splits(prop_df, row_idx, splits, output_dirs[prop])
    return summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UMAP/Agglomerative five-fold data splits.")
    parser.add_argument("--mode", choices=("multitask", "properties", "both"), default="multitask")
    parser.add_argument("--input-file", type=Path, default=default_training_data_file())
    parser.add_argument("--output-dir", type=Path, default=default_fold_output_dir("umap"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Also plot PCA/UMAP/t-SNE fold visualizations for multitask data.")
    args = parser.parse_args()

    if args.mode in ("multitask", "both"):
        summary_df = save_multitask_umap_split(args.input_file, args.output_dir, args.n_splits, args.seed)
        print(f"Splits saved to: {args.output_dir}")
        print(summary_df)

        if args.plot:
            df = pd.read_csv(args.input_file)
            splits, _, _ = umap_split(df, n_splits=args.n_splits, random_state=args.seed)
            X, _ = featurize_smiles_column(df, smiles_col="SMILES", radius=2, n_bits=1024)
            plot_embeddings_one_figure_per_mode(
                X=X,
                splits=splits,
                out_dir=args.output_dir,
                method_name="UMAP",
                cfg=EmbeddingCfg(random_state=args.seed),
                centers=None,
                modes=("pca", "umap", "tsne"),
            )

    if args.mode in ("properties", "both"):
        summaries = save_property_umap_splits(args.input_file, args.n_splits, args.seed)
        for prop, summary_df in summaries.items():
            print(f"{prop} splits saved")
            print(summary_df)
