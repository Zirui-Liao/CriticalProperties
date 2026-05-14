from rdkit.ML.Cluster import Butina
import argparse
import random

from split_utils import *


def compute_condensed_distance_list(fps) -> List[float]:
    """Condensed (upper-triangular) distance list for Butina (1 - Tanimoto)."""
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - sim for sim in sims])
    return dists


def butina_clusters(fps, cutoff_distance: float) -> List[Tuple[int, ...]]:
    """Return list of clusters, each a tuple of indices."""
    dists = compute_condensed_distance_list(fps)
    clusters = Butina.ClusterData(dists, len(fps), cutoff_distance, isDistData=True)
    return clusters


def assign_clusters_to_folds(clusters: List[Tuple[int, ...]], n_folds: int, seed: int) -> np.ndarray:
    """
    Randomly assign clusters to n_folds folds. Return fold_id (expanded by molecule index).
    Use round-robin with randomly shuffled cluster order to avoid large clusters being concentrated.
    """
    rng = random.Random(seed)
    cluster_indices = list(range(len(clusters)))
    rng.shuffle(cluster_indices)

    fold_of_cluster: Dict[int, int] = {}
    for rank, cid in enumerate(cluster_indices):
        fold_of_cluster[cid] = rank % n_folds

    n_mols = sum(len(c) for c in clusters)
    fold_id = np.empty(n_mols, dtype=int)
    for cid, members in enumerate(clusters):
        fold = fold_of_cluster[cid]
        for idx in members:
            fold_id[idx] = fold
    return fold_id


def butina_split(
        df: pd.DataFrame,
        n_splits: int = 5,
        random_state: int = 42,
        cutoff: float = 0.75,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray]:
    """Cluster molecules with Butina, assign clusters to folds, and build train/val/test splits."""
    _, row_idx = featurize_smiles_column(df, smiles_col="SMILES", radius=2, n_bits=1024)
    split_df = df.iloc[row_idx].reset_index(drop=True)
    fps = smiles_to_morgan_bitvects(split_df, radius=2, nBits=1024)
    clusters = butina_clusters(fps, cutoff)
    print(f"Total clusters found: {len(clusters)}")
    labels = assign_clusters_to_folds(clusters, n_splits, random_state)
    splits = make_cv_splits(labels, n_splits=n_splits, random_state=random_state)
    return splits, row_idx


def save_multitask_butina_split(
        input_file: Path = default_training_data_file(),
        output_dir: Path = default_fold_output_dir("butina"),
        n_splits: int = 5,
        random_state: int = 42,
        cutoff: float = 0.75,
) -> pd.DataFrame:
    """Generate and save the multitask Butina split."""
    df = pd.read_csv(input_file)
    splits, row_idx = butina_split(df, n_splits=n_splits, random_state=random_state, cutoff=cutoff)
    return attach_and_save_splits(df, row_idx, splits, output_dir)


def save_property_butina_splits(
        input_file: Path = default_training_data_file(),
        n_splits: int = 5,
        random_state: int = 42,
        cutoff: float = 0.75,
) -> Dict[str, pd.DataFrame]:
    """Generate and save per-property Butina splits after dropping missing target values."""
    df = pd.read_csv(input_file)
    property_dfs = make_property_dataframes(df)
    output_dirs = make_property_output_dirs(repo_root(), "butina")
    summaries: Dict[str, pd.DataFrame] = {}
    for prop, prop_df in property_dfs.items():
        splits, row_idx = butina_split(prop_df, n_splits=n_splits, random_state=random_state, cutoff=cutoff)
        summaries[prop] = attach_and_save_splits(prop_df, row_idx, splits, output_dirs[prop])
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Butina-cluster five-fold data splits.")
    parser.add_argument("--mode", choices=("multitask", "properties", "both"), default="multitask")
    parser.add_argument("--input-file", type=Path, default=default_training_data_file())
    parser.add_argument("--output-dir", type=Path, default=default_fold_output_dir("butina"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=float, default=0.75)
    parser.add_argument("--plot", action="store_true", help="Also plot PCA/UMAP/t-SNE fold visualizations for multitask data.")
    args = parser.parse_args()

    if args.mode in ("multitask", "both"):
        summary_df = save_multitask_butina_split(
            args.input_file, args.output_dir, args.n_splits, args.seed, args.cutoff
        )
        print(f"Splits saved to: {args.output_dir}")
        print(summary_df)

        if args.plot:
            df = pd.read_csv(args.input_file)
            splits, _ = butina_split(df, n_splits=args.n_splits, random_state=args.seed, cutoff=args.cutoff)
            X, _ = featurize_smiles_column(df, smiles_col="SMILES", radius=2, n_bits=1024)
            plot_embeddings_one_figure_per_mode(
                X=X,
                splits=splits,
                out_dir=args.output_dir,
                method_name="Butina",
                cfg=EmbeddingCfg(random_state=args.seed),
                centers=None,
                modes=("pca", "umap", "tsne"),
            )

    if args.mode in ("properties", "both"):
        summaries = save_property_butina_splits(args.input_file, args.n_splits, args.seed, args.cutoff)
        for prop, summary_df in summaries.items():
            print(f"{prop} splits saved")
            print(summary_df)
