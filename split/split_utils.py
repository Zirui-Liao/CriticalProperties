from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

import umap


property_columns = ("Tc", "Pc", "Vc", "omega")
paper_plot_style = {
    "font.family": "Times New Roman",
    "font.sans-serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
}

def apply_split_plot_style() -> None:
    """Apply the Matplotlib style shared by split analysis figures."""
    plt.rcParams.update(paper_plot_style)


apply_split_plot_style()

# # for umap and butina
cluster_colors = {
    3: "#1f77b4",  # blue
    4: "#ff7f0e",  # orange
    1: "#2ca02c",  # green
    0: "#d62728",  # red
    2: "#9467bd",  # purple
}
# for kmeans:
# cluster_colors = {
#     0: "#1f77b4",  # blue
#     1: "#ff7f0e",  # orange
#     2: "#2ca02c",  # green
#     3: "#d62728",  # red
#     4: "#9467bd",  # purple
# }


# -------------------- Feature/utility functions (common across scripts) --------------------
def smiles_to_fp_array(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Convert a SMILES into a binary Morgan fingerprint (numpy int array).
    """
    mol = Chem.MolFromSmiles(smiles)
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def smiles_to_morgan_bitvects(df, smiles_col='SMILES', radius=2, n_bits=1024):
    """
    Convert the SMILES column in the DataFrame to a list of Morgan fingerprints (ExplicitBitVect) in batch.
    Returns:
    bitvects : list[ExplicitBitVect]
        A list of fingerprint objects for each molecule.
    """
    bitvects = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
        bitvects.append(fp)
    return bitvects


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True)


def compute_max_tanimoto_similarities(fps: Sequence) -> np.ndarray:
    """Compute each fingerprint's maximum Tanimoto similarity to any other fingerprint."""
    max_sims: List[float] = []
    for i, fp in enumerate(fps):
        sims = list(DataStructs.BulkTanimotoSimilarity(fp, fps))
        if i < len(sims):
            sims[i] = 0.0
        max_sims.append(max(sims) if sims else 0.0)
    return np.asarray(max_sims, dtype=float)


def filter_by_max_similarity_threshold(
        df: pd.DataFrame,
        row_idx: np.ndarray,
        max_similarities: np.ndarray,
        cutoff: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into kept and removed sets using max Tanimoto similarity > cutoff."""
    keep_mask = max_similarities > cutoff
    kept_idx = [row_idx[i] for i, keep in enumerate(keep_mask) if bool(keep)]
    removed_idx = [row_idx[i] for i, keep in enumerate(keep_mask) if not bool(keep)]
    kept_df = df.iloc[kept_idx].reset_index(drop=True)
    removed_df = df.iloc[removed_idx].reset_index(drop=True)
    return kept_df, removed_df


def extract_threshold_from_name(name: str) -> float:
    """Extract a threshold value from names like exp_data_no_outliers_035.csv."""
    if pd.isna(name):
        return 0.0

    text = str(name)
    match = re.search(r"no_outliers[_\-]?(\d+)", text)
    if match:
        token = match.group(1)
        if len(token) == 1:
            return float(token)
        return float(token) / (10 ** (len(token) - 1))

    decimal_match = re.search(r"(\d+\.\d+)", text)
    if decimal_match:
        return float(decimal_match.group(1))

    integer_match = re.search(r"(\d+)", text)
    if integer_match:
        token = integer_match.group(1)
        if len(token) == 1:
            return float(token)
        return float(token) / (10 ** (len(token) - 1))

    return 0.0


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series into [0, 1]."""
    x = series.astype(float)
    xmin, xmax = float(x.min()), float(x.max())
    if np.isclose(xmax - xmin, 0.0):
        return pd.Series(np.ones(len(x)), index=series.index)
    return (x - xmin) / (xmax - xmin)


def featurize_smiles_column(
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        radius: int = 2,
        n_bits: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorize a DataFrame SMILES column into fingerprints and return (x, row_idx).

    SMILES are assumed to have been validated before running split generation.
    """
    fps = [
        smiles_to_fp_array(s, radius=radius, n_bits=n_bits)
        for s in df[smiles_col].tolist()
    ]
    return np.vstack(fps).astype(int), np.arange(len(df))


def sample_validation_indices(labels: np.ndarray, test_mask: np.ndarray, val_frac: float,
                              random_state: int) -> np.ndarray:
    """Sample validation indices from non-test set using stratified sampling by cluster labels."""
    non_test_idx = np.where(~test_mask)[0]
    if non_test_idx.size == 0:
        return np.array([], dtype=int)
    stratify_labels = labels[non_test_idx]
    try:
        _, val_idx_rel = train_test_split(
            non_test_idx,
            test_size=val_frac,
            random_state=random_state,
            stratify=stratify_labels if len(np.unique(stratify_labels)) > 1 else None,
        )
    except ValueError:
        # In case some classes are too small, fall back to unstratified sampling
        _, val_idx_rel = train_test_split(
            non_test_idx,
            test_size=val_frac,
            random_state=random_state,
            stratify=None,
        )
    return val_idx_rel


def make_cv_splits(labels: np.ndarray, n_splits: int, random_state: int) -> Dict[int, Dict[str, np.ndarray]]:
    """Create CV splits on valid rows only.

    For fold f: test = (labels == f). Validation is sampled from remaining rows with
    val_frac = 1/(n_splits-1). Train = remaining non-test & non-val.
    Returns: fold_id -> {"train": idx, "val": idx, "test": idx}
    """
    uniq = np.unique(labels)
    assert n_splits == len(uniq), (
        "n_splits should match the number of clusters produced; "
        f"got n_splits={n_splits}, unique clusters={len(uniq)}"
    )

    val_frac = 1.0 / (n_splits - 1)
    splits: Dict[int, Dict[str, np.ndarray]] = {}

    for f in range(n_splits):
        test_mask = np.asarray(labels == f)
        test_idx = np.where(test_mask)[0]
        val_idx = sample_validation_indices(labels, test_mask, val_frac=val_frac, random_state=random_state + f)
        val_mask = np.zeros_like(labels, dtype=bool)
        val_mask[val_idx] = True
        train_idx = np.where(~test_mask & ~val_mask)[0]
        splits[f] = {"train": train_idx, "val": val_idx, "test": test_idx}
    return splits


def attach_and_save_splits(
        df: pd.DataFrame,
        row_idx: np.ndarray,
        splits: Dict[int, Dict[str, np.ndarray]],
        out_dir: Path,
        *,
        counts_only: bool = False,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"fold": [], "train": [], "val": [], "test": []}

    for f, part in splits.items():
        split_col = pd.Series(index=df.index, data="train", dtype="object")
        split_col.iloc[row_idx[part["train"]]] = "train"
        split_col.iloc[row_idx[part["val"]]] = "val"
        split_col.iloc[row_idx[part["test"]]] = "test"

        fold_df = df.copy()
        fold_df["split"] = split_col

        if not counts_only:
            fold_path = out_dir / f"fold_{f}.csv"
            fold_df.to_csv(fold_path, index=False)

        summary["fold"].append(f)
        summary["train"].append((fold_df["split"] == "train").sum())
        summary["val"].append((fold_df["split"] == "val").sum())
        summary["test"].append((fold_df["split"] == "test").sum())

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "cv_counts_summary.csv", index=False)
    return summary_df


# -------------------- Split/labels helpers --------------------
def build_fold_label_vector(n_samples: int, splits: Mapping[int, Mapping[str, np.ndarray]]) -> np.ndarray:
    """Create a label vector y (shape [n_samples]) where label == fold_id for indices in that fold's TEST set.
    Assumes k-fold so each sample appears in exactly one test set.
    """
    y = np.full((n_samples,), fill_value=-1, dtype=int)
    for f, part in splits.items():
        test_idx = np.asarray(part["test"])  # required key
        y[test_idx] = int(f)
    if np.any(y < 0):
        # If any sample wasn't covered, we still fill them with a dummy class to keep metrics defined.
        # But for k-fold, this shouldn't happen.
        missing = np.where(y < 0)[0]
        if len(missing) > 0:
            # put them into a catch-all class = max_fold+1
            y[missing] = (max(splits.keys()) + 1) if len(splits) else 0
    return y


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_dir() -> Path:
    """Return the repository split directory."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Return the repository root directory."""
    return split_dir().parent


def default_training_data_file(suffix: str = "035") -> Path:
    """Return the default cleaned experimental data file for a threshold suffix."""
    return repo_root() / "training_data" / f"exp_data_no_outliers_{suffix}.csv"


def default_fold_output_dir(split_name: str, suffix: str = "no_outliers_035") -> Path:
    """Return the default multitask fold output directory for a split method."""
    return repo_root() / "data_folds" / f"{split_name}_{suffix}"


def make_property_dataframes(
        df: pd.DataFrame,
        *,
        smiles_col: str = "SMILES",
        properties: Iterable[str] = property_columns,
) -> Dict[str, pd.DataFrame]:
    """Return one SMILES/property dataframe per target property, dropping rows missing that target."""
    return {
        prop: df[[smiles_col, prop]].dropna(subset=[prop]).reset_index(drop=True)
        for prop in properties
    }


def make_property_output_dirs(
        root_dir: Path,
        split_name: str,
        *,
        suffix: str = "no_outliers_035",
        properties: Iterable[str] = property_columns,
) -> Dict[str, Path]:
    """Build and create the per-property fold output directories used by split scripts."""
    output_dirs = {
        prop: root_dir / "data_folds" / f"{split_name}_{prop}_{suffix}"
        for prop in properties
    }
    for out_dir in output_dirs.values():
        ensure_dir(out_dir)
    return output_dirs


# -------------------- Clustering-quality evaluation (common policy) --------------------
def evaluate_split_clustering_quality(x: np.ndarray, splits: Mapping[int, Mapping[str, np.ndarray]]) -> Dict[
    str, float]:
    """Treat each fold's TEST set as a class label and compute clustering quality metrics on x.
    Metrics: silhouette (higher better), Calinski–Harabasz (higher better), Davies–Bouldin (lower better).
    """
    y = build_fold_label_vector(len(x), splits)
    # edge cases: need at least 2 labels, and each metric has its own numeric constraints
    labels_present = np.unique(y)
    if any(np.sum(y == label) < 2 for label in np.unique(y)) or labels_present.size < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    out: Dict[str, float] = {"silhouette": float(silhouette_score(x, y)),
                             "calinski_harabasz": float(calinski_harabasz_score(x, y)),
                             "davies_bouldin": float(davies_bouldin_score(x, y))}

    return out


def save_clustering_metrics(metrics: Dict[str, float], out_dir: Path, method_name: str):
    ensure_dir(out_dir)
    dst = out_dir / "clustering_metrics_by_fold_as_class.csv"
    row = {"method": method_name, **metrics}
    if dst.exists():
        prev = pd.read_csv(dst)
        df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(dst, index=False)


# -------------------- 2D Embeddings & plotting (PCA/UMAP/t-SNE) --------------------
@dataclass
class EmbeddingCfg:
    random_state: int = 42
    # UMAP params
    umap_n_components: int = 2
    umap_n_neighbors: int = 100
    umap_min_dist: float = 0.0
    umap_metric: str = "jaccard"  # good for binary fingerprints
    linkage: str = "ward"  # 'ward' requires Euclidean distances; used on embedding space
    # t-SNE params
    tsne_perplexity: Optional[float] = None  # if None -> auto
    tsne_n_iter: int = 1000


def _fit_pca(x: np.ndarray, cfg: EmbeddingCfg) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=cfg.random_state)
    coords = pca.fit_transform(x)
    return coords, pca


def _fit_umap(x: np.ndarray, cfg: EmbeddingCfg) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    return reducer.fit_transform(x)


def _fit_tsne(x: np.ndarray, cfg: EmbeddingCfg) -> np.ndarray:
    n = len(x)
    perpl = cfg.tsne_perplexity if cfg.tsne_perplexity is not None else max(5, min(30, int((n - 1) / 3)))
    tsne = TSNE(
        n_components=2,
        perplexity=perpl,
        learning_rate="auto",
        init="pca",
        n_iter=cfg.tsne_n_iter,
        random_state=cfg.random_state,
        verbose=0,
    )
    return tsne.fit_transform(x)


def _plot_all_folds_one_figure(
        coords: np.ndarray,
        splits: Mapping[int, Mapping[str, np.ndarray]],
        out_file: Path,
        title: str,
        *,
        centers_2d: Optional[np.ndarray] = None,
        font_size: int = 12
):
    ensure_dir(out_file.parent)
    plt.figure(figsize=(7.2, 6.2))
    # plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.12, label="All")
    for f in sorted(splits.keys()):
        test_idx = np.asarray(splits[f]["test"])
        color = cluster_colors[int(f)]

        plt.scatter(
            coords[test_idx, 0],
            coords[test_idx, 1],
            s=18,
            alpha=0.95,
            label=f"Fold {f}",
            color=color,
        )

    if centers_2d is not None and len(centers_2d) > 0:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], s=90, marker="X", label="Centers")
    # plt.title(title)
    plt.xlabel("Dim-1", fontsize=font_size)
    plt.ylabel("Dim-2", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(ncol=1, fontsize=font_size - 6, frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=240, bbox_inches='tight')
    plt.savefig(out_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(out_file.with_suffix('.svg'), bbox_inches='tight')
    plt.close()


def plot_embeddings_one_figure_per_mode(
        x: np.ndarray,
        splits: Mapping[int, Mapping[str, np.ndarray]],
        out_dir: Path,
        method_name: str,
        cfg: EmbeddingCfg = EmbeddingCfg(),
        *,
        centers: Optional[np.ndarray] = None,
        modes: Iterable[str] = ("pca", "umap", "tsne")
):
    modes = tuple(modes)
    coords_cache: Dict[str, np.ndarray] = {}
    pca_model: Optional[PCA] = None
    if "pca" in modes:
        coords_cache["pca"], pca_model = _fit_pca(x, cfg)
    if "umap" in modes:
        coords_cache["umap"] = _fit_umap(x, cfg)
    if "tsne" in modes:
        coords_cache["tsne"] = _fit_tsne(x, cfg)
    centers_pca_2d: Optional[np.ndarray] = None
    if centers is not None and pca_model is not None:
        try:
            centers_pca_2d = pca_model.transform(centers)
        except Exception:
            centers_pca_2d = None
    plots_dir = Path(out_dir)
    if "pca" in modes:
        _plot_all_folds_one_figure(coords_cache["pca"], splits, plots_dir / f"{method_name.lower()}_pca_tests.png",
                                   f"{method_name} — PCA", centers_2d=centers_pca_2d, font_size=20)
    if "umap" in modes:
        _plot_all_folds_one_figure(coords_cache["umap"], splits, plots_dir / f"{method_name.lower()}_umap_tests.png",
                                   f"{method_name} — UMAP", font_size=20)
    if "tsne" in modes:
        _plot_all_folds_one_figure(coords_cache["tsne"], splits, plots_dir / f"{method_name.lower()}_tsne_tests.png",
                                   f"{method_name} — t-SNE", font_size=20)
