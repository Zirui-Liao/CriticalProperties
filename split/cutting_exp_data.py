from pathlib import Path

import numpy as np
import pandas as pd

from split_utils import (
    compute_max_tanimoto_similarities,
    ensure_dir,
    featurize_smiles_column,
    filter_by_max_similarity_threshold,
    smiles_to_morgan_bitvects,
)


def main(cutoff=0.35):
    """Remove experimental molecules whose max pairwise Tanimoto similarity is <= cutoff."""
    current_dir = Path(__file__).resolve().parent
    input_dir = current_dir.parent / "raw_data"
    output_dir = current_dir.parent / "training_data"
    input_file = input_dir / "exp_data.csv"
    output_file = output_dir / f"exp_data_no_outliers_{str(cutoff).replace('.', '')}.csv"

    df = pd.read_csv(input_file)
    print(f"Loaded dataset: {len(df)} rows")

    _, row_idx = featurize_smiles_column(df, smiles_col="SMILES", radius=2, n_bits=1024)
    print(f"Molecules: {len(row_idx)} / {len(df)}")

    bitvecs = smiles_to_morgan_bitvects(df.iloc[row_idx], smiles_col="SMILES", radius=2, nBits=1024)
    max_similarities = compute_max_tanimoto_similarities(bitvecs)
    clean_df, removed_df = filter_by_max_similarity_threshold(df, row_idx, max_similarities, cutoff)

    print(f"Detected {len(removed_df)} outliers (max Tanimoto <= {cutoff})")

    ensure_dir(output_dir)
    clean_df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")


if __name__ == "__main__":
    for cutoff in np.linspace(0.05, 0.5, 10):
        main(cutoff=round(cutoff, 2))
