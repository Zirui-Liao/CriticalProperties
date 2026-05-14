from pathlib import Path

import pandas as pd

from split_utils import ensure_dir


def smiles_only_allowed(smiles: str, allowed_chars: set[str]) -> bool:
    """Return True when a SMILES string uses only characters observed in experimental SMILES."""
    return set(smiles) <= allowed_chars


def generate_pretraining_data(
        est_path: Path,
        exp_path: Path,
        output_path: Path,
) -> pd.DataFrame:
    """Filter raw estimated data for pretraining and save the result."""
    est_df = pd.read_csv(est_path)
    exp_df = pd.read_csv(exp_path)

    if "SMILES" not in est_df.columns or "SMILES" not in exp_df.columns:
        raise KeyError("Both CSV files must contain a 'SMILES' column.")

    exp_smiles_series = exp_df["SMILES"].astype(str)
    allowed_chars = set("".join(exp_smiles_series.tolist()))

    est_smiles_series = est_df["SMILES"].astype(str)
    mask_allowed = est_smiles_series.apply(lambda smiles: smiles_only_allowed(smiles, allowed_chars))
    est_df_allowed = est_df[mask_allowed].reset_index(drop=True)

    exp_smiles_set = set(exp_smiles_series.tolist())
    pretrain_df = est_df_allowed[~est_df_allowed["SMILES"].isin(exp_smiles_set)].reset_index(drop=True)

    ensure_dir(output_path.parent)
    pretrain_df.to_csv(output_path, index=False)

    print(f"Original est_data rows: {len(est_df)}")
    print(f"Rows after removing rare-char SMILES (not in exp charset): {len(est_df_allowed)}")
    print(f"Rows removed due to rare chars: {len(est_df) - len(est_df_allowed)}")
    print(f"Rows after additionally removing overlaps with exp: {len(pretrain_df)}")
    print(f"Total rows removed (rare chars + overlaps): {len(est_df) - len(pretrain_df)}")
    print(f"Cleaned pretraining dataset saved to: {output_path}")
    return pretrain_df


if __name__ == "__main__":
    current_path = Path(__file__).resolve().parent
    generate_pretraining_data(
        est_path=current_path.parent / "raw_data" / "est_data.csv",
        exp_path=current_path.parent / "training_data" / "exp_data_no_outliers_035.csv",
        output_path=current_path.parent / "training_data" / "pretraining_data_035.csv",
    )
