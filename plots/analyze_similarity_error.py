import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plots_utils import (
    apply_paper_plot_style,
    classical_mtl_root,
    classical_stl_root,
    fold_prediction_path,
    make_output_dir,
    mtl_experiments,
    prediction_column,
    properties,
    split_file,
    stl_prediction_root,
)


def smiles_to_fp(smiles: str, radius: int, n_bits: int):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_max_tanimoto_to_train(train_smiles, test_smiles, radius: int, n_bits: int):
    from rdkit import DataStructs

    train_fps = [
        fp
        for fp in (smiles_to_fp(smi, radius=radius, n_bits=n_bits) for smi in train_smiles)
        if fp is not None
    ]
    if not train_fps:
        return [np.nan] * len(test_smiles)

    max_sims = []
    for smi in test_smiles:
        fp = smiles_to_fp(smi, radius=radius, n_bits=n_bits)
        if fp is None:
            max_sims.append(np.nan)
            continue
        max_sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return max_sims


def build_bins_and_labels(bin_start: float, bin_end: float, bin_width: float):
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    if bin_end <= bin_start:
        raise ValueError("bin_end must be greater than bin_start.")

    n_steps = int(np.round((bin_end - bin_start) / bin_width))
    if not np.isclose(bin_start + n_steps * bin_width, bin_end):
        raise ValueError("bin range must be divisible by bin_width.")

    bins = [bin_start + i * bin_width for i in range(n_steps + 1)]
    labels = [
        f"[{bins[i]:.1f},{bins[i + 1]:.1f}]" if i == 0 else f"({bins[i]:.1f},{bins[i + 1]:.1f}]"
        for i in range(n_steps)
    ]
    return bins, labels


def normalize_predictions(pred_df: pd.DataFrame, props):
    out = pred_df[["SMILES"]].copy()
    for prop in props:
        try:
            col = prediction_column(pred_df, prop)
        except KeyError:
            continue
        out[f"{prop}_pred"] = pred_df[col]
    return out


def experiment_catalog():
    catalog = {}
    for model_family in ("mpnn", "transformer"):
        for finetuned, tag in ((False, "nofinetune"), (True, "finetune")):
            for experiment in mtl_experiments(model_family, finetuned=finetuned):
                name = f"{model_family}_{experiment['split_type']}_mtl_{tag}"
                catalog[name] = {
                    "name": name,
                    "kind": "mtl",
                    "model_family": model_family,
                    "split_type": experiment["split_type"],
                    "root": experiment["root"],
                }
        for split_name in ("kmeans", "random"):
            name = f"{model_family}_{split_name}_stl"
            catalog[name] = {
                "name": name,
                "kind": "stl",
                "model_family": model_family,
                "split_type": split_name,
            }

    for split_name in ("kmeans", "random"):
        name = f"clscl_{split_name}_mtl"
        catalog[name] = {
            "name": name,
            "kind": "mtl",
            "model_family": "clscl",
            "split_type": split_name,
            "root": classical_mtl_root(split_name),
        }
        for model_name in ("mlp", "rf", "svm", "xgboost"):
            name = f"clscl_{split_name}_{model_name}_stl"
            catalog[name] = {
                "name": name,
                "kind": "classical_stl",
                "model_family": "clscl",
                "split_type": split_name,
                "model_name": model_name,
            }
    return catalog


def split_train_test_smiles(split_df: pd.DataFrame):
    if "split" not in split_df.columns:
        raise KeyError("Split CSV must contain a 'split' column.")
    test_df = split_df[split_df["split"] == "test"].copy()
    train_smiles = split_df.loc[split_df["split"] != "test", "SMILES"].dropna().unique().tolist()
    return train_smiles, test_df


def collect_mtl_samples(experiment: dict, props, n_folds: int, radius: int, n_bits: int):
    rows = []
    for fold in range(n_folds):
        fold_split = pd.read_csv(split_file(experiment["split_type"], fold=fold))
        train_smiles, test_df = split_train_test_smiles(fold_split)
        pred_path = fold_prediction_path(experiment["root"], fold)
        if not pred_path.exists():
            print(f"[WARN] Missing predictions: {pred_path}")
            continue

        pred_df = normalize_predictions(pd.read_csv(pred_path), props)
        merged = test_df[["SMILES", *props]].merge(pred_df, on="SMILES", how="inner")
        if merged.empty:
            print(f"[WARN] No overlapping SMILES: {pred_path}")
            continue

        merged["max_tanimoto_to_train"] = compute_max_tanimoto_to_train(
            train_smiles,
            merged["SMILES"].tolist(),
            radius=radius,
            n_bits=n_bits,
        )
        merged["fold"] = fold
        for prop in props:
            pred_col = f"{prop}_pred"
            if pred_col not in merged.columns:
                continue
            prop_df = merged[["SMILES", "fold", "max_tanimoto_to_train", prop, pred_col]].dropna().copy()
            prop_df["property"] = prop
            prop_df["abs_error"] = (prop_df[pred_col] - prop_df[prop]).abs()
            rows.append(prop_df[["SMILES", "fold", "property", "max_tanimoto_to_train", "abs_error"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stl_root_for_experiment(experiment: dict, prop: str) -> Path:
    if experiment["kind"] == "classical_stl":
        return classical_stl_root(experiment["split_type"]) / prop / experiment["model_name"]
    return stl_prediction_root(experiment["model_family"], experiment["split_type"], prop)


def collect_stl_samples(experiment: dict, props, n_folds: int, radius: int, n_bits: int):
    rows = []
    for prop in props:
        for fold in range(n_folds):
            fold_split = pd.read_csv(split_file(experiment["split_type"], prop=prop, fold=fold))
            train_smiles, test_df = split_train_test_smiles(fold_split)
            pred_path = fold_prediction_path(stl_root_for_experiment(experiment, prop), fold)
            if not pred_path.exists():
                print(f"[WARN] Missing predictions: {pred_path}")
                continue

            pred_df = normalize_predictions(pd.read_csv(pred_path), [prop])
            pred_col = f"{prop}_pred"
            if pred_col not in pred_df.columns:
                print(f"[WARN] Missing {pred_col}: {pred_path}")
                continue

            merged = test_df[["SMILES", prop]].merge(pred_df, on="SMILES", how="inner").dropna()
            if merged.empty:
                print(f"[WARN] No overlapping SMILES: {pred_path}")
                continue

            merged["max_tanimoto_to_train"] = compute_max_tanimoto_to_train(
                train_smiles,
                merged["SMILES"].tolist(),
                radius=radius,
                n_bits=n_bits,
            )
            merged["fold"] = fold
            merged["property"] = prop
            merged["abs_error"] = (merged[pred_col] - merged[prop]).abs()
            rows.append(merged[["SMILES", "fold", "property", "max_tanimoto_to_train", "abs_error"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_binned_error_grid(all_samples_df, props, out_dir, labels, min_count, bar_color, right_axis_color, prop_colors, font_size, left_y_lims=None):
    import matplotlib.pyplot as plt

    df = all_samples_df.copy()
    df["sim_bin"] = pd.Categorical(df["sim_bin"], categories=labels, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()
    ylabel_map = {
        "Tc": r"$|\Delta T_c|$ / K",
        "Pc": r"$|\Delta P_c|$ / bar",
        "Vc": r"$|\Delta V_c|$ / mL/mol",
        "omega": r"$|\Delta \omega|$",
    }
    x_all = np.arange(len(labels))
    pos_map = {label: i for i, label in enumerate(labels)}

    for idx, (ax, prop) in enumerate(zip(axes, props)):
        prop_df = df[df["property"] == prop]
        count_series = prop_df.groupby("sim_bin", observed=False).size().reindex(labels, fill_value=0)
        valid_labels = [label for label in labels if count_series.loc[label] >= min_count]
        invalid_labels = [label for label in labels if count_series.loc[label] < min_count]
        if not valid_labels:
            ax.axis("off")
            continue

        x_valid = [pos_map[label] for label in valid_labels]
        counts_valid = count_series.loc[valid_labels].astype(int).values
        line_color = prop_colors.get(prop, "black")

        ax_right = ax.twinx()
        ax_right.bar(x_valid, counts_valid, alpha=0.18, width=0.85, linewidth=0, color=bar_color, zorder=0)
        ax_right.set_ylabel("Count", color=right_axis_color, fontsize=font_size)
        ax_right.tick_params(axis="y", colors=right_axis_color, labelsize=font_size)
        ax_right.spines["right"].set_color(right_axis_color)
        ax_right.spines["top"].set_visible(False)
        ax_right.set_ylim(0, max(1, counts_valid.max()) * 1.15)

        violin_data = [
            prop_df.loc[prop_df["sim_bin"] == label, "abs_error"].dropna().values
            for label in valid_labels
        ]
        if any(len(values) > 0 for values in violin_data):
            vp = ax.violinplot(
                violin_data,
                positions=x_valid,
                widths=0.8,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(line_color)
                body.set_edgecolor(line_color)
                body.set_alpha(0.20)
                body.set_linewidth(0.8)
            ax.boxplot(
                violin_data,
                positions=x_valid,
                widths=0.28,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", markersize=3.0, markerfacecolor="white", markeredgecolor=line_color),
                medianprops=dict(color=line_color, linewidth=1.4),
                whiskerprops=dict(color=line_color, linewidth=1.0),
                capprops=dict(color=line_color, linewidth=1.0),
                boxprops=dict(facecolor="white", edgecolor=line_color, linewidth=1.1),
            )

        for label in invalid_labels:
            ax.axvline(x=pos_map[label], color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)

        ax.set_ylabel(ylabel_map.get(prop, "|error|"), fontsize=font_size, color=line_color)
        ax.tick_params(axis="y", labelsize=font_size, colors=line_color)
        ax.tick_params(axis="x", labelsize=font_size - 2)
        ax.spines["left"].set_color(line_color)
        ax.set_zorder(2)
        ax.patch.set_alpha(0)
        if left_y_lims is not None and prop in left_y_lims:
            ax.set_ylim(*left_y_lims[prop])
        if idx in (2, 3):
            ax.set_xlabel("Max Tanimoto similarity to train (binned)", fontsize=font_size)

    xticks_show = x_all[::2]
    xticklabels_show = [labels[i] for i in range(0, len(labels), 2)]
    for ax in axes:
        ax.set_xticks(xticks_show)
        ax.set_xticklabels(xticklabels_show, rotation=45)

    plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.98])
    png_path = out_dir / "similarity_vs_error_binned_box_violin_all_props.png"
    svg_path = out_dir / "similarity_vs_error_binned_box_violin_all_props.svg"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}, {svg_path}")


def run_experiment(experiment, args, labels, bins):
    props = properties
    if experiment["kind"] == "mtl":
        samples_df = collect_mtl_samples(experiment, props, args.n_folds, args.radius, args.n_bits)
    else:
        samples_df = collect_stl_samples(experiment, props, args.n_folds, args.radius, args.n_bits)

    if samples_df.empty:
        print(f"[WARN] No samples collected for {experiment['name']}; skipped.")
        return

    out_dir = make_output_dir(f"{experiment['name']}_similarity_error")
    samples_df["sim_bin"] = pd.cut(
        samples_df["max_tanimoto_to_train"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    per_sample_path = out_dir / "similarity_vs_error_per_sample.csv"
    samples_df.to_csv(per_sample_path, index=False)
    print(f"Saved: {per_sample_path}")

    counts = (
        samples_df.groupby(["property", "sim_bin"], observed=False)
        .size()
        .reset_index(name="count")
    )
    counts_path = out_dir / "similarity_vs_error_binned_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"Saved: {counts_path}")

    plot_binned_error_grid(
        samples_df,
        props,
        out_dir,
        labels,
        min_count=args.min_count,
        bar_color=args.bar_color,
        right_axis_color=args.right_axis_color,
        prop_colors={"Tc": "blue", "Pc": "orange", "Vc": "green", "omega": "red"},
        font_size=args.font_size,
        # left_y_lims={"Tc": (0, 80), "Pc": (0, 8), "Vc": (0, 150), "omega": (0, 0.25)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze max train similarity vs prediction error.")
    parser.add_argument("--experiment", default="mpnn_kmeans_mtl_finetune")
    parser.add_argument("--all", action="store_true", help="Run every known experiment.")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--bin-start", type=float, default=0.0)
    parser.add_argument("--bin-end", type=float, default=1.0)
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--bar-color", default="gray")
    parser.add_argument("--right-axis-color", default="gray")
    parser.add_argument("--font-size", type=int, default=18)
    args = parser.parse_args()

    catalog = experiment_catalog()
    if args.list_experiments:
        for name in sorted(catalog):
            print(name)
        return

    if not args.all and args.experiment not in catalog:
        choices = ", ".join(sorted(catalog))
        raise ValueError(f"Unknown experiment: {args.experiment}. Choices: {choices}")

    apply_paper_plot_style()
    bins, labels = build_bins_and_labels(args.bin_start, args.bin_end, args.bin_width)
    selected = catalog.values() if args.all else [catalog[args.experiment]]
    for experiment in selected:
        run_experiment(experiment, args, labels, bins)


if __name__ == "__main__":
    main()
