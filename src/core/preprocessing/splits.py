"""Functions that build modeling splits and longitudinal merges."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def timepoint_split(env, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return baseline-only rows and a longitudinal copy."""
    config = env.configs.data
    timepoint_col = config["columns"]["mapping"]["timepoint"]
    baseline_timepoint = config["timepoints"]["baseline"]

    baseline_df = df[df[timepoint_col] == baseline_timepoint].copy()
    longitudinal_df = df.copy()
    return baseline_df, longitudinal_df


def create_modeling_splits(
    env, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split."""
    config = env.configs.data
    run_cfg = getattr(env.configs, "run", {}) or {}
    seed = run_cfg.get("seed", 42)

    ratio_cfg = config.get("splits", {})
    ratios = np.array(
        [
            float(ratio_cfg.get("train", 0.8)),
            float(ratio_cfg.get("val", 0.1)),
            float(ratio_cfg.get("test", 0.1)),
        ]
    )
    ratios /= ratios.sum()

    id_col = config["columns"]["mapping"]["id"]
    timepoint_col = config["columns"]["mapping"]["timepoint"]

    qc_pass_df = df[df["qc_pass"]].copy().reset_index(drop=True)
    if qc_pass_df.empty:
        raise ValueError("No QC-pass rows available for splitting")

    # No stratification for regression mode — use dummy labels
    labels = pd.Series(["all"] * len(qc_pass_df))

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=ratios[0],
        random_state=seed,
    )
    train_idx, remaining_idx = next(sss.split(qc_pass_df, labels))

    train = qc_pass_df.iloc[train_idx].copy()
    remaining = qc_pass_df.iloc[remaining_idx].copy()
    remaining_labels = labels.iloc[remaining_idx]

    if np.isclose(ratios[2], 0.0):
        val = remaining.copy()
        test = remaining.iloc[0:0].copy()
    else:
        remainder_ratio = ratios[1] + ratios[2]
        second_split = StratifiedShuffleSplit(
            n_splits=1,
            train_size=ratios[1] / remainder_ratio,
            random_state=seed,
        )
        val_idx, test_idx = next(second_split.split(remaining, remaining_labels))
        val = remaining.iloc[val_idx].copy()
        test = remaining.iloc[test_idx].copy()

    split_map = pd.concat(
        [
            train[[id_col, timepoint_col]].assign(split="train"),
            val[[id_col, timepoint_col]].assign(split="val"),
            test[[id_col, timepoint_col]].assign(split="test"),
        ],
        ignore_index=True,
    )

    return train, val, test, split_map


def merge_longitudinal(
    env,
    baseline_df: pd.DataFrame,
    full_df: pd.DataFrame,
    target_col: str,
    followup: str = "year2",
    imaging_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create wide-format dataframe: baseline imaging + baseline/followup target + delta.

    Inner join: only subjects with both baseline imaging AND follow-up target.

    When ``imaging_cols`` is provided, follow-up imaging columns are also merged
    and delta imaging columns are computed (followup - baseline).  This enables
    Q3 delta-brain analyses.

    Args:
        env: Environment with configs.
        baseline_df: Baseline-only dataframe with imaging features.
        full_df: Full longitudinal dataframe (all timepoints).
        target_col: Target column name (e.g. 'mh_y_pps__severity_score').
        followup: Key in data.yaml timepoints (e.g. 'year2', 'year4', 'year6').
        imaging_cols: Optional list of imaging column names.  When provided,
            follow-up imaging is merged and delta imaging columns are computed.

    Returns:
        Wide dataframe with columns:
            - All baseline imaging columns
            - {target_col}_baseline, {target_col}_{followup}, {target_col}_delta
            - (if imaging_cols) {col}_{followup} and {col}_delta for each imaging col
    """
    config = env.configs.data
    id_col = config["columns"]["mapping"]["id"]
    timepoint_col = config["columns"]["mapping"]["timepoint"]
    baseline_tp = config["timepoints"]["baseline"]
    followup_tp = config["timepoints"][followup]

    # Get baseline target
    bl_targets = full_df[full_df[timepoint_col] == baseline_tp][[id_col, target_col]].copy()
    bl_targets = bl_targets.rename(columns={target_col: f"{target_col}_baseline"})

    # Get follow-up target
    fu_targets = full_df[full_df[timepoint_col] == followup_tp][[id_col, target_col]].copy()
    fu_targets = fu_targets.rename(columns={target_col: f"{target_col}_{followup}"})

    # Merge: baseline imaging + baseline target + follow-up target
    wide = baseline_df.merge(bl_targets, on=id_col, how="inner")
    wide = wide.merge(fu_targets, on=id_col, how="inner")

    # Compute change score (target)
    bl_col = f"{target_col}_baseline"
    fu_col = f"{target_col}_{followup}"
    wide[f"{target_col}_delta"] = wide[fu_col] - wide[bl_col]

    # Required non-NaN columns for dropna guard
    required_cols = [bl_col, fu_col, f"{target_col}_delta"]

    # ── Optional: merge follow-up imaging and compute delta brain features ──
    if imaging_cols:
        present_img = [c for c in imaging_cols if c in full_df.columns]
        fu_imaging = full_df[full_df[timepoint_col] == followup_tp][
            [id_col] + present_img
        ].copy()
        fu_imaging = fu_imaging.rename(
            columns={c: f"{c}_{followup}" for c in present_img}
        )
        wide = wide.merge(fu_imaging, on=id_col, how="inner")

        # Compute delta: followup - baseline
        for col in present_img:
            delta_name = f"{col}_delta"
            wide[delta_name] = wide[f"{col}_{followup}"] - wide[col]
            required_cols.append(delta_name)

        n_delta = len(present_img)
        print(f"  Delta imaging: {n_delta} features computed ({followup} - baseline)")

    # Drop subjects with NaN in any required column
    wide = wide.dropna(subset=required_cols).reset_index(drop=True)

    print(f"  Longitudinal merge ({followup}): {len(wide)} subjects with baseline imaging + {followup} target")
    return wide
