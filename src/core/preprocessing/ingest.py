"""Data ingestion for ABCD Release 6 (parquet + CSV multi-format support)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _read_file(path: Path, usecols=None) -> pd.DataFrame:
    """Read a data file, detecting format by extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if usecols is not None:
            available = [c for c in usecols if c in df.columns]
            df = df[available]
        return df
    elif suffix == ".csv":
        if usecols is not None:
            available_cols = pd.read_csv(path, nrows=0).columns
            cols_to_read = [c for c in usecols if c in available_cols]
        else:
            cols_to_read = None
        return pd.read_csv(path, usecols=cols_to_read)
    elif suffix == ".tsv":
        if usecols is not None:
            available_cols = pd.read_csv(path, sep="\t", nrows=0).columns
            cols_to_read = [c for c in usecols if c in available_cols]
        else:
            cols_to_read = None
        return pd.read_csv(path, sep="\t", usecols=cols_to_read)
    else:
        raise ValueError(f"Unsupported file format: {suffix} for {path}")


def get_columns_for_file(env, filepath: str | Path) -> list[str] | None:
    """Return columns to load for a file based on config."""
    config = env.configs.data
    columns_cfg = config["columns"]["mapping"]
    files_cfg = config["files"]
    str_path = str(filepath)

    metadata_files: Sequence[str] = files_cfg.get("metadata", [])
    imaging_files: Sequence[str] = files_cfg.get("imaging", [])

    if str_path in metadata_files:
        metadata_cols = list(config["columns"].get("metadata", []))
        derived_cols = list(config["columns"].get("derived", []))
        mapping_cols = [
            value for value in columns_cfg.values() if isinstance(value, str)
        ]
        cols = metadata_cols + derived_cols + mapping_cols
        unique_cols = list(dict.fromkeys(cols))
        return unique_cols

    if str_path in imaging_files:
        return None  # Load all columns for imaging files

    raise ValueError(f"File {filepath} not found in config")


def load_and_merge(env) -> pd.DataFrame:
    """Load and outer-merge metadata and imaging files (parquet/CSV).

    Static files (listed in data.yaml under ``static_files``) are merged on
    ``participant_id`` only — they have one row per subject with no session_id.
    All other files are merged on ``[participant_id, session_id]``.
    """

    config = env.configs.data
    columns_cfg = config["columns"]["mapping"]
    files_cfg = config["files"]
    id_col = columns_cfg["id"]
    timepoint_col = columns_cfg["timepoint"]
    baseline_value = config["timepoints"]["baseline"]

    # Static files: merge on participant_id only (one row per subject)
    static_set = set(config.get("static_files", []))

    merged: pd.DataFrame | None = None
    for file in files_cfg["metadata"] + files_cfg["imaging"]:
        path = env.repo_root / file

        if not path.exists():
            print(f"  Warning: {file} not found, skipping")
            continue

        usecols = get_columns_for_file(env, file)
        df = _read_file(path, usecols=usecols)

        is_static = file in static_set

        if merged is None:
            # First file — if static, add a placeholder session_id
            if is_static and timepoint_col not in df.columns:
                print(f"  Static file: {file} ({len(df)} subjects)")
            elif timepoint_col not in df.columns:
                df[timepoint_col] = baseline_value
                print(f"  Added {timepoint_col}={baseline_value} to {file}")
            merged = df
            continue

        if is_static:
            # Static files: merge on id only (broadcasts to all timepoints)
            merge_keys = [id_col]
            # Drop session_id from static df if present to avoid conflicts
            if timepoint_col in df.columns:
                df = df.drop(columns=[timepoint_col])
        else:
            if timepoint_col not in df.columns:
                df[timepoint_col] = baseline_value
                print(f"  Added {timepoint_col}={baseline_value} to {file}")
            merge_keys = [id_col, timepoint_col]

        merged = merged.merge(df, on=merge_keys, how="outer")

    if merged is None:
        raise ValueError("No files loaded during merge")

    return merged
