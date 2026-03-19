"""Column transformations (recoding, binning)."""

from __future__ import annotations

import pandas as pd


def recode(env, df: pd.DataFrame) -> pd.DataFrame:
    """Apply value mappings from config derived entries."""
    config = env.configs.data
    derived_cfg = config.get("derived", {})

    for var_name, var_def in derived_cfg.items():
        if not isinstance(var_def, dict) or "map" not in var_def:
            continue

        source_col = var_def.get("source")
        target_col = var_def.get("target", f"{var_name}_mapped")

        if source_col and source_col in df.columns:
            mapping = var_def["map"]
            df[target_col] = df[source_col].map(mapping)
            n_mapped = df[target_col].notna().sum()
            n_total = len(df)
            print(f"  Recoded {source_col} -> {target_col}: {n_mapped}/{n_total} mapped")

    return df
