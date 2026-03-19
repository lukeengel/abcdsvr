"""Feature selection utilities (extracted from tsne/embeddings.py)."""

from __future__ import annotations


def get_imaging_columns(
    df, prefixes: list[str], roi_columns: list[str] | None = None
) -> list[str]:
    """Get imaging columns by prefix or explicit ROI list."""
    if roi_columns:
        return [c for c in roi_columns if c in df.columns]
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def get_roi_columns_from_config(
    data_config: dict, roi_networks: list[str]
) -> list[str]:
    """Extract ROI columns from data.yaml network definitions."""
    roi_features = data_config.get("roi_features", {})
    columns = []
    for net in roi_networks:
        net_def = roi_features.get(net, {})
        for feat_type in ("structural", "connectivity"):
            columns.extend(net_def.get(feat_type) or [])
    # Deduplicate preserving order
    return list(dict.fromkeys(columns))
