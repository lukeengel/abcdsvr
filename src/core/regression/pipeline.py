"""Regression pipeline for predicting psychosis severity with nested CV.

Per-fold processing order (ratio method — default):
  1. ICV ratio correction (deterministic, per-subject)
  2. ComBat harmonization (fit on train, apply to test)
  3. Feature transform (AI, total, or raw L/R)
  4. StandardScaler
  5. SVR / Ridge

Per-fold processing order (residualize fallback):
  1. ComBat harmonization (fit on train, apply to test)
  2. ICV residualization (fit on train, apply to test)
  3. Feature transform (AI, total, or raw L/R)
  4. StandardScaler
  5. SVR / Ridge
"""

import io
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..features import get_imaging_columns, get_roi_columns_from_config
from ..preprocessing.tbv_correction import (
    apply_icv_correction,
    apply_icv_ratio_correction,
    fit_icv_correction,
    identify_thickness_features,
    identify_volume_features,
)
from .evaluation import aggregate_cv_results, compute_regression_metrics
from .models import MODEL_REGISTRY, create_baseline, model_supports_sample_weight
from .run_tracker import save_run_metadata
from .visualization import (
    create_summary_figure,
    plot_coefficients,
    plot_permutation_importance,
    plot_predictions,
    plot_residuals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harmonization helpers (inlined from mlp/pipeline.py)
# ---------------------------------------------------------------------------

def _extract_harmonization_data(df, env):
    """Extract imaging features and covariates for ComBat."""
    harm_config = env.configs.harmonize
    reg_config = env.configs.regression

    roi_columns = None
    if reg_config.get("feature_mode") == "roi":
        roi_networks = reg_config.get("roi_networks", [])
        if roi_networks:
            roi_columns = get_roi_columns_from_config(env.configs.data, roi_networks)

    imaging_cols = get_imaging_columns(
        df, reg_config.get("imaging_prefixes", []), roi_columns
    )
    X = df[imaging_cols].values

    site_col = harm_config["site_column"]
    covariate_cols = [site_col] + harm_config.get("covariates", [])
    covars = df[covariate_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})

    for col in list(covars.columns):
        if col == "SITE":
            continue
        if not pd.api.types.is_numeric_dtype(covars[col]):
            covars[col] = pd.Categorical(covars[col]).codes
    for col in list(covars.columns):
        if col == "SITE":
            continue
        if covars[col].nunique() <= 1:
            covars = covars.drop(columns=col)

    return X, covars, imaging_cols


def _fit_harmonize_scale(train_df, env, seed):
    """Fit ComBat + scaler on train set. Returns (pipeline_dict, X_train_scaled).

    Processing order depends on ICV correction method:
      - ratio:       ICV ratio → ComBat → Scale
      - residualize: ComBat → ICV residualize → Scale
    """
    harm_config = env.configs.harmonize

    X, covars, imaging_cols = _extract_harmonization_data(train_df, env)

    # Remove zero-variance features
    feature_vars = np.var(X, axis=0)
    valid_features = feature_vars > 1e-10
    X = X[:, valid_features]
    surviving_cols = [imaging_cols[i] for i, v in enumerate(valid_features) if v]

    # ICV correction config
    icv_col = env.configs.data.get("icv_column")
    icv_correction_cfg = env.configs.regression.get("icv_correction", {})
    icv_enabled = icv_correction_cfg.get("enabled", False) and icv_col and icv_col in train_df.columns
    icv_method = icv_correction_cfg.get("method", "residualize") if icv_enabled else None
    icv_fitted = None
    vol_indices = None
    thk_indices = None

    if icv_enabled:
        vol_substring = icv_correction_cfg.get("volume_substring", "__vol__")
        vol_indices = identify_volume_features(surviving_cols, vol_substring)
        thk_substring = icv_correction_cfg.get("thickness_substring", "__thk__")
        thk_indices = identify_thickness_features(surviving_cols, thk_substring)

    # --- Ratio method: ICV ratio BEFORE ComBat ---
    if icv_method == "ratio" and (vol_indices or thk_indices):
        icv_train = train_df[icv_col].values.astype(float)
        X = apply_icv_ratio_correction(X, icv_train, vol_indices or [], thk_indices)

    eb = harm_config.get("empirical_bayes", True)
    smooth_terms = harm_config.get("smooth_terms", [])
    combat_model, X_harm = harmonizationLearn(X, covars, eb=eb, smooth_terms=smooth_terms)

    # --- Residualize method: ICV residualize AFTER ComBat ---
    if icv_method == "residualize" and vol_indices:
        icv_train = train_df[icv_col].values.astype(float)
        icv_fitted = fit_icv_correction(X_harm, icv_train, vol_indices)
        X_harm = apply_icv_correction(X_harm, icv_train, icv_fitted)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_harm)

    pipeline = {
        "combat_model": combat_model,
        "scaler": scaler,
        "valid_features": valid_features,
        "surviving_cols": surviving_cols,
        "n_features": X_scaled.shape[1],
        "icv_fitted": icv_fitted,
        "icv_method": icv_method,
        "vol_indices": vol_indices,
        "thk_indices": thk_indices,
    }
    return pipeline, X_scaled


def _apply_harmonize_scale(fold_df, fitted_pipeline, env):
    """Apply ComBat + ICV correction + scaler to a fold.

    Processing order matches _fit_harmonize_scale:
      - ratio:       ICV ratio → ComBat → Scale
      - residualize: ComBat → ICV residualize → Scale
    """
    X_fold, fold_covars, _ = _extract_harmonization_data(fold_df, env)
    X_fold = X_fold[:, fitted_pipeline["valid_features"]]

    icv_method = fitted_pipeline.get("icv_method")
    vol_indices = fitted_pipeline.get("vol_indices")
    thk_indices = fitted_pipeline.get("thk_indices")

    # --- Ratio method: ICV ratio BEFORE ComBat ---
    if icv_method == "ratio" and (vol_indices or thk_indices):
        icv_col = env.configs.data.get("icv_column")
        if icv_col and icv_col in fold_df.columns:
            icv_fold = fold_df[icv_col].values.astype(float)
            X_fold = apply_icv_ratio_correction(X_fold, icv_fold, vol_indices or [], thk_indices)

    combat_model = fitted_pipeline["combat_model"]
    n_orig = len(fold_covars)

    if combat_model is not None:
        # Pad missing training sites (same fix as old mlp/pipeline.py)
        if "SITE" in fold_covars.columns:
            training_sites = list(combat_model.get("SITE_labels", []))
            present_sites = set(fold_covars["SITE"].unique())
            missing_sites = [s for s in training_sites if s not in present_sites]

            if missing_sites:
                dummy_covars = pd.concat(
                    [fold_covars.iloc[0:1].assign(SITE=s) for s in missing_sites],
                    ignore_index=True,
                )
                dummy_X = np.tile(X_fold[0:1], (len(missing_sites), 1))
                fold_covars = pd.concat([fold_covars, dummy_covars], ignore_index=True)
                X_fold_for_harm = np.vstack([X_fold, dummy_X])
            else:
                X_fold_for_harm = X_fold
        else:
            X_fold_for_harm = X_fold

        try:
            X_harm_aug = harmonizationApply(X_fold_for_harm, fold_covars, combat_model)
            X_harm = X_harm_aug[:n_orig]
        except Exception as _combat_err:
            warnings.warn(
                f"ComBat harmonizationApply failed ({type(_combat_err).__name__}: {_combat_err}). "
                "Falling back to unharmonized features for this fold.",
                RuntimeWarning,
                stacklevel=2,
            )
            X_harm = X_fold
    else:
        X_harm = X_fold

    # --- Residualize method: ICV residualize AFTER ComBat ---
    icv_fitted = fitted_pipeline.get("icv_fitted")
    if icv_fitted is not None:
        icv_col = env.configs.data.get("icv_column")
        if icv_col and icv_col in fold_df.columns:
            icv_fold = fold_df[icv_col].values.astype(float)
            X_harm = apply_icv_correction(X_harm, icv_fold, icv_fitted)

    X_scaled = fitted_pipeline["scaler"].transform(X_harm)
    return X_scaled


# ---------------------------------------------------------------------------
# Covariate residualization
# ---------------------------------------------------------------------------

def _prepare_covariates(df, covariate_cols):
    """Prepare covariate matrix from dataframe."""
    X_cov_list = []
    for col in covariate_cols:
        if col in df.columns:
            values = df[col].values
            if not pd.api.types.is_numeric_dtype(df[col]):
                values = pd.Categorical(df[col]).codes
            X_cov_list.append(values.reshape(-1, 1))
    return np.hstack(X_cov_list)


def fit_residualize(y, df, covariate_cols):
    """Fit residualization model on training data."""
    from sklearn.linear_model import LinearRegression
    X_cov = _prepare_covariates(df, covariate_cols)
    model = LinearRegression()
    model.fit(X_cov, y)
    return model


def apply_residualize(y, df, covariate_cols, model):
    """Apply fitted residualization model."""
    X_cov = _prepare_covariates(df, covariate_cols)
    return y - model.predict(X_cov)


# ---------------------------------------------------------------------------
# Sample weighting
# ---------------------------------------------------------------------------

def compute_sample_weights(y, bin_edges, method="inverse_freq"):
    """Compute sample weights for imbalanced regression."""
    n_bins = len(bin_edges) - 1
    bin_indices = np.digitize(y, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    if method == "inverse_freq":
        weights_per_bin = len(y) / (n_bins * np.maximum(bin_counts, 1))
        return weights_per_bin[bin_indices]
    else:
        raise ValueError(f"Unknown weighting method: {method}")


def apply_sample_weighting(y_train, target_name, env, method="inverse_freq"):
    """Apply sample weighting (subjects already filtered by bin range)."""
    reg_config = env.configs.regression
    bin_filter = reg_config.get("bin_filter", {})

    if target_name not in bin_filter:
        return None

    min_val, max_val = bin_filter[target_name]
    # Create equal-width bins within the filter range for weighting
    n_bins = 5
    bin_edges = np.linspace(min_val, max_val, n_bins + 1).tolist()
    return compute_sample_weights(y_train, bin_edges, method)


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_full_dataset(env) -> pd.DataFrame:
    """Load all data for nested CV (train+val+test combined)."""
    run_cfg = env.configs.run
    data_dir = (
        env.repo_root / "outputs" / run_cfg["run_name"] / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}" / "datasets"
    )
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def filter_target_data(
    df: pd.DataFrame,
    target_config: dict,
    harmonize_config: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter data for specific regression target."""
    target_col = target_config["column"]

    mask = df[target_col].notna()
    df_filtered = df[mask].copy()
    y = df_filtered[target_col].values

    # Exclude non-binary sex subjects
    if "sex_mapped" in df_filtered.columns:
        non_binary = ~df_filtered["sex_mapped"].isin(["male", "female"])
        non_binary = non_binary & df_filtered["sex_mapped"].notna()
        if non_binary.any():
            n_nb = int(non_binary.sum())
            if verbose:
                print(f"  filter_target_data: excluding {n_nb} non-binary sex subject(s)")
            df_filtered = df_filtered[~non_binary].reset_index(drop=True)
            y = df_filtered[target_col].values

    # Remove subjects with NaN in ComBat columns
    if harmonize_config is not None:
        site_col = harmonize_config.get("site_column", "mr_y_adm__info__dev_model")
        cov_cols = harmonize_config.get("covariates", [])
        check_cols = [c for c in [site_col] + cov_cols if c in df_filtered.columns]
        if check_cols:
            nan_mask = df_filtered[check_cols].isna().any(axis=1)
            if nan_mask.any():
                n_drop = int(nan_mask.sum())
                if verbose:
                    print(f"  filter_target_data: dropping {n_drop} subject(s) with NaN in ComBat columns")
                df_filtered = df_filtered[~nan_mask].reset_index(drop=True)
                y = y[~nan_mask.values]

    return df_filtered, y


# ---------------------------------------------------------------------------
# Feature name resolution
# ---------------------------------------------------------------------------

def get_feature_names(env, df, n_imaging):
    """Resolve feature names based on config."""
    reg_config = env.configs.regression
    feature_transform = reg_config.get("feature_transform", "raw")

    if reg_config.get("feature_mode") == "roi" and feature_transform != "raw":
        from .univariate import extract_bilateral_pairs
        roi_networks = reg_config.get("roi_networks", [])
        bilateral_pairs, _ = extract_bilateral_pairs(env.configs.data, roi_networks)
        if feature_transform == "asymmetry":
            names = [f"{name}_AI" for name, _, _ in bilateral_pairs]
        elif feature_transform == "ai_total":
            names = ([f"{name}_AI" for name, _, _ in bilateral_pairs]
                     + [f"{name}_total" for name, _, _ in bilateral_pairs])
        else:
            names = ([f"{name}_AI" for name, _, _ in bilateral_pairs]
                     + [f"{name}_total" for name, _, _ in bilateral_pairs])
        return sorted(names)[:n_imaging]

    if reg_config.get("feature_mode") == "roi":
        roi_networks = reg_config.get("roi_networks", [])
        roi_columns = get_roi_columns_from_config(env.configs.data, roi_networks) if roi_networks else []
        return [c for c in roi_columns if c in df.columns][:n_imaging]

    imaging_cols = get_imaging_columns(df.iloc[:10], reg_config.get("imaging_prefixes", []))
    return imaging_cols[:n_imaging]


# ---------------------------------------------------------------------------
# Single fold execution
# ---------------------------------------------------------------------------

def run_single_fold(
    env,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fold_idx: int,
    seed: int,
    target_name: str,
    model_name: str,
    residualized: bool = False,
) -> dict:
    """Process a single outer CV fold: ComBat → ICV correction → scale → train → predict."""

    reg_config = env.configs.regression

    logger.info(
        f"  Train: {len(y_train)} (mean={y_train.mean():.2f}) | "
        f"Test: {len(y_test)} (mean={y_test.mean():.2f})"
    )

    feature_mode = reg_config.get("feature_mode", "raw")

    if feature_mode == "roi":
        roi_networks = reg_config.get("roi_networks", [])
        roi_columns = get_roi_columns_from_config(env.configs.data, roi_networks) if roi_networks else []
        roi_cols_present = [c for c in roi_columns if c in train_df.columns]

        if fold_idx == 0:
            print(f"  ROI feature selection: {len(roi_cols_present)} features (networks: {roi_networks})")

        # Filter to ROI + metadata before harmonization
        # Exclude ALL imaging columns (by prefix) AND the ROI columns themselves
        # to avoid duplicate columns when ROI prefixes aren't in imaging_prefixes
        imaging_prefixes = reg_config.get("imaging_prefixes", [])
        roi_set = set(roi_cols_present)
        meta_cols = [c for c in train_df.columns
                     if not any(c.startswith(p) for p in imaging_prefixes)
                     and c not in roi_set]
        train_roi_df = train_df[meta_cols + roi_cols_present]
        test_roi_df = test_df[meta_cols + roi_cols_present]

        fitted_pipeline, X_train = _fit_harmonize_scale(train_roi_df, env, seed + fold_idx)
        X_test = _apply_harmonize_scale(test_roi_df, fitted_pipeline, env)
    else:
        fitted_pipeline, X_train = _fit_harmonize_scale(train_df, env, seed + fold_idx)
        X_test = _apply_harmonize_scale(test_df, fitted_pipeline, env)

    # Optional: transform L/R to asymmetry features
    feature_transform = reg_config.get("feature_transform", "raw")
    if feature_transform != "raw" and feature_mode == "roi":
        from .univariate import extract_bilateral_pairs, compute_asymmetry_features

        X_train_harm = fitted_pipeline["scaler"].inverse_transform(X_train)
        X_test_harm = fitted_pipeline["scaler"].inverse_transform(X_test)

        bilateral_pairs, _ = extract_bilateral_pairs(env.configs.data, roi_networks)
        surviving_cols = fitted_pipeline["surviving_cols"]
        valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs
                       if l in surviving_cols and r in surviving_cols]

        train_asym = compute_asymmetry_features(X_train_harm, surviving_cols, valid_pairs)
        test_asym = compute_asymmetry_features(X_test_harm, surviving_cols, valid_pairs)

        if feature_transform == "asymmetry":
            transform_names = sorted(k for k in train_asym if k.endswith("_AI"))
        elif feature_transform == "total":
            transform_names = sorted(k for k in train_asym if k.endswith("_total"))
        elif feature_transform == "ai_total":
            ai_names = sorted(k for k in train_asym if k.endswith("_AI"))
            tot_names = sorted(k for k in train_asym if k.endswith("_total"))
            transform_names = ai_names + tot_names
        else:
            transform_names = sorted(train_asym.keys())

        X_train = np.column_stack([train_asym[k] for k in transform_names])
        X_test = np.column_stack([test_asym[k] for k in transform_names])

        ai_scaler = StandardScaler()
        X_train = ai_scaler.fit_transform(X_train)
        X_test = ai_scaler.transform(X_test)

        if fold_idx == 0:
            print(f"  Feature transform: {feature_transform} -> {len(transform_names)} features")

    # Sample weighting
    weighting_cfg = reg_config.get("sample_weighting", {})
    if weighting_cfg.get("enabled", False):
        method = weighting_cfg.get("method", "inverse_freq")
        result = apply_sample_weighting(y_train, target_name, env, method)
        sample_weights = result
        X_train_weighted = X_train
        y_train_weighted = y_train
    else:
        sample_weights = None
        X_train_weighted = X_train
        y_train_weighted = y_train

    # Scale target for SVR
    scale_target = model_name == "svr"
    if scale_target:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_weighted.reshape(-1, 1)).ravel()
    else:
        y_scaler = None
        y_train_scaled = y_train_weighted

    # Train baseline (Ridge)
    baseline_model = create_baseline(reg_config, seed + fold_idx)
    if sample_weights is not None:
        baseline_model.fit(X_train_weighted, y_train_weighted, sample_weight=sample_weights)
    else:
        baseline_model.fit(X_train_weighted, y_train_weighted)
    baseline_pred = baseline_model.predict(X_test)
    if not residualized:
        baseline_pred = np.clip(baseline_pred, 0, None)
    baseline_metrics = compute_regression_metrics(y_test, baseline_pred)

    # Train target model
    model_fn = MODEL_REGISTRY[model_name]
    target_model = model_fn(reg_config, seed + fold_idx)
    if sample_weights is not None and model_supports_sample_weight(model_name, reg_config):
        target_model.fit(X_train_weighted, y_train_scaled, sample_weight=sample_weights)
    else:
        target_model.fit(X_train_weighted, y_train_scaled)

    target_pred = target_model.predict(X_test)
    if scale_target and y_scaler is not None:
        target_pred = y_scaler.inverse_transform(target_pred.reshape(-1, 1)).ravel()
    if not residualized:
        target_pred = np.clip(target_pred, 0, None)
    target_metrics = compute_regression_metrics(y_test, target_pred)

    return {
        "baseline": {
            "model": baseline_model,
            "metrics": baseline_metrics,
            "y_pred": baseline_pred,
            "y_test": y_test,
        },
        model_name: {
            "model": target_model,
            "metrics": target_metrics,
            "y_pred": target_pred,
            "y_test": y_test,
            "y_train": y_train,
            "X_test": X_test,
            "X_train": X_train,
            "pipeline": fitted_pipeline,
        },
    }


# ---------------------------------------------------------------------------
# Nested CV orchestrator
# ---------------------------------------------------------------------------

def run_target_with_nested_cv(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    model_name: str,
    verbose: bool = True,
):
    """Run regression for single target with nested CV."""
    target_name = target_config["name"]
    target_col = target_config["column"]
    seed = env.configs.run["seed"]
    reg_config = env.configs.regression

    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Target: {target_name} ({target_col})")
        logger.info(f"Model: {model_name.upper()}")
        logger.info(f"{'='*60}")

    # Filter data
    harm_config = env.configs.harmonize
    df_filtered, y = filter_target_data(full_df, target_config, harmonize_config=harm_config, verbose=verbose)

    # Filter by bin range (independent of sample weighting)
    bin_filter = reg_config.get("bin_filter", {})
    if target_name in bin_filter and bin_filter[target_name] is not None:
        min_val, max_val = bin_filter[target_name]
        valid_mask = (y >= min_val) & (y < max_val)
        n_excluded = (~valid_mask).sum()
        if n_excluded > 0 and verbose:
            print(f"Excluding {n_excluded} subjects outside bin range [{min_val}, {max_val})")
        df_filtered = df_filtered[valid_mask].reset_index(drop=True)
        y = y[valid_mask]

    # Residualization config
    residualized = False
    covariate_cols_for_residualize = None
    cov_cfg = reg_config.get("covariates", {})
    if cov_cfg.get("residualize", False):
        is_raw_score = target_name.endswith("_raw")
        apply_to_raw_only = cov_cfg.get("apply_to_raw_scores_only", True)
        if not apply_to_raw_only or is_raw_score:
            covariate_cols_for_residualize = cov_cfg.get("columns", ["interview_age", "sex_mapped"])
            residualized = True
            if verbose:
                print(f"Will residualize target per-fold (removing {', '.join(covariate_cols_for_residualize)} effects)")

    if verbose:
        logger.info(f"Total samples: {len(y)}")

    # CV setup
    n_splits = reg_config.get("cv", {}).get("n_outer_splits", 5)
    harm_config = env.configs.harmonize
    site_col = harm_config.get("site_column", "mr_y_adm__info__dev_model")
    feature_mode = reg_config.get("feature_mode", "raw")

    # Filter small sites
    if feature_mode in ["raw", "roi"] and site_col in df_filtered.columns:
        site_counts = df_filtered[site_col].value_counts()
        min_required = n_splits
        small_sites = site_counts[site_counts < min_required].index.tolist()
        if small_sites:
            n_excluded = df_filtered[site_col].isin(small_sites).sum()
            if verbose:
                logger.warning(f"Excluding {n_excluded} subjects from {len(small_sites)} small sites")
            valid_site_mask = ~df_filtered[site_col].isin(small_sites)
            df_filtered = df_filtered[valid_site_mask].reset_index(drop=True)
            y = y[valid_site_mask.values]

    y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")

    # Stratification key
    stratify_key = y_binned
    if feature_mode in ["raw", "roi"] and site_col in df_filtered.columns:
        site_codes = pd.Categorical(df_filtered[site_col]).codes
        n_sites = len(df_filtered[site_col].unique())
        combined_key = y_binned * n_sites + site_codes
        if pd.Series(combined_key).value_counts().min() >= n_splits:
            stratify_key = combined_key

    # Family-aware CV
    cv_cfg = reg_config.get("cv", {})
    use_family_aware = cv_cfg.get("family_aware", True)
    family_col = env.configs.data["columns"]["mapping"].get("family_id", "rel_family_id")

    family_groups = None
    if use_family_aware and family_col in df_filtered.columns:
        family_groups = pd.to_numeric(df_filtered[family_col], errors="coerce").values
        missing_mask = np.isnan(family_groups)
        if missing_mask.any():
            max_id = np.nanmax(family_groups) if (~missing_mask).any() else 0
            if np.isnan(max_id):
                max_id = 0
            unique_ids = np.arange(max_id + 1, max_id + 1 + missing_mask.sum())
            family_groups[missing_mask] = unique_ids

    if family_groups is not None:
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Storage
    baseline_folds = []
    model_folds = []
    resid_info = []

    split_args = (df_filtered, stratify_key, family_groups) if family_groups is not None else (df_filtered, stratify_key)

    # CV fold diagnostics
    if verbose:
        scanner_col = env.configs.data["columns"]["mapping"].get("scanner_model", "mr_y_adm__info__dev_model")
        has_scanner = scanner_col in df_filtered.columns
        has_family = family_groups is not None

        print(f"\n{'─'*70}")
        print(f"  CV FOLD DIAGNOSTICS (n={len(df_filtered)}, {n_splits} folds)")
        print(f"{'─'*70}")
        header = f"  {'Fold':<6} {'n_train':>8} {'n_test':>7} {'y_train':>12} {'y_test':>12}"
        if has_scanner:
            header += f" {'sites_tr':>9} {'sites_te':>9}"
        if has_family:
            header += f" {'fam_leak':>9}"
        print(header)

        _diag_splits = list(outer_cv.split(*split_args))
        for fi, (tr_idx, te_idx) in enumerate(_diag_splits):
            y_tr, y_te = y[tr_idx], y[te_idx]
            line = f"  {fi+1:<6} {len(tr_idx):>8} {len(te_idx):>7} {y_tr.mean():>6.1f}+/-{y_tr.std():>4.1f} {y_te.mean():>6.1f}+/-{y_te.std():>4.1f}"
            if has_scanner:
                n_sites_tr = df_filtered.iloc[tr_idx][scanner_col].nunique()
                n_sites_te = df_filtered.iloc[te_idx][scanner_col].nunique()
                line += f" {n_sites_tr:>9} {n_sites_te:>9}"
            if has_family:
                fam_tr = set(family_groups[tr_idx])
                fam_te = set(family_groups[te_idx])
                leaked = len(fam_tr & fam_te)
                line += f" {leaked:>9}"
            print(line)
        print(f"{'─'*70}\n")

        # Re-create splitter (consumed by diagnostics)
        if family_groups is not None:
            outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Outer CV loop
    cv_iter = outer_cv.split(*split_args)
    if verbose:
        cv_iter = tqdm(cv_iter, total=outer_cv.n_splits, desc="CV Folds")
    for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{outer_cv.n_splits}")

        train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Per-fold residualization
        if covariate_cols_for_residualize is not None:
            resid_model = fit_residualize(y_train, train_df, covariate_cols_for_residualize)
            resid_info.append({
                "fold": fold_idx + 1,
                "intercept": resid_model.intercept_,
                "coef": dict(zip(covariate_cols_for_residualize, resid_model.coef_)),
            })
            y_train = apply_residualize(y_train, train_df, covariate_cols_for_residualize, resid_model)
            y_test = apply_residualize(y_test, test_df, covariate_cols_for_residualize, resid_model)

        if verbose:
            fold_result = run_single_fold(
                env, train_df, test_df, y_train, y_test,
                fold_idx, seed, target_name, model_name,
                residualized=(covariate_cols_for_residualize is not None),
            )
        else:
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fold_result = run_single_fold(
                    env, train_df, test_df, y_train, y_test,
                    fold_idx, seed, target_name, model_name,
                    residualized=(covariate_cols_for_residualize is not None),
                )

        baseline_folds.append(fold_result["baseline"])
        fold_entry = fold_result[model_name]
        fold_entry["train_idx"] = train_idx.tolist()
        fold_entry["test_idx"] = test_idx.tolist()
        model_folds.append(fold_entry)

    # Aggregate
    baseline_agg = aggregate_cv_results(baseline_folds)
    model_agg = aggregate_cv_results(model_folds)

    # Always include concatenated predictions for plotting
    model_agg["y_true"] = np.concatenate([f["y_test"] for f in model_folds])
    model_agg["y_pred"] = np.concatenate([f["y_pred"] for f in model_folds])

    if not verbose:
        return {"baseline": baseline_agg, model_name: model_agg}

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {target_name} - {model_name.upper()}")
    logger.info(f"  Pearson r: {model_agg['overall']['pearson_r']:.3f}")
    logger.info(f"  R2: {model_agg['overall']['r2']:.3f}")
    logger.info(f"  MAE: {model_agg['overall']['mae']:.3f}")
    logger.info(f"{'='*60}\n")

    # Visualizations
    data_dir = env.repo_root / "outputs" / env.configs.run["run_name"] / env.configs.run["run_id"] / f"seed_{seed}"
    reg_dir = data_dir / "regression" / target_name / model_name
    reg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reg_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_y_true = np.concatenate([f["y_test"] for f in model_folds])
    all_y_pred = np.concatenate([f["y_pred"] for f in model_folds])

    plot_predictions(all_y_true, all_y_pred, f"{model_name.upper()} - {target_name}",
                     plots_dir / f"predictions_{model_name}_{target_name}.png",
                     residualized=residualized)
    plot_residuals(all_y_true, all_y_pred, f"{model_name.upper()} - {target_name}",
                   plots_dir / f"residuals_{model_name}_{target_name}.png")

    # Coefficients
    coefficients = None
    feature_names = None
    svr_is_linear = model_name == "svr" and reg_config.get("models", {}).get("svr", {}).get("kernel", "rbf") == "linear"
    if model_name in ["ridge"] or svr_is_linear:
        coef_shapes = [f["model"].coef_.shape for f in model_folds]
        if len(set(coef_shapes)) == 1:
            all_coefs = np.array([f["model"].coef_.ravel() for f in model_folds])
            coefficients = np.mean(all_coefs, axis=0)
        else:
            coefficients = model_folds[-1]["model"].coef_.ravel()

        n_features = len(coefficients)
        feature_names = get_feature_names(env, df_filtered, n_features)

        plot_coefficients(coefficients, feature_names,
                          f"{model_name.upper()} Coefficients - {target_name}",
                          plots_dir / f"coefficients_{model_name}_{target_name}.png", top_n=30)

    create_summary_figure(all_y_true, all_y_pred, coefficients, feature_names,
                          f"{model_name.upper()} - {target_name}",
                          plots_dir / f"summary_{model_name}_{target_name}.png")

    # Save results
    _id_col = env.configs.data["columns"]["mapping"].get("id", "participant_id")
    _subject_ids = (
        df_filtered[_id_col].tolist()
        if _id_col in df_filtered.columns
        else df_filtered.index.tolist()
    )
    results = {
        "baseline": baseline_agg,
        model_name: model_agg,
        "baseline_folds": baseline_folds,
        f"{model_name}_folds": model_folds,
        "coefficients": coefficients,
        "feature_names": feature_names,
        "residualization": resid_info if resid_info else None,
        "subject_ids": _subject_ids,
        "n_subjects": len(_subject_ids),
    }
    with open(reg_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    save_run_metadata(env, results_dir=reg_dir,
                      metrics={"pearson_r": model_agg["overall"]["pearson_r"],
                               "n_samples": model_agg["n_samples"],
                               "model_name": model_name})

    return {"baseline": baseline_agg, model_name: model_agg}


# ---------------------------------------------------------------------------
# Fast SVR on saved folds (permutation / importance)
# ---------------------------------------------------------------------------

def run_svr_on_saved_folds(
    fold_data: list,
    rng: np.random.RandomState,
    shuffle: bool = False,
    feature_idx: int | None = None,
    residualized: bool = False,
    model_cfg: dict | None = None,
) -> tuple:
    """Run SVR on pre-saved CV fold splits (fast permutation / feature importance)."""
    from sklearn.svm import SVR

    if model_cfg is None:
        model_cfg = {}
    kernel = model_cfg.get("kernel", "linear")
    C = model_cfg.get("C", 1.0)
    epsilon = model_cfg.get("epsilon", 0.1)

    all_true, all_pred = [], []
    for fold in fold_data:
        X_tr = fold["X_train"].copy()
        X_te = fold["X_test"].copy()
        y_tr = fold["y_train"].copy()
        y_te = fold["y_test"].copy()

        if shuffle:
            rng.shuffle(y_tr)
        if feature_idx is not None:
            X_te[:, feature_idx] = rng.permutation(X_te[:, feature_idx])

        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
        svr.fit(X_tr, y_tr_s)
        pred = y_scaler.inverse_transform(svr.predict(X_te).reshape(-1, 1)).ravel()
        if not residualized:
            pred = np.clip(pred, 0, None)
        all_true.extend(y_te)
        all_pred.extend(pred)

    return np.array(all_true), np.array(all_pred)


# ---------------------------------------------------------------------------
# Lateralization comparison
# ---------------------------------------------------------------------------

def run_lateralization_comparison(
    df: pd.DataFrame,
    y: np.ndarray,
    env,
    valid_feature_cols: list,
    valid_pairs: list,
    covariate_cols: list | None = None,
    residualized: bool = False,
    fold_splits: list | None = None,
) -> dict:
    """Compare SVR performance across four lateralization feature sets with per-fold ComBat."""
    from sklearn.svm import SVR
    from scipy.stats import pearsonr
    from .univariate import build_lateralization_feature_sets

    reg_config = env.configs.regression
    harm_config = env.configs.harmonize
    seed = env.configs.run.get("seed", 42)
    n_splits = reg_config.get("cv", {}).get("n_outer_splits", 5)
    site_col = harm_config.get("site_column", "mr_y_adm__info__dev_model")
    eb = harm_config.get("empirical_bayes", True)
    icv_col = env.configs.data.get("icv_column")
    icv_correction_cfg = reg_config.get("icv_correction", {})

    # Stratification
    y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")
    stratify_key = y_binned
    if site_col in df.columns:
        site_codes = pd.Categorical(df[site_col]).codes
        n_sites = len(df[site_col].unique())
        combined_key = y_binned * n_sites + site_codes
        if pd.Series(combined_key).value_counts().min() >= n_splits:
            stratify_key = combined_key

    # Family-aware CV
    family_col = env.configs.data["columns"]["mapping"].get("family_id", "rel_family_id")
    family_groups = None
    use_family_aware = reg_config.get("cv", {}).get("family_aware", True)
    if use_family_aware and family_col in df.columns:
        family_groups = pd.to_numeric(df[family_col], errors="coerce").values
        missing = np.isnan(family_groups)
        if missing.any():
            max_id = np.nanmax(family_groups) if (~missing).any() else 0
            family_groups[missing] = np.arange(max_id + 1, max_id + 1 + missing.sum())
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    set_names = ["Asymmetry only (AI)", "Total volume only", "AI + Total", "Original L/R"]
    fold_results = {name: {"all_true": [], "all_pred": []} for name in set_names}

    if fold_splits is not None:
        splits_iter = [(np.array(fs["train_idx"]), np.array(fs["test_idx"])) for fs in fold_splits]
    else:
        split_args = (df, stratify_key, family_groups) if family_groups is not None else (df, stratify_key)
        splits_iter = list(outer_cv.split(*split_args))

    print(f"  Running {len(splits_iter)}-fold CV with ComBat + ICV correction per fold ({len(set_names)} feature sets)...")

    icv_method = icv_correction_cfg.get("method", "residualize") if icv_correction_cfg.get("enabled", False) else None

    for fold_idx, (train_idx, test_idx) in enumerate(splits_iter):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        y_train = y[train_idx].copy()
        y_test = y[test_idx].copy()

        if covariate_cols:
            resid_model = fit_residualize(y_train, df.iloc[train_idx], covariate_cols)
            y_train = apply_residualize(y_train, df.iloc[train_idx], covariate_cols, resid_model)
            y_test = apply_residualize(y_test, df.iloc[test_idx], covariate_cols, resid_model)

        # Per-fold raw features
        X_train_raw = train_df[valid_feature_cols].values.astype(float)
        X_test_raw = test_df[valid_feature_cols].values.astype(float)

        # --- Ratio method: ICV ratio BEFORE ComBat ---
        vol_indices = None
        if icv_correction_cfg.get("enabled", False) and icv_col and icv_col in train_df.columns:
            vol_substring = icv_correction_cfg.get("volume_substring", "__vol__")
            vol_indices = identify_volume_features(valid_feature_cols, vol_substring)

        if icv_method == "ratio" and vol_indices:
            icv_train = train_df[icv_col].values.astype(float)
            icv_test = test_df[icv_col].values.astype(float)
            X_train_raw = apply_icv_ratio_correction(X_train_raw, icv_train, vol_indices)
            X_test_raw = apply_icv_ratio_correction(X_test_raw, icv_test, vol_indices)

        # Per-fold ComBat
        train_covars = train_df[[site_col] + harm_config.get("covariates", [])].copy()
        train_covars = train_covars.rename(columns={site_col: "SITE"})
        test_covars = test_df[[site_col] + harm_config.get("covariates", [])].copy()
        test_covars = test_covars.rename(columns={site_col: "SITE"})
        for _cov in list(train_covars.columns):
            if _cov == "SITE":
                continue
            if not pd.api.types.is_numeric_dtype(train_covars[_cov]):
                train_covars[_cov] = pd.Categorical(train_covars[_cov]).codes.astype(float)
                test_covars[_cov] = pd.Categorical(test_covars[_cov]).codes.astype(float)

        try:
            combat_model_fold, X_train_harm = harmonizationLearn(X_train_raw, train_covars, eb=eb)
            X_test_harm = harmonizationApply(X_test_raw, test_covars, combat_model_fold)
        except Exception:
            X_train_harm, X_test_harm = X_train_raw, X_test_raw

        # --- Residualize method: ICV residualize AFTER ComBat ---
        if icv_method == "residualize" and vol_indices:
            icv_train = train_df[icv_col].values.astype(float)
            icv_test = test_df[icv_col].values.astype(float)
            icv_fitted = fit_icv_correction(X_train_harm, icv_train, vol_indices)
            X_train_harm = apply_icv_correction(X_train_harm, icv_train, icv_fitted)
            X_test_harm = apply_icv_correction(X_test_harm, icv_test, icv_fitted)

        train_sets = build_lateralization_feature_sets(X_train_harm, valid_feature_cols, valid_pairs)
        test_sets = build_lateralization_feature_sets(X_test_harm, valid_feature_cols, valid_pairs)

        for set_name in set_names:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_sets[set_name])
            X_te = scaler.transform(test_sets[set_name])
            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            svr = SVR(kernel="linear", C=1.0)
            svr.fit(X_tr, y_tr_s)
            pred = y_scaler.inverse_transform(svr.predict(X_te).reshape(-1, 1)).ravel()
            if not residualized:
                pred = np.clip(pred, 0, None)
            fold_results[set_name]["all_true"].extend(y_test)
            fold_results[set_name]["all_pred"].extend(pred)

    output = {}
    for set_name in set_names:
        all_true = np.array(fold_results[set_name]["all_true"])
        all_pred = np.array(fold_results[set_name]["all_pred"])
        r, p = pearsonr(all_true, all_pred)
        output[set_name] = {"all_true": all_true, "all_pred": all_pred, "r": r, "p": p}

    return output
