"""Univariate analyses: feature correlations, asymmetry, sex differences, circuit interactions.

Updated for R6 column naming: __lh_sum/__rh_sum and __lh_wmean/__rh_wmean suffixes.
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn
from scipy.stats import pearsonr, ttest_ind
from statsmodels.stats.multitest import multipletests

from ..features import get_roi_columns_from_config
from ..preprocessing.tbv_correction import (
    apply_icv_correction,
    apply_icv_ratio_correction,
    fit_icv_correction,
    identify_thickness_features,
    identify_volume_features,
)
from .pipeline import fit_residualize, apply_residualize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bilateral pair extraction (R6 naming)
# ---------------------------------------------------------------------------

def extract_bilateral_pairs(data_config, network_names):
    """Extract L/R bilateral pairs from data.yaml network definitions.

    R6 columns end with __lh_sum/__rh_sum (volumes) or __lh_wmean/__rh_wmean (DTI).
    """
    roi_features = data_config.get("roi_features", {})
    all_features = []
    for net in network_names:
        net_def = roi_features.get(net, {})
        for feat_type in ("structural", "connectivity"):
            all_features.extend(net_def.get(feat_type) or [])
    all_features = list(dict.fromkeys(all_features))

    bilateral_pairs = []
    used = set()

    # R6 suffix patterns: __lh_sum <-> __rh_sum, __lh_wmean <-> __rh_wmean
    r6_suffix_pairs = [
        ("__lh_sum", "__rh_sum"),
        ("__lh_wmean", "__rh_wmean"),
        ("__lh_mean", "__rh_mean"),
    ]
    # Legacy R5 suffix patterns
    legacy_suffix_pairs = [("lh", "rh"), ("l", "r")]

    for f in all_features:
        if f in used:
            continue
        matched = False

        # Try R6 patterns first
        for lsuf, rsuf in r6_suffix_pairs:
            if f.endswith(lsuf):
                candidate = f[: -len(lsuf)] + rsuf
                if candidate in all_features and candidate not in used:
                    short = _make_short_name(f[: -len(lsuf)])
                    bilateral_pairs.append((short, f, candidate))
                    used.update([f, candidate])
                    matched = True
                    break
            elif f.endswith(rsuf):
                candidate = f[: -len(rsuf)] + lsuf
                if candidate in all_features and candidate not in used:
                    short = _make_short_name(f[: -len(rsuf)])
                    bilateral_pairs.append((short, candidate, f))
                    used.update([f, candidate])
                    matched = True
                    break

        # Fallback to legacy patterns
        if not matched:
            for lsuf, rsuf in legacy_suffix_pairs:
                if f.endswith(lsuf):
                    candidate = f[: -len(lsuf)] + rsuf
                    if candidate in all_features and candidate not in used:
                        short = _make_short_name(f[: -len(lsuf)])
                        bilateral_pairs.append((short, f, candidate))
                        used.update([f, candidate])
                        matched = True
                        break
                elif f.endswith(rsuf):
                    candidate = f[: -len(rsuf)] + lsuf
                    if candidate in all_features and candidate not in used:
                        short = _make_short_name(f[: -len(rsuf)])
                        bilateral_pairs.append((short, candidate, f))
                        used.update([f, candidate])
                        matched = True
                        break

    unilateral = [f for f in all_features if f not in used]
    return bilateral_pairs, unilateral


def _make_short_name(base):
    """Derive a short display name from the feature base string.

    Handles R6 abbreviations: __cd__ -> caudate, __pt__ -> putamen, etc.
    """
    # R6 abbreviation decoder
    r6_abbrev = {
        "cd": "caudate",
        "pt": "putamen",
        "pl": "pallidum",
        "vdc": "VEDC_VTA",
        "ab": "accumbens",
        "ag": "amygdala",
        "th": "thalamus",
        "hc": "hippocampus",
        "scs": "SCS",
        "sfrt": "superior_frontal",
        "rmfrt": "rostral_middle_frontal",
        "cmfrt": "caudal_middle_frontal",
        "lobfrt": "lateral_orbitofrontal",
        "mobfrt": "medial_orbitofrontal",
        "rac": "rostral_anterior_cingulate",
        "cac": "caudal_anterior_cingulate",
        "ins": "insula",
        "stmp": "superior_temporal",
        "er": "entorhinal",
    }

    # Try to find R6 abbreviation in the base string
    parts = base.split("__")
    for part in reversed(parts):
        part_clean = part.strip("_")
        if part_clean in r6_abbrev:
            short = r6_abbrev[part_clean]
            # Add modality suffix
            if "dti__is__fa" in base or "dtifa" in base:
                short += "_FA"
            elif "dti__is__md" in base or "dtimd" in base:
                short += "_MD"
            return short

    # Fallback: use last meaningful segment
    parts_simple = base.split("_")
    short = parts_simple[-1] if parts_simple[-1] else parts_simple[-2] if len(parts_simple) > 1 else base
    if "fa" in base.lower() and "dti" in base.lower():
        short += "_FA"
    elif "md" in base.lower() and "dti" in base.lower():
        short += "_MD"
    return short


# ---------------------------------------------------------------------------
# Asymmetry computation — ENIGMA convention: AI = (L - R) / (L + R)
# ---------------------------------------------------------------------------

def compute_asymmetry_features(X, feature_cols, bilateral_pairs):
    """Compute asymmetry index and total for each bilateral pair."""
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    result = {}
    for name, lcol, rcol in bilateral_pairs:
        if lcol not in col_to_idx or rcol not in col_to_idx:
            logger.warning("Skipping pair %s: columns not in feature list", name)
            continue
        li, ri = col_to_idx[lcol], col_to_idx[rcol]
        if li >= X.shape[1] or ri >= X.shape[1]:
            logger.warning(
                "Skipping pair %s: index %d/%d out of bounds (X has %d cols)",
                name, li, ri, X.shape[1],
            )
            continue
        L = X[:, li]
        R = X[:, ri]
        total = L + R
        total_safe = np.where(np.abs(total) < 1e-6, np.nan, total)
        result[f"{name}_AI"] = (L - R) / total_safe
        result[f"{name}_total"] = total
    return result


# ---------------------------------------------------------------------------
# Data preparation with full-sample ComBat + ICV correction
# ---------------------------------------------------------------------------

def prepare_harmonized_data(
    df,
    feature_cols,
    harmonize_config,
    regression_config,
    target_col,
    target_name=None,
    residualize_age_sex=True,
    data_config=None,
):
    """Full-sample ComBat + ICV correction + optional target residualization.

    Appropriate for univariate tests (no model training => no data-leakage risk).
    """
    mask = df[target_col].notna()
    df_f = df[mask].copy()
    y = df_f[target_col].values.astype(float)

    # Bin filter
    if target_name is not None:
        bin_filter = regression_config.get("bin_filter", {})
        if target_name in bin_filter and bin_filter[target_name] is not None:
            min_val, max_val = bin_filter[target_name]
            keep = (y >= min_val) & (y < max_val)
            df_f = df_f[keep].reset_index(drop=True)
            y = y[keep]

    # Residualize
    if residualize_age_sex:
        cov_cfg = regression_config.get("covariates", {})
        if cov_cfg.get("residualize", False):
            is_raw = target_name is not None and target_name.endswith("_raw")
            if not cov_cfg.get("apply_to_raw_scores_only", True) or is_raw:
                cov_cols = cov_cfg.get("columns", [])
                resid_model = fit_residualize(y, df_f, cov_cols)
                y = apply_residualize(y, df_f, cov_cols, resid_model)

    # Ensure features present
    valid_cols = [c for c in feature_cols if c in df_f.columns]
    feat_valid = df_f[valid_cols].notna().all(axis=1) & ~np.isnan(y)
    df_f = df_f[feat_valid].reset_index(drop=True)
    y = y[feat_valid.values]

    # Site filter
    site_col = harmonize_config.get("site_column", "mr_y_adm__info__dev_model")
    n_splits = regression_config.get("cv", {}).get("n_outer_splits", 5)
    if site_col in df_f.columns:
        site_counts = df_f[site_col].value_counts()
        small = site_counts[site_counts < n_splits].index.tolist()
        if small:
            keep = ~df_f[site_col].isin(small)
            df_f = df_f[keep].reset_index(drop=True)
            y = y[keep.values]

    # ComBat
    harm_cov_cols = [
        c for c in harmonize_config.get("covariates", [])
        if c in df_f.columns and df_f[c].notna().sum() > 0
    ]
    all_cov_for_check = [c for c in [site_col] + harm_cov_cols if c in df_f.columns]
    cov_nan_mask = df_f[all_cov_for_check].isna().any(axis=1)
    if cov_nan_mask.any():
        n_drop = int(cov_nan_mask.sum())
        print(f"  ComBat: dropping {n_drop} subjects with NaN in site/covariate columns")
        df_f = df_f[~cov_nan_mask].reset_index(drop=True)
        y = y[~cov_nan_mask.values]

    X_raw = df_f[valid_cols].values.astype(float)

    # ICV correction config
    icv_col = data_config.get("icv_column") if data_config is not None else None
    icv_correction_cfg = regression_config.get("icv_correction", {})
    icv_enabled = (
        icv_correction_cfg.get("enabled", False)
        and icv_col
        and icv_col in df_f.columns
    )
    icv_method = icv_correction_cfg.get("method", "residualize") if icv_enabled else None
    vol_indices = None
    thk_indices = None
    if icv_enabled:
        vol_substring = icv_correction_cfg.get("volume_substring", "__vol__")
        vol_indices = identify_volume_features(valid_cols, vol_substring)
        thk_substring = icv_correction_cfg.get("thickness_substring", "__thk__")
        thk_indices = identify_thickness_features(valid_cols, thk_substring)

    # --- Ratio method: ICV ratio BEFORE ComBat ---
    if icv_method == "ratio" and (vol_indices or thk_indices):
        icv_vals = df_f[icv_col].values.astype(float)
        X_raw = apply_icv_ratio_correction(X_raw, icv_vals, vol_indices or [], thk_indices)

    covars = df_f[[site_col] + harm_cov_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})
    for col in list(covars.columns):
        if col == "SITE":
            continue
        if not pd.api.types.is_numeric_dtype(covars[col]):
            covars[col] = pd.Categorical(covars[col]).codes.astype(float)
        else:
            covars[col] = covars[col].astype(float)
    for col in list(covars.columns):
        if col == "SITE":
            continue
        if covars[col].nunique() <= 1:
            covars = covars.drop(columns=col)

    eb = harmonize_config.get("empirical_bayes", True)
    _, X_harm = harmonizationLearn(X_raw, covars, eb=eb)

    # --- Residualize method: ICV residualize AFTER ComBat ---
    if icv_method == "residualize" and vol_indices:
        icv_vals = df_f[icv_col].values.astype(float)
        icv_fitted = fit_icv_correction(X_harm, icv_vals, vol_indices)
        X_harm = apply_icv_correction(X_harm, icv_vals, icv_fitted)

    if X_harm.shape[1] != len(valid_cols):
        logger.warning(
            "Column mismatch: X_harm has %d cols but valid_cols has %d entries",
            X_harm.shape[1], len(valid_cols),
        )
    return X_harm, y, df_f, valid_cols


# ---------------------------------------------------------------------------
# Univariate correlations
# ---------------------------------------------------------------------------

def univariate_correlations(
    X, y, feature_names, corrections=("bonferroni", "fdr_bh"), partial_covariates=None
):
    """Pearson r for each feature vs target with multiple-comparison correction."""
    rows = []
    for i, name in enumerate(feature_names):
        x_i = X[:, i] if X.ndim == 2 else X[name] if isinstance(X, dict) else X
        r, p = pearsonr(x_i, y)
        rows.append({"feature": name, "r": r, "p_raw": p, "n": len(y)})

    df = pd.DataFrame(rows)
    p_raw = df["p_raw"].values
    for method in corrections:
        reject, p_corr, _, _ = multipletests(p_raw, method=method)
        df[f"p_{method}"] = p_corr
        df[f"sig_{method}"] = reject

    df["abs_r"] = df["r"].abs()
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Volume vs Asymmetry comparison
# ---------------------------------------------------------------------------

def _steiger_z(r_ai, r_tot, r_ai_tot, n):
    """Meng, Rosenthal, Rubin (1992) Z-test for two dependent correlations."""
    from scipy.stats import norm
    r_mean = (r_ai**2 + r_tot**2) / 2
    r_ai_z = np.arctanh(r_ai)
    r_tot_z = np.arctanh(r_tot)
    f = (1 - r_ai_tot) / (2 * (1 - r_mean))
    f = np.clip(f, 0.0, 1.0)
    h = (1 - f * r_mean) / (1 - r_mean)
    z = (r_ai_z - r_tot_z) * np.sqrt((n - 3) / (2 * (1 - r_ai_tot) * h))
    p = 2 * norm.sf(abs(z))
    return z, p


def volume_vs_asymmetry_tests(X_harm, y, feature_cols, bilateral_pairs):
    """Compare r(AI, target) vs r(Total, target) for each structure."""
    asym = compute_asymmetry_features(X_harm, feature_cols, bilateral_pairs)
    n = len(y)
    rows = []
    for name, lcol, rcol in bilateral_pairs:
        ai = asym[f"{name}_AI"]
        tot = asym[f"{name}_total"]
        r_ai, p_ai = pearsonr(ai, y)
        r_tot, p_tot = pearsonr(tot, y)
        r_ai_tot, _ = pearsonr(ai, tot)
        z, p_steiger = _steiger_z(r_ai, r_tot, r_ai_tot, n)
        rows.append({
            "structure": name, "r_AI": r_ai, "p_AI": p_ai,
            "r_total": r_tot, "p_total": p_tot,
            "steiger_z": z, "p_steiger": p_steiger,
        })
    df = pd.DataFrame(rows)
    _, p_fdr, _, _ = multipletests(df["p_steiger"].values, method="fdr_bh")
    df["p_steiger_fdr"] = p_fdr
    return df


# ---------------------------------------------------------------------------
# Sex differences
# ---------------------------------------------------------------------------

def sex_differences_anova(asymmetry_data, sex_labels):
    """Independent t-test per AI feature: male vs female."""
    sex = np.asarray(sex_labels)
    is_female = (sex == "female") | (sex == 1)
    rows = []
    for name, vals in asymmetry_data.items():
        if not name.endswith("_AI"):
            continue
        male_vals = vals[~is_female]
        female_vals = vals[is_female]
        t_stat, p_val = ttest_ind(male_vals, female_vals)
        pooled_std = np.sqrt(
            ((len(male_vals) - 1) * male_vals.std(ddof=1) ** 2
             + (len(female_vals) - 1) * female_vals.std(ddof=1) ** 2)
            / (len(male_vals) + len(female_vals) - 2)
        )
        d = (male_vals.mean() - female_vals.mean()) / pooled_std if pooled_std > 0 else 0
        rows.append({
            "feature": name, "mean_male": male_vals.mean(), "mean_female": female_vals.mean(),
            "t": t_stat, "p_raw": p_val, "cohens_d": d,
            "n_male": len(male_vals), "n_female": len(female_vals),
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        _, p_fdr, _, _ = multipletests(df["p_raw"].values, method="fdr_bh")
        df["p_fdr"] = p_fdr
    return df


def sex_interaction_test(asymmetry_data, y, sex_labels):
    """OLS: y ~ AI + Sex + AI*Sex per feature."""
    import statsmodels.api as sm
    sex = np.asarray(sex_labels)
    is_female = ((sex == "female") | (sex == 1)).astype(float)
    rows = []
    for name, vals in asymmetry_data.items():
        if not name.endswith("_AI"):
            continue
        X_design = np.column_stack([vals, is_female, vals * is_female])
        X_design = sm.add_constant(X_design)
        try:
            model = sm.OLS(y, X_design).fit()
            p_interaction = model.pvalues[3]
            beta_interaction = model.params[3]
        except Exception:
            p_interaction = np.nan
            beta_interaction = np.nan
        r_male, p_male = pearsonr(vals[is_female == 0], y[is_female == 0])
        r_female, p_female = pearsonr(vals[is_female == 1], y[is_female == 1])
        rows.append({
            "feature": name, "beta_interaction": beta_interaction,
            "p_interaction": p_interaction,
            "r_male": r_male, "p_male": p_male,
            "r_female": r_female, "p_female": p_female,
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        _, p_fdr, _, _ = multipletests(df["p_interaction"].dropna().values, method="fdr_bh")
        df.loc[df["p_interaction"].notna(), "p_interaction_fdr"] = p_fdr
    return df


# ---------------------------------------------------------------------------
# Lateralization feature set builder
# ---------------------------------------------------------------------------

def build_lateralization_feature_sets(
    X_h: np.ndarray,
    valid_feature_cols: list,
    valid_pairs: list,
) -> dict:
    """Build four feature representations from a harmonized bilateral feature matrix."""
    asym = compute_asymmetry_features(X_h, valid_feature_cols, valid_pairs)
    ai_names = sorted(k for k in asym if k.endswith("_AI"))
    tot_names = sorted(k for k in asym if k.endswith("_total"))

    X_ai = np.column_stack([asym[k] for k in ai_names]) if ai_names else X_h[:, :0]
    X_tot = np.column_stack([asym[k] for k in tot_names]) if tot_names else X_h[:, :0]

    lr_cols = []
    for _, lcol, rcol in valid_pairs:
        for col in [lcol, rcol]:
            if col in valid_feature_cols:
                lr_cols.append(valid_feature_cols.index(col))
    X_lr = X_h[:, sorted(set(lr_cols))] if lr_cols else X_h

    return {
        "Asymmetry only (AI)": X_ai,
        "Total volume only": X_tot,
        "AI + Total": np.column_stack([X_ai, X_tot]) if X_ai.shape[1] and X_tot.shape[1] else X_ai,
        "Original L/R": X_lr,
    }
