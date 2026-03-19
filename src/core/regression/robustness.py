"""Robustness analyses for the dopamine-psychosis regression pipeline.

All functions accept an optional `env` to fall back on config-driven defaults
(n_perms, n_boot, seed, cutoffs). Results are returned as DataFrames or dicts
so notebooks stay thin: one function call per cell.

Updated for ABCD Release 6 column naming and simplified architecture
(no mlp/svm/tsne dependencies).
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn
from scipy.stats import pearsonr

from ..features import get_roi_columns_from_config
from .univariate import compute_asymmetry_features, extract_bilateral_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_reg_config(env=None):
    """Return regression config dict or empty dict if no env."""
    if env is not None:
        return env.configs.regression
    return {}


def _combat_harmonize(X_raw, df, harm_config, min_site_n=5):
    """Run full-sample ComBat harmonization with proper covariate handling.

    Filters for complete covariates and removes small sites before harmonization.
    Gracefully drops covariates that are entirely missing (e.g., demographics
    not collected at follow-up timepoints).

    This is a standalone helper for robustness analyses that need full-sample
    ComBat (cutoff sweep, sex stratification, network specificity null). The
    main regression pipeline uses per-fold ComBat via _fit_harmonize_scale.

    Returns:
        X_harm: harmonized feature matrix
        keep_mask: boolean mask of rows that were kept (relative to input)
        combat_model: fitted ComBat model (for apply if needed)
    """
    site_col = harm_config.get("site_column", "mr_y_adm__info__dev_model")
    cov_cols = harm_config.get("covariates", [])
    eb = harm_config.get("empirical_bayes", True)

    # Start with all rows valid
    keep = np.ones(len(df), dtype=bool)

    # Remove rows with any NaN in the feature matrix
    finite_rows = np.all(np.isfinite(X_raw), axis=1)
    if not np.all(finite_rows):
        n_nan = (~finite_rows).sum()
        logger.info(f"ComBat: dropping {n_nan} rows with NaN/Inf features")
        keep &= finite_rows

    # Only use covariates that actually have data
    available_cov_cols = []
    for col in cov_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            available_cov_cols.append(col)
            keep &= df[col].notna().values

    # Site column is required
    if site_col in df.columns:
        keep &= df[site_col].notna().values
    else:
        return X_raw, np.ones(len(df), dtype=bool), None

    if keep.sum() < 30:
        return X_raw[keep], keep, None

    df_f = df[keep].reset_index(drop=True)
    X_f = X_raw[keep]

    # Remove small sites
    site_counts = df_f[site_col].value_counts()
    small_sites = site_counts[site_counts < min_site_n].index.tolist()
    if small_sites:
        site_keep = ~df_f[site_col].isin(small_sites)
        df_f = df_f[site_keep].reset_index(drop=True)
        X_f = X_f[site_keep.values]
        keep_indices = np.where(keep)[0]
        keep[keep_indices[~site_keep.values]] = False

    if len(df_f) < 30:
        return X_f, keep, None

    covars = df_f[[site_col] + available_cov_cols].copy()
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

    if available_cov_cols != cov_cols:
        dropped = set(cov_cols) - set(available_cov_cols)
        print(f"  ComBat: dropped unavailable covariates {dropped}")

    try:
        combat_model, X_harm = harmonizationLearn(X_f, covars, eb=eb)
    except Exception as e:
        logger.warning(f"ComBat failed: {e}")
        print(f"  ComBat failed: {e}")
        return X_f, keep, None

    if np.any(np.isnan(X_harm)):
        nan_count = np.isnan(X_harm).sum()
        logger.warning(f"ComBat returned {nan_count} NaN values, using raw data")
        return X_f, keep, None

    return X_harm, keep, combat_model


def _pallidum_ai_r(X_harm, y, feature_cols, bilateral_pairs):
    """Quick helper: compute pallidum_AI and correlate with y."""
    asym = compute_asymmetry_features(X_harm, feature_cols, bilateral_pairs)
    ai = asym.get("pallidum_AI")
    if ai is None:
        return np.nan, np.nan
    valid = np.isfinite(ai) & np.isfinite(y)
    if valid.sum() < 10:
        return np.nan, np.nan
    return pearsonr(ai[valid], y[valid])


def _get_family_col(env):
    """Get family_id column name from config or default."""
    if env is not None:
        return env.configs.data.get("columns", {}).get("mapping", {}).get(
            "family_id", "rel_family_id"
        )
    return "rel_family_id"


# ---------------------------------------------------------------------------
# 1. Cutoff sensitivity
# ---------------------------------------------------------------------------

def cutoff_sensitivity(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    target_name: str | None = None,
    cutoffs=None,
    min_n: int = 50,
    env=None,
) -> pd.DataFrame:
    """Sweep minimum PQ-BC severity threshold; compute pallidum AI univariate r.

    Returns:
        DataFrame: cutoff, r, p, n.
    """
    if cutoffs is None:
        cutoffs = list(range(0, 65, 5))

    harm_config = env.configs.harmonize if env else {}

    rows = []
    for cutoff in cutoffs:
        mask = full_df[target_col].notna() & (full_df[target_col] >= cutoff)
        df_cut = full_df[mask].copy()
        y_cut = df_cut[target_col].values.astype(float)

        if len(df_cut) < min_n:
            rows.append({"cutoff": cutoff, "r": np.nan, "p": np.nan, "n": len(df_cut)})
            continue

        try:
            present_cols = [c for c in feature_cols if c in df_cut.columns]
            X_raw = df_cut[present_cols].values.astype(float)
            X_harm, keep, _ = _combat_harmonize(X_raw, df_cut, harm_config,
                                                  min_site_n=5)
            y_h = y_cut[keep]

            if keep.sum() < min_n:
                rows.append({"cutoff": cutoff, "r": np.nan, "p": np.nan,
                              "n": int(keep.sum())})
                continue

            r, p = _pallidum_ai_r(X_harm, y_h, present_cols, bilateral_pairs)
            rows.append({"cutoff": cutoff, "r": r, "p": p, "n": int(keep.sum())})
        except Exception as e:
            logger.warning(f"cutoff_sensitivity: cutoff={cutoff} failed: {e}")
            rows.append({"cutoff": cutoff, "r": np.nan, "p": np.nan, "n": np.nan})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Split-half replication
# ---------------------------------------------------------------------------

def split_half_replication(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    n_iterations: int = 100,
) -> pd.DataFrame:
    """Run nested CV SVR on 100 random 50/50 stratified splits (family-aware).

    Returns:
        DataFrame: iteration, r_first_half, r_second_half.
    """
    from .pipeline import run_target_with_nested_cv

    seed = env.configs.run.get("seed", 42)
    rng = np.random.RandomState(seed)
    family_col = _get_family_col(env)

    target_col = target_config["column"]
    mask = full_df[target_col].notna()
    df_base = full_df[mask].copy()

    rows = []
    for i in range(n_iterations):
        if family_col in df_base.columns:
            unique_fam = df_base[family_col].unique()
            rng.shuffle(unique_fam)
            half = len(unique_fam) // 2
            fam_first = set(unique_fam[:half])
            mask_first = df_base[family_col].isin(fam_first)
        else:
            n = len(df_base)
            idx = rng.permutation(n)
            mask_first = pd.Series([False] * n)
            mask_first.iloc[idx[:n // 2]] = True

        df_a = df_base[mask_first].reset_index(drop=True)
        df_b = df_base[~mask_first].reset_index(drop=True)

        r_a, r_b = np.nan, np.nan
        for half_df, label in [(df_a, "a"), (df_b, "b")]:
            try:
                res = run_target_with_nested_cv(env, half_df, target_config,
                                                 model_name="svr", verbose=False)
                val = res["svr"]["overall"]["pearson_r"]
                if label == "a":
                    r_a = val
                else:
                    r_b = val
            except Exception as e:
                logger.warning(f"split_half iter={i} half={label} failed: {e}")

        rows.append({"iteration": i, "r_first_half": r_a, "r_second_half": r_b})
        if (i + 1) % 10 == 0:
            print(f"  Split-half: {i+1}/{n_iterations} done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Leave-one-feature-out (LOFO)
# ---------------------------------------------------------------------------

def leave_one_feature_out(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
) -> pd.DataFrame:
    """Drop each feature one at a time, rerun nested CV SVR, compute delta r.

    Returns:
        DataFrame: feature_dropped, r_without, delta_r (= r_baseline - r_without).
    """
    from copy import deepcopy
    from .pipeline import run_target_with_nested_cv

    print("  LOFO: running baseline...")
    try:
        res_base = run_target_with_nested_cv(env, full_df, target_config,
                                              model_name="svr", verbose=False)
        r_baseline = res_base["svr"]["overall"]["pearson_r"]
    except Exception as e:
        logger.error(f"LOFO baseline failed: {e}")
        return pd.DataFrame()

    print(f"  LOFO baseline r={r_baseline:.3f}")

    reg_config = env.configs.regression
    roi_networks = reg_config.get("roi_networks", [])
    feature_cols = get_roi_columns_from_config(env.configs.data, roi_networks)
    feature_cols = [c for c in feature_cols if c in full_df.columns]

    rows = []
    for i, drop_feat in enumerate(feature_cols):
        env_copy = deepcopy(env)
        df_drop = full_df.drop(columns=[drop_feat], errors="ignore")

        try:
            res = run_target_with_nested_cv(env_copy, df_drop, target_config,
                                             model_name="svr", verbose=False)
            r_without = res["svr"]["overall"]["pearson_r"]
        except Exception as e:
            logger.warning(f"LOFO drop={drop_feat} failed: {e}")
            r_without = np.nan

        delta_r = r_baseline - r_without
        rows.append({"feature_dropped": drop_feat, "r_without": r_without,
                     "delta_r": delta_r})
        if (i + 1) % 5 == 0:
            print(f"  LOFO: {i+1}/{len(feature_cols)} done")

    return pd.DataFrame(rows).sort_values("delta_r", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Network specificity null
# ---------------------------------------------------------------------------

def network_specificity_null(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    target_name: str | None = None,
    n_perms: int = 1000,
    n_svr_perms: int = 0,
    seed: int = 42,
    env=None,
) -> dict:
    """Compare dopamine network against other defined networks (SVR) and a random null.

    Three-part analysis:
      1. Named networks: run full ComBat-per-fold SVR for each network defined in
         data.yaml roi_features.
      2. Fast univariate null: pre-harmonize a pool of ALL imaging features once, then
         for each permutation randomly sample bilateral pairs and compute the best
         AI correlation.
      3. SVR null (optional, n_svr_perms > 0): same random bilateral pair sampling
         but runs full ComBat-per-fold SVR for each permutation.

    Returns:
        dict with keys: named_df, null_df, svr_null_df, dopa_pct, dopa_svr_pct.
    """
    from .pipeline import run_target_with_nested_cv

    reg_config = _get_reg_config(env)
    harm_config = env.configs.harmonize if env else {}
    rng = np.random.RandomState(seed)

    # ── 1. Named network SVR comparison ─────────────────────────────────────
    roi_features_cfg = env.configs.data.get("roi_features", {})
    all_defined_networks = [k for k, v in roi_features_cfg.items() if isinstance(v, dict)]
    ordered_networks = ["dopamine_core"] + [n for n in all_defined_networks if n != "dopamine_core"]

    target_config = {"name": target_name or target_col, "column": target_col}
    reg_orig = env.configs.regression

    print("Named network SVR comparison:")
    named_rows = []
    dopa_r = np.nan
    for net_name in ordered_networks:
        net_feat_cols = get_roi_columns_from_config(env.configs.data, [net_name])
        net_feat_cols = [c for c in net_feat_cols if c in full_df.columns]
        if len(net_feat_cols) < 2:
            print(f"  {net_name}: skipped (no features in df)")
            continue
        reg_override = copy.deepcopy(dict(reg_orig))
        reg_override["roi_networks"] = [net_name]
        env.configs.regression = reg_override
        try:
            result = run_target_with_nested_cv(
                env, full_df, target_config, model_name="svr", verbose=False
            )
            r = result["svr"]["overall"]["pearson_r"]
            n = result["svr"]["n_samples"]
        finally:
            env.configs.regression = reg_orig
        print(f"  {net_name:<20} r = {r:+.4f}  (n={n})")
        named_rows.append({"network": net_name, "r": r, "n": n})
        if net_name == "dopamine_core":
            dopa_r = r
    named_df = pd.DataFrame(named_rows)

    # ── 2. Random null pool: ALL imaging modalities ─────────────────────────
    # R6 prefix patterns for all imaging features
    dopa_set = set(feature_cols)
    imaging_cfg = env.configs.data.get("imaging", {})
    all_imaging_prefixes = []
    for family_name, family_cfg in imaging_cfg.items():
        all_imaging_prefixes.extend(family_cfg.get("prefixes", []))

    pool_cols = [
        c for c in full_df.columns
        if any(c.startswith(p) for p in all_imaging_prefixes)
        and c not in dopa_set
        and full_df[c].notna().any()
    ]

    mask_base = full_df[target_col].notna()
    df_base = full_df[mask_base].copy().reset_index(drop=True)
    y_base = df_base[target_col].values.astype(float)

    pool_present = [c for c in pool_cols if c in df_base.columns]
    print(f"\nPre-harmonizing null pool ({len(pool_present)} features) once...")
    X_pool_raw = df_base[pool_present].values.astype(float)
    X_pool_h, keep_pool, _ = _combat_harmonize(X_pool_raw, df_base, harm_config, min_site_n=5)
    y_pool = y_base[keep_pool]
    n_pool_feats = X_pool_h.shape[1]
    n_dopa = len(feature_cols)
    print(f"  Pool: {n_pool_feats} harmonized features, {len(y_pool)} subjects")

    if n_pool_feats < n_dopa:
        print(f"  Warning: pool ({n_pool_feats}) smaller than dopamine set ({n_dopa})")
        n_sample = n_pool_feats
    else:
        n_sample = n_dopa

    # Find bilateral pairs in pool (R6 naming)
    pool_pair_indices = []
    used_idx = set()
    r6_suffix_pairs = [
        ("__lh_sum", "__rh_sum"),
        ("__lh_wmean", "__rh_wmean"),
        ("__lh_mean", "__rh_mean"),
    ]
    for i, c in enumerate(pool_present):
        if i in used_idx:
            continue
        for lsuf, rsuf in r6_suffix_pairs:
            if c.endswith(lsuf):
                cand = c[:-len(lsuf)] + rsuf
                if cand in pool_present:
                    j = pool_present.index(cand)
                    if j not in used_idx:
                        name = c[:-len(lsuf)].rstrip("_")
                        pool_pair_indices.append((name, i, j))
                        used_idx.update([i, j])
                        break
        else:
            # Legacy fallback (lh/rh suffixes)
            for lsuf, rsuf in [("lh", "rh"), ("l", "r")]:
                if c.endswith(lsuf):
                    cand = c[:-len(lsuf)] + rsuf
                    if cand in pool_present:
                        j = pool_present.index(cand)
                        if j not in used_idx:
                            name = c[:-len(lsuf)].rstrip("_")
                            pool_pair_indices.append((name, i, j))
                            used_idx.update([i, j])
                            break

    print(f"  Bilateral pairs in pool: {len(pool_pair_indices)}")

    null_rows = []
    for perm in range(n_perms):
        n_pairs_dopa = len(bilateral_pairs)
        if len(pool_pair_indices) < n_pairs_dopa:
            sampled_pairs = pool_pair_indices
        else:
            chosen = rng.choice(len(pool_pair_indices), size=n_pairs_dopa, replace=False)
            sampled_pairs = [pool_pair_indices[k] for k in chosen]

        if not sampled_pairs:
            null_rows.append({"perm": perm, "best_r": np.nan})
            continue

        best_r = 0.0
        for name, li, ri in sampled_pairs:
            L = X_pool_h[:, li]; R = X_pool_h[:, ri]
            total = L + R
            ai = np.where(np.abs(total) < 1e-10, 0.0, (L - R) / total)
            valid = np.isfinite(ai) & np.isfinite(y_pool)
            if valid.sum() > 10:
                r, _ = pearsonr(ai[valid], y_pool[valid])
                if abs(r) > abs(best_r):
                    best_r = r
        null_rows.append({"perm": perm, "best_r": best_r})

        if (perm + 1) % 200 == 0:
            print(f"  Random null: {perm+1}/{n_perms}")

    null_df = pd.DataFrame(null_rows)
    null_valid = null_df["best_r"].dropna().values
    dopa_pct = float((np.abs(null_valid) < abs(dopa_r)).mean() * 100) if len(null_valid) else np.nan

    if len(null_valid):
        named_df["p_vs_null"] = named_df["r"].apply(
            lambda r: float((np.abs(null_valid) >= abs(r)).mean())
        )

    # ── 3. SVR null (optional) ────────────────────────────────────────────────
    svr_null_df = pd.DataFrame(columns=["perm", "r"])
    dopa_svr_pct = np.nan

    if n_svr_perms > 0 and pool_pair_indices:
        print(f"\nRunning SVR null ({n_svr_perms} perms)...")
        # Pool prefix roots for feature detection
        pool_prefix_roots = list({p.rstrip("_") for p in all_imaging_prefixes})
        meta_cols_base = [c for c in full_df.columns
                          if not any(c.startswith(p) for p in all_imaging_prefixes)]

        n_pairs_dopa = len(bilateral_pairs)
        svr_null_rows = []
        for perm in range(n_svr_perms):
            if len(pool_pair_indices) < n_pairs_dopa:
                sampled_pairs = pool_pair_indices
            else:
                chosen = rng.choice(len(pool_pair_indices), size=n_pairs_dopa, replace=False)
                sampled_pairs = [pool_pair_indices[k] for k in chosen]

            rand_feat_cols = []
            for name, li, ri in sampled_pairs:
                rand_feat_cols.extend([pool_present[li], pool_present[ri]])

            null_df_rand = full_df[meta_cols_base + rand_feat_cols].copy()

            # Override reg config for random feature set
            reg_override = copy.deepcopy(dict(reg_orig))
            reg_override["feature_mode"] = "raw"
            reg_override["imaging_prefixes"] = pool_prefix_roots
            env.configs.regression = reg_override
            try:
                res = run_target_with_nested_cv(
                    env, null_df_rand, target_config, model_name="svr", verbose=False
                )
                svr_null_rows.append({"perm": perm, "r": res["svr"]["overall"]["pearson_r"]})
            except Exception as exc:
                logger.warning(f"SVR null perm {perm} failed: {exc}")
                svr_null_rows.append({"perm": perm, "r": np.nan})
            finally:
                env.configs.regression = reg_orig

            if (perm + 1) % 10 == 0:
                print(f"  SVR null: {perm+1}/{n_svr_perms}")

        svr_null_df = pd.DataFrame(svr_null_rows)
        svr_valid = svr_null_df["r"].dropna().values
        if len(svr_valid):
            dopa_svr_pct = float((np.abs(svr_valid) < abs(dopa_r)).mean() * 100)
            named_df["p_vs_svr_null"] = named_df["r"].apply(
                lambda r: float((np.abs(svr_valid) >= abs(r)).mean())
            )
            print(f"  Dopamine at {dopa_svr_pct:.1f}th percentile of SVR null")

    return {
        "named_df": named_df,
        "null_df": null_df,
        "svr_null_df": svr_null_df,
        "dopa_pct": dopa_pct,
        "dopa_svr_pct": dopa_svr_pct,
    }


# ---------------------------------------------------------------------------
# 5. Sex-stratified analysis
# ---------------------------------------------------------------------------

def sex_stratified_analysis(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    env,
    min_n: int = 30,
) -> pd.DataFrame:
    """ComBat full sample, then split by sex, correlate pallidum AI per sex.

    Full-sample ComBat BEFORE sex-split avoids unstable harmonization in small
    per-sex subsets. Only the correlation step is sex-stratified.

    Returns:
        DataFrame: sex, feature, r, p, n per AI feature.
    """
    harm_config = env.configs.harmonize

    mask = full_df[target_col].notna()
    df_f = full_df[mask].copy()
    y = df_f[target_col].values.astype(float)

    present_cols = [c for c in feature_cols if c in df_f.columns]
    X_raw = df_f[present_cols].values.astype(float)
    X_harm, keep, _ = _combat_harmonize(X_raw, df_f, harm_config)
    df_f = df_f[keep].reset_index(drop=True)
    y = y[keep]

    if "sex_mapped" not in df_f.columns:
        logger.warning("sex_mapped column not found; cannot stratify by sex.")
        return pd.DataFrame()

    asym = compute_asymmetry_features(X_harm, present_cols, bilateral_pairs)
    rows = []
    for sex_label in df_f["sex_mapped"].unique():
        mask_sex = (df_f["sex_mapped"] == sex_label).values
        if mask_sex.sum() < min_n:
            continue
        y_sex = y[mask_sex]
        for feat_name, ai in asym.items():
            if not feat_name.endswith("_AI"):
                continue
            ai_sex = ai[mask_sex]
            valid = np.isfinite(ai_sex) & np.isfinite(y_sex)
            if valid.sum() < min_n:
                continue
            r, p = pearsonr(ai_sex[valid], y_sex[valid])
            rows.append({"sex": sex_label, "feature": feat_name,
                         "r": r, "p": p, "n": int(valid.sum())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5b. Sex-stratified multivariate SVR
# ---------------------------------------------------------------------------

def sex_stratified_svr(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    model_name: str = "svr",
) -> dict:
    """Run nested CV SVR separately for male and female subjects.

    For each sex:
      - Filter full_df to that sex
      - Deep-copy env, remove sex_mapped from ComBat covariates and
        residualization covariates (constant within group)
      - Run run_target_with_nested_cv

    Returns:
        dict with keys: male_r, female_r, male_n, female_n,
        fisher_z, fisher_p, per_sex (dict of full results).
    """
    from .pipeline import run_target_with_nested_cv
    from .evaluation import fisher_z_compare

    if "sex_mapped" not in full_df.columns:
        logger.warning("sex_mapped column not found; cannot run sex-stratified SVR.")
        return {}

    per_sex = {}
    for sex_label in ["male", "female"]:
        sex_df = full_df[full_df["sex_mapped"] == sex_label].reset_index(drop=True)
        if len(sex_df) < 30:
            logger.warning(f"Too few {sex_label} subjects ({len(sex_df)}), skipping.")
            continue

        env_copy = copy.deepcopy(env)

        # Remove sex_mapped from ComBat covariates (constant within sex group)
        harm_covs = list(env_copy.configs.harmonize.get("covariates", []))
        if "sex_mapped" in harm_covs:
            harm_covs.remove("sex_mapped")
            env_copy.configs.harmonize["covariates"] = harm_covs

        # Remove sex_mapped from residualization covariates
        cov_cfg = env_copy.configs.regression.get("covariates", {})
        if "columns" in cov_cfg:
            resid_cols = list(cov_cfg["columns"])
            if "sex_mapped" in resid_cols:
                resid_cols.remove("sex_mapped")
                cov_cfg["columns"] = resid_cols

        try:
            result = run_target_with_nested_cv(
                env_copy, sex_df, target_config,
                model_name=model_name, verbose=False,
            )
            r = result[model_name]["overall"]["pearson_r"]
            n = result[model_name]["n_samples"]
            per_sex[sex_label] = {"r": r, "n": n, "result": result}
            print(f"  {sex_label}: r = {r:+.4f}, n = {n}")
        except Exception as e:
            logger.warning(f"Sex-stratified SVR ({sex_label}) failed: {e}")
            print(f"  {sex_label}: FAILED ({e})")
            per_sex[sex_label] = {"r": np.nan, "n": 0, "result": None}

    # Fisher z comparison
    male_r = per_sex.get("male", {}).get("r", np.nan)
    female_r = per_sex.get("female", {}).get("r", np.nan)
    male_n = per_sex.get("male", {}).get("n", 0)
    female_n = per_sex.get("female", {}).get("n", 0)

    fisher_z, fisher_p = np.nan, np.nan
    if np.isfinite(male_r) and np.isfinite(female_r) and male_n > 3 and female_n > 3:
        fisher_z, fisher_p = fisher_z_compare(male_r, male_n, female_r, female_n)
        print(f"  Fisher z = {fisher_z:.3f}, p = {fisher_p:.4f}")

    return {
        "male_r": male_r,
        "female_r": female_r,
        "male_n": male_n,
        "female_n": female_n,
        "fisher_z": fisher_z,
        "fisher_p": fisher_p,
        "per_sex": per_sex,
    }


# ---------------------------------------------------------------------------
# 5c. Per-region SVR
# ---------------------------------------------------------------------------

def per_region_svr(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    model_name: str = "svr",
) -> pd.DataFrame:
    """Run SVR for each bilateral region pair individually.

    For each bilateral pair (L, R): deep-copy env, override roi_features to
    contain only that pair's 2 columns, run run_target_with_nested_cv.

    Returns:
        DataFrame sorted by |r|: region, r, p_raw, p_fdr, n.
    """
    from statsmodels.stats.multitest import multipletests
    from .pipeline import run_target_with_nested_cv

    reg_config = env.configs.regression
    roi_networks = reg_config.get("roi_networks", [])
    net_name = roi_networks[0] if roi_networks else "dopamine_core"

    # Handle delta-suffixed columns: strip _delta before pair extraction, re-add after
    net_def = env.configs.data.get("roi_features", {}).get(net_name, {})
    all_structural = net_def.get("structural") or []
    is_delta = bool(all_structural) and all(c.endswith("_delta") for c in all_structural)

    if is_delta:
        env_for_pairs = copy.deepcopy(env)
        stripped = [c[:-6] for c in all_structural]  # remove "_delta"
        env_for_pairs.configs.data["roi_features"][net_name]["structural"] = stripped
        bilateral_pairs, _ = extract_bilateral_pairs(env_for_pairs.configs.data, roi_networks)
        strip_to_delta = {c[:-6]: c for c in all_structural}
        bilateral_pairs = [
            (name, strip_to_delta.get(lcol, lcol + "_delta"), strip_to_delta.get(rcol, rcol + "_delta"))
            for name, lcol, rcol in bilateral_pairs
        ]
    else:
        bilateral_pairs, _ = extract_bilateral_pairs(env.configs.data, roi_networks)

    rows = []
    for i, (region_name, lcol, rcol) in enumerate(bilateral_pairs):
        env_copy = copy.deepcopy(env)

        # Override roi_features to contain only this bilateral pair
        net_name = roi_networks[0] if roi_networks else "dopamine_core"
        # Determine feature type based on column suffix
        if "__vol__" in lcol or "__thk__" in lcol or "__area__" in lcol:
            env_copy.configs.data["roi_features"][net_name] = {
                "structural": [lcol, rcol],
                "connectivity": [],
            }
        else:
            env_copy.configs.data["roi_features"][net_name] = {
                "structural": [],
                "connectivity": [lcol, rcol],
            }

        try:
            result = run_target_with_nested_cv(
                env_copy, full_df, target_config,
                model_name=model_name, verbose=False,
            )
            r = result[model_name]["overall"]["pearson_r"]
            p = result[model_name]["overall"].get("pearson_p_parametric", np.nan)
            n = result[model_name]["n_samples"]
        except Exception as e:
            logger.warning(f"per_region_svr: {region_name} failed: {e}")
            r, p, n = np.nan, np.nan, 0

        rows.append({"region": region_name, "r": r, "p_raw": p, "n": n})
        print(f"  {region_name:<30} r = {r:+.4f}, p = {p:.4f}, n = {n}")

    if not rows:
        logger.warning("per_region_svr: no bilateral pairs found — returning empty DataFrame")
        return pd.DataFrame(columns=["region", "r", "p_raw", "p_fdr", "n"])

    df_out = pd.DataFrame(rows)

    # FDR correction across regions
    valid_p = df_out["p_raw"].notna()
    if valid_p.sum() > 1:
        _, p_fdr, _, _ = multipletests(df_out.loc[valid_p, "p_raw"].values, method="fdr_bh")
        df_out.loc[valid_p, "p_fdr"] = p_fdr
    else:
        df_out["p_fdr"] = df_out["p_raw"]

    df_out["abs_r"] = df_out["r"].abs()
    df_out = df_out.sort_values("abs_r", ascending=False).drop(columns="abs_r").reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# 6. Scanner-stratified analysis
# ---------------------------------------------------------------------------

def scanner_stratified_analysis(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    env,
    min_per_scanner: int = 20,
) -> pd.DataFrame:
    """Correlate pallidum AI with target separately per scanner model (no ComBat).

    No ComBat is applied since each scanner group is a single site.

    Returns:
        DataFrame: scanner, r_pallidum_AI, p, n.
    """
    scanner_col = env.configs.data["columns"]["mapping"].get(
        "scanner_model", "mr_y_adm__info__dev_model"
    )
    if scanner_col not in full_df.columns:
        logger.warning(f"Scanner column '{scanner_col}' not found.")
        return pd.DataFrame()

    mask = full_df[target_col].notna()
    df_f = full_df[mask].copy()
    y_all = df_f[target_col].values.astype(float)
    present_cols = [c for c in feature_cols if c in df_f.columns]

    rows = []
    for scanner in df_f[scanner_col].unique():
        sc_mask = (df_f[scanner_col] == scanner).values
        if sc_mask.sum() < min_per_scanner:
            continue

        X_sc = df_f.loc[sc_mask, present_cols].values.astype(float)
        y_sc = y_all[sc_mask]
        valid = np.all(np.isfinite(X_sc), axis=1) & np.isfinite(y_sc)
        X_sc = X_sc[valid]
        y_sc = y_sc[valid]
        if len(y_sc) < min_per_scanner:
            continue

        asym = compute_asymmetry_features(X_sc, present_cols, bilateral_pairs)
        ai = asym.get("pallidum_AI")
        if ai is None or np.isfinite(ai).sum() < 10:
            continue
        r, p = pearsonr(ai[np.isfinite(ai)], y_sc[np.isfinite(ai)])
        rows.append({"scanner": scanner, "r_pallidum_AI": r, "p": p,
                     "n": int(valid.sum())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Bootstrap feature CIs
# ---------------------------------------------------------------------------

def bootstrap_feature_ci(
    X_harm: np.ndarray,
    y: np.ndarray,
    bilateral_pairs: list,
    valid_cols: list,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CIs on AI feature correlations with target.

    Returns:
        DataFrame: feature, r_obs, ci_lo, ci_hi, p_boot, p_boot_fdr, n.
    """
    from statsmodels.stats.multitest import multipletests

    rng = np.random.RandomState(seed)
    asym = compute_asymmetry_features(X_harm, valid_cols, bilateral_pairs)
    ai_keys = [k for k in asym if k.endswith("_AI")]
    rows = []

    for key in ai_keys:
        ai = asym[key]
        valid = np.isfinite(ai) & np.isfinite(y)
        ai_v, y_v = ai[valid], y[valid]
        if len(ai_v) < 10:
            continue

        r_obs, _ = pearsonr(ai_v, y_v)
        boot_rs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(len(ai_v), size=len(ai_v), replace=True)
            boot_rs[b], _ = pearsonr(ai_v[idx], y_v[idx])

        ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
        p_boot = (boot_rs <= 0).mean() if r_obs > 0 else (boot_rs >= 0).mean()
        rows.append({"feature": key, "r_obs": r_obs, "ci_lo": ci_lo,
                     "ci_hi": ci_hi, "p_boot": p_boot, "n": int(valid.sum())})

    df = pd.DataFrame(rows).sort_values("r_obs").reset_index(drop=True)

    # FDR correction across features
    if len(df) > 1:
        _, p_fdr, _, _ = multipletests(df["p_boot"].values, method="fdr_bh")
        df["p_boot_fdr"] = p_fdr
    elif len(df) == 1:
        df["p_boot_fdr"] = df["p_boot"]

    return df


# ---------------------------------------------------------------------------
# 8. One-per-family permutation
# ---------------------------------------------------------------------------

def one_per_family_permutation(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    n_perms: int = 500,
) -> dict:
    """Permutation test after randomly removing one sibling per family.

    Runs n_perms iterations: each time randomly pick one subject per family,
    fit the full nested CV SVR pipeline, collect r.

    Returns:
        dict: observed_r, boot_rs, ci_lo, ci_hi, n_valid.
    """
    from .pipeline import run_target_with_nested_cv

    seed = env.configs.run.get("seed", 42)
    rng = np.random.RandomState(seed)
    family_col = _get_family_col(env)

    boot_rs = []
    for i in range(n_perms):
        if family_col in full_df.columns:
            keep_idx = (
                full_df.groupby(family_col)
                .apply(lambda g: g.sample(1, random_state=int(rng.randint(1e8))), include_groups=False)
                .index.get_level_values(1)
            )
            df_opf = full_df.loc[keep_idx].reset_index(drop=True)
        else:
            df_opf = full_df.copy()

        try:
            res = run_target_with_nested_cv(env, df_opf, target_config,
                                             model_name="svr", verbose=False)
            boot_rs.append(res["svr"]["overall"]["pearson_r"])
        except Exception as e:
            logger.warning(f"one_per_family perm={i} failed: {e}")
            boot_rs.append(np.nan)

        if (i + 1) % 50 == 0:
            print(f"  One-per-family: {i+1}/{n_perms}")

    boot_rs = np.array(boot_rs)
    valid_rs = boot_rs[np.isfinite(boot_rs)]
    obs_r = np.nanmedian(boot_rs)
    ci_lo, ci_hi = (np.percentile(valid_rs, [2.5, 97.5])
                    if len(valid_rs) > 10 else (np.nan, np.nan))

    return {
        "observed_r": obs_r,
        "boot_rs": boot_rs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_valid": len(valid_rs),
    }
