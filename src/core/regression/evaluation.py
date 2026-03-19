"""Evaluation metrics and statistical tests for regression."""

import numpy as np
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

    if len(y_true) > 1:
        if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
            metrics["pearson_r"] = 0.0
            metrics["pearson_p"] = 1.0
            metrics["spearman_r"] = 0.0
            metrics["spearman_p"] = 1.0
        else:
            r, p = pearsonr(y_true, y_pred)
            metrics["pearson_r"] = float(r)
            metrics["pearson_p"] = float(p)
            rho, p_rho = spearmanr(y_true, y_pred)
            metrics["spearman_r"] = float(rho)
            metrics["spearman_p"] = float(p_rho)
    else:
        metrics["pearson_r"] = 0.0
        metrics["pearson_p"] = 1.0
        metrics["spearman_r"] = 0.0
        metrics["spearman_p"] = 1.0

    return metrics


def aggregate_cv_results(folds: list[dict]) -> dict:
    """Aggregate regression results across CV folds."""
    all_y_true = np.concatenate([fold["y_test"] for fold in folds])
    all_y_pred = np.concatenate([fold["y_pred"] for fold in folds])
    overall_metrics = compute_regression_metrics(all_y_true, all_y_pred)

    if "pearson_p" in overall_metrics:
        overall_metrics["pearson_p_parametric"] = overall_metrics.pop("pearson_p")

    per_fold_metrics = {}
    metric_names = ["r2", "mae", "mse", "rmse", "pearson_r", "spearman_r"]

    for metric_name in metric_names:
        fold_values = [fold["metrics"][metric_name] for fold in folds]
        per_fold_metrics[f"{metric_name}_mean"] = float(np.mean(fold_values))
        per_fold_metrics[f"{metric_name}_std"] = float(np.std(fold_values))
        per_fold_metrics[f"{metric_name}_min"] = float(np.min(fold_values))
        per_fold_metrics[f"{metric_name}_max"] = float(np.max(fold_values))

    return {
        "overall": overall_metrics,
        "per_fold": per_fold_metrics,
        "n_folds": len(folds),
        "n_samples": len(all_y_true),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = (
        (n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)
    ) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    return float((group1.mean() - group2.mean()) / pooled_std) if pooled_std > 0 else 0.0


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 10000,
    metrics: list[str] | None = None,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals on held-out CV predictions."""
    if metrics is None:
        metrics = ["pearson_r", "spearman_r", "r2"]

    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (1 - ci) / 2
    boot_samples: dict[str, list] = {m: [] for m in metrics}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        m = compute_regression_metrics(y_true[idx], y_pred[idx])
        for metric in metrics:
            boot_samples[metric].append(m.get(metric, np.nan))

    observed = compute_regression_metrics(y_true, y_pred)
    results = {}
    for metric in metrics:
        dist = np.array(boot_samples[metric])
        results[metric] = {
            "observed": float(observed.get(metric, np.nan)),
            "lower": float(np.nanpercentile(dist, 100 * alpha)),
            "upper": float(np.nanpercentile(dist, 100 * (1 - alpha))),
            "boot_dist": dist,
            "ci_level": ci,
            "n_bootstrap": n_bootstrap,
        }
    return results


def fisher_z_compare(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Fisher z-test comparing two independent Pearson correlations."""
    r1c = np.clip(r1, -0.9999, 0.9999)
    r2c = np.clip(r2, -0.9999, 0.9999)
    z1 = np.arctanh(r1c)
    z2 = np.arctanh(r2c)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    p = float(2 * norm.sf(abs(z)))
    return float(z), p


def compute_permutation_pvalue(observed_r: float, null_distribution: np.ndarray) -> float:
    """Two-tailed empirical p-value from a permutation null distribution."""
    n_exceed = int(np.sum(np.abs(null_distribution) >= abs(observed_r)))
    n = len(null_distribution)
    return (n_exceed + 1) / (n + 1)


def permutation_test(
    env,
    full_df,
    target_config: dict,
    model_name: str,
    n_permutations: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Pipeline-matched permutation test: shuffles labels, runs full nested CV."""
    from .pipeline import run_target_with_nested_cv

    reg_config = env.configs.regression
    if n_permutations is None:
        n_permutations = reg_config.get("permutation", {}).get("n_permutations", 1000)
    if seed is None:
        seed = env.configs.run.get("seed", 42)

    target_col = target_config["column"]
    target_name = target_config["name"]

    if verbose:
        print(f"Pipeline-matched permutation test: {target_name} / {model_name}")
        print(f"  n_permutations={n_permutations} | shuffle_seed={seed}")
        print("  Per-fold ComBat + ICV correction + residualization + family-aware CV")

    null_rs: list[float] = []
    rng = np.random.RandomState(seed)

    perm_iter: range | object = range(n_permutations)
    if verbose:
        from tqdm import tqdm
        perm_iter = tqdm(perm_iter, desc="Permutations")

    for _ in perm_iter:
        perm_df = full_df.copy()
        perm_df[target_col] = rng.permutation(perm_df[target_col].values)
        result = run_target_with_nested_cv(
            env, perm_df, target_config, model_name, verbose=False
        )
        null_rs.append(result[model_name]["overall"]["pearson_r"])

    null_arr = np.array(null_rs)
    return {
        "null_distribution": null_arr,
        "n_permutations": n_permutations,
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
    }
