"""Model definitions for regression (SVR + Ridge only)."""

from sklearn.linear_model import Ridge
from sklearn.svm import SVR


_DEFAULT_SAMPLE_WEIGHT_SUPPORT: dict[str, bool] = {
    "ridge": True,
    "svr": True,
}


def model_supports_sample_weight(model_name: str, config: dict) -> bool:
    """Return True if this model accepts sample_weight in fit()."""
    model_cfg = config.get("models", {}).get(model_name, {})
    if "supports_sample_weight" in model_cfg:
        return bool(model_cfg["supports_sample_weight"])
    return _DEFAULT_SAMPLE_WEIGHT_SUPPORT.get(model_name, False)


def create_baseline(config: dict, seed: int) -> Ridge:
    """Ridge regression baseline from config."""
    baseline_cfg = config.get("baseline", {})
    return Ridge(alpha=baseline_cfg.get("alpha", 1.0), random_state=seed)


def create_ridge(config: dict, seed: int) -> Ridge:
    """Ridge regression from config."""
    model_cfg = config.get("models", {}).get("ridge", {})
    return Ridge(alpha=model_cfg.get("alpha", 1.0), random_state=seed)


def create_svr(config: dict, seed: int) -> SVR:
    """Support Vector Regression from config."""
    model_cfg = config.get("models", {}).get("svr", {})
    return SVR(
        kernel=model_cfg.get("kernel", "rbf"),
        C=model_cfg.get("C", 10.0),
        epsilon=model_cfg.get("epsilon", 0.01),
        gamma=model_cfg.get("gamma", "scale"),
    )


MODEL_REGISTRY = {
    "ridge": create_ridge,
    "svr": create_svr,
}
