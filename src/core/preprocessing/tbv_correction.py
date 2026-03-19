"""Per-fold ICV correction for TBV/ICV correction.

Applied to volumetric and cortical thickness features in both regression and
univariate analyses. Inserted before ComBat harmonization (ratio method) or
after (residualize method), before StandardScaler.

Correction rules (ratio method):
  - Volumes:            V_corrected = V_raw / ICV
  - Cortical thickness: CT_corrected = CT_raw / ICV^(1/3)
  - DTI / FA / MD:      no correction

Rationale:
  - Subcortical volumes scale linearly with ICV → divide by ICV
  - Cortical thickness scales with the cube root of brain volume (a linear
    dimension) → divide by ICV^(1/3)
  - AI = (L-R)/(L+R) is naturally unaffected because ICV cancels
  - Per-fold fit (train only) → apply (train + test) avoids data leakage
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def identify_volume_features(feature_cols: list[str], volume_substring: str = "__vol__") -> list[int]:
    """Return indices of volumetric features in the feature column list.

    Args:
        feature_cols: Ordered list of feature column names.
        volume_substring: Substring that identifies volumetric columns.

    Returns:
        List of integer indices into feature_cols for volumetric features.
    """
    return [i for i, c in enumerate(feature_cols) if volume_substring in c]


def identify_thickness_features(feature_cols: list[str], thickness_substring: str = "__thk__") -> list[int]:
    """Return indices of cortical thickness features in the feature column list.

    Args:
        feature_cols: Ordered list of feature column names.
        thickness_substring: Substring that identifies cortical thickness columns.

    Returns:
        List of integer indices into feature_cols for thickness features.
    """
    return [i for i, c in enumerate(feature_cols) if thickness_substring in c]


def apply_icv_ratio_correction(
    X: np.ndarray,
    icv: np.ndarray,
    vol_indices: list[int],
    thk_indices: list[int] | None = None,
) -> np.ndarray:
    """Ratio-based ICV correction for volumes and cortical thickness.

    - Volumes:   X / ICV
    - Thickness: X / ICV^(1/3)

    Deterministic, no train/test leakage concern — each subject's features
    are divided by their own ICV.  AI = (L-R)/(L+R) is mathematically
    unaffected because ICV cancels in both numerator and denominator.

    Args:
        X: Feature matrix (n_subjects, n_features).
        icv: ICV values (n_subjects,).
        vol_indices: Indices of volumetric features to correct (/ ICV).
        thk_indices: Indices of cortical thickness features to correct (/ ICV^(1/3)).

    Returns:
        Corrected feature matrix (copy; original unchanged).
    """
    X_corrected = X.copy()
    icv_flat = icv.ravel().astype(float)

    # Volume correction: divide by ICV
    for idx in vol_indices:
        X_corrected[:, idx] = X_corrected[:, idx] / icv_flat

    # Cortical thickness correction: divide by ICV^(1/3)
    if thk_indices:
        icv_cbrt = np.cbrt(icv_flat)
        for idx in thk_indices:
            X_corrected[:, idx] = X_corrected[:, idx] / icv_cbrt

    return X_corrected


def fit_icv_correction(
    X_train: np.ndarray,
    icv_train: np.ndarray,
    vol_indices: list[int],
) -> dict:
    """Fit per-feature linear regression against ICV on training data.

    Args:
        X_train: Training feature matrix (n_subjects, n_features).
        icv_train: ICV values for training subjects (n_subjects,).
        vol_indices: Indices of volumetric features to correct.

    Returns:
        Dict with 'models' (list of LinearRegression per vol feature),
        'train_means' (mean per vol feature on training set), 'vol_indices'.
    """
    n_features = X_train.shape[1]
    models: list[LinearRegression | None] = [None] * n_features
    train_means: list[float | None] = [None] * n_features

    icv_2d = icv_train.reshape(-1, 1)
    for idx in vol_indices:
        model = LinearRegression()
        model.fit(icv_2d, X_train[:, idx])
        models[idx] = model
        train_means[idx] = float(X_train[:, idx].mean())

    return {"models": models, "train_means": train_means, "vol_indices": vol_indices}


def apply_icv_correction(
    X: np.ndarray,
    icv: np.ndarray,
    fitted: dict,
) -> np.ndarray:
    """Residualize volumetric features against ICV.

    Correction: X_corrected = X - predicted + train_mean
    This removes ICV-driven variance while preserving the training set's
    grand mean (avoiding zero-centering artifacts).

    Args:
        X: Feature matrix to correct (n_subjects, n_features).
        icv: ICV values (n_subjects,).
        fitted: Output from fit_icv_correction.

    Returns:
        Corrected feature matrix (copy; original unchanged).
    """
    X_corrected = X.copy()
    icv_2d = icv.reshape(-1, 1)

    for idx in fitted["vol_indices"]:
        model = fitted["models"][idx]
        if model is None:
            continue
        predicted = model.predict(icv_2d)
        train_mean = fitted["train_means"][idx]
        X_corrected[:, idx] = X_corrected[:, idx] - predicted + train_mean

    return X_corrected
