"""Population Stability Index (PSI) for feature drift.

The reference profile is taken from the training data and stored in the model card.
At monitoring time, recent feature values are binned with the same edges and compared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PSI_WARN = 0.1
PSI_ALERT = 0.25
_EPS = 1e-6


def reference_profile(df: pd.DataFrame, features: list[str], bins: int = 10) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for col in features:
        values = df[col].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values, bins=edges)
        profile[col] = {
            "edges": [float(e) for e in edges],
            "proportions": (counts / counts.sum()).tolist(),
        }
    return profile


def psi(reference: dict[str, Any], current: np.ndarray) -> float:
    edges = np.asarray(reference["edges"], dtype=float)
    expected = np.asarray(reference["proportions"], dtype=float)
    counts, _ = np.histogram(np.asarray(current, dtype=float), bins=edges)
    if counts.sum() == 0:
        return float("nan")
    actual = counts / counts.sum()
    expected, actual = np.clip(expected, _EPS, None), np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(profile: dict[str, Any], current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, ref in profile.items():
        if col not in current:
            continue
        value = psi(ref, current[col].dropna().to_numpy())
        status = "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
