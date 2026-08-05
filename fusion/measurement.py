"""Algoritmadan bagimsiz radar olcum kovaryansi yardimcilari."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


TQ_MIN, TQ_MAX = 1.0, 15.0
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5


def tq_to_covariance(track_quality: float) -> np.ndarray:
    tq = float(np.clip(track_quality, TQ_MIN, TQ_MAX))
    fraction = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    sigma_position = SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** fraction
    sigma_velocity = SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** fraction
    return np.diag([
        sigma_position**2, sigma_velocity**2,
        sigma_position**2, sigma_velocity**2,
        sigma_position**2, sigma_velocity**2,
    ])


def measurement_covariance(values: Mapping[str, Any]) -> np.ndarray:
    try:
        sigma_position = float(values.get("sigma_pos_m"))
        sigma_velocity = float(values.get("sigma_vel_mps"))
    except (TypeError, ValueError):
        sigma_position = sigma_velocity = math.nan
    if (
        math.isfinite(sigma_position) and sigma_position > 0.0
        and math.isfinite(sigma_velocity) and sigma_velocity > 0.0
    ):
        return np.diag([
            sigma_position**2, sigma_velocity**2,
            sigma_position**2, sigma_velocity**2,
            sigma_position**2, sigma_velocity**2,
        ])
    return tq_to_covariance(float(values.get("track_quality", TQ_MAX)))
