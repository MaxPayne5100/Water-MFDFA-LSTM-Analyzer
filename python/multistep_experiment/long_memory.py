"""Training-only semiparametric estimation of fractional order ``d``."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import D_MAX, D_MIN


@dataclass(frozen=True)
class FractionalOrderEstimate:
    local_whittle: float
    gph: float
    selected: float
    bandwidth: int
    disagreement: float
    unstable: bool
    clipped: bool
    bootstrap_std_local_whittle: float | None = None
    bootstrap_std_gph: float | None = None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def detrend_linear(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if len(series) < 8 or not np.isfinite(series).all():
        raise ValueError("Long-memory estimation requires at least 8 finite values")
    time = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(time, series, 1)
    return series - (slope * time + intercept)


def _low_frequency_periodogram(
    values: np.ndarray, bandwidth: int | None
) -> tuple[np.ndarray, np.ndarray, int]:
    residuals = detrend_linear(values)
    n = len(residuals)
    maximum = max(2, n // 2 - 1)
    chosen = int(np.floor(n**0.65)) if bandwidth is None else int(bandwidth)
    chosen = min(maximum, max(5, chosen))
    if chosen > maximum:
        raise ValueError("Series is too short for the requested spectral bandwidth")
    frequencies = 2.0 * np.pi * np.arange(1, chosen + 1) / n
    fft_values = np.fft.fft(residuals)
    periodogram = np.square(np.abs(fft_values[1 : chosen + 1])) / (2.0 * np.pi * n)
    periodogram = np.maximum(periodogram, np.finfo(float).tiny)
    return frequencies, periodogram, chosen


def estimate_local_whittle(
    values: np.ndarray,
    *,
    bandwidth: int | None = None,
    bounds: tuple[float, float] = (D_MIN, D_MAX),
) -> tuple[float, int]:
    frequencies, periodogram, chosen = _low_frequency_periodogram(values, bandwidth)
    mean_log_frequency = float(np.mean(np.log(frequencies)))

    def objective(order: float) -> float:
        scaled = periodogram * np.power(frequencies, 2.0 * order)
        return float(np.log(np.mean(scaled)) - 2.0 * order * mean_log_frequency)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise ValueError("Local Whittle optimization did not converge")
    return float(result.x), chosen


def estimate_gph(
    values: np.ndarray,
    *,
    bandwidth: int | None = None,
) -> tuple[float, int]:
    frequencies, periodogram, chosen = _low_frequency_periodogram(values, bandwidth)
    regressor = -2.0 * np.log(2.0 * np.sin(frequencies / 2.0))
    design = np.column_stack([np.ones_like(regressor), regressor])
    coefficients, *_ = np.linalg.lstsq(design, np.log(periodogram), rcond=None)
    return float(coefficients[1]), chosen


def _moving_block_resample(
    values: np.ndarray, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    starts = np.arange(0, len(series) - block_length + 1)
    blocks_needed = int(np.ceil(len(series) / block_length))
    selected = rng.choice(starts, size=blocks_needed, replace=True)
    return np.concatenate([series[start : start + block_length] for start in selected])[
        : len(series)
    ]


def estimate_fractional_order(
    training_values: np.ndarray,
    *,
    bandwidth: int | None = None,
    disagreement_tolerance: float = 0.15,
    bootstrap_repeats: int = 0,
    seed: int = 42,
) -> FractionalOrderEstimate:
    """Use Local Whittle as primary and GPH as a robustness diagnostic."""
    local_whittle, chosen = estimate_local_whittle(
        training_values, bandwidth=bandwidth
    )
    gph_raw, _ = estimate_gph(training_values, bandwidth=chosen)
    gph = float(np.clip(gph_raw, D_MIN, D_MAX))
    selected = float(np.clip(local_whittle, D_MIN, D_MAX))
    clipped = not np.isclose(gph, gph_raw) or not np.isclose(selected, local_whittle)

    local_std = None
    gph_std = None
    if bootstrap_repeats > 0:
        rng = np.random.default_rng(seed)
        block_length = max(4, int(round(np.sqrt(len(training_values)))))
        local_samples: list[float] = []
        gph_samples: list[float] = []
        for _ in range(bootstrap_repeats):
            sample = _moving_block_resample(training_values, block_length, rng)
            try:
                local_samples.append(estimate_local_whittle(sample, bandwidth=chosen)[0])
                gph_samples.append(estimate_gph(sample, bandwidth=chosen)[0])
            except ValueError:
                continue
        if len(local_samples) >= 2:
            local_std = float(np.std(local_samples, ddof=1))
        if len(gph_samples) >= 2:
            gph_std = float(np.std(gph_samples, ddof=1))

    disagreement = abs(selected - gph)
    return FractionalOrderEstimate(
        local_whittle=selected,
        gph=gph,
        selected=selected,
        bandwidth=chosen,
        disagreement=disagreement,
        unstable=disagreement > disagreement_tolerance,
        clipped=clipped,
        bootstrap_std_local_whittle=local_std,
        bootstrap_std_gph=gph_std,
    )
