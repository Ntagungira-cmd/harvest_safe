"""Merge NISR yields with harvest-window weather and build the loss target."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "total_harvest_rainfall_mm",
    "heavy_rain_days",
    "mean_relative_humidity",
    "peak_temperature_c",
    "rainfall_intensity_ratio",
    "moisture_hazard_index",
    "log_yield_exposure",
)
TARGET_COLUMN = "post_harvest_loss_pct"


def _minmax(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())
    if high == low:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series - low) / (high - low)


def build_feature_table(
    yields: pd.DataFrame,
    weather: pd.DataFrame,
    output_csv: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Inner-join yields and weather, engineer features, and write the CSV.

    ``post_harvest_loss_pct`` is a literature-bounded synthetic target. NISR
    does not publish district-level post-harvest loss, so the label is a
    reproducible function of moisture hazard, heavy-rain days, and exposed
    yield, plus small Gaussian noise.
    """
    merged = yields.merge(weather, on="district", how="inner", validate="one_to_one")
    if len(merged) != 30:
        missing_yields = set(yields["district"]) - set(weather["district"])
        missing_weather = set(weather["district"]) - set(yields["district"])
        raise AssertionError(
            f"Expected 30 merged districts, found {len(merged)}. "
            f"Missing weather: {sorted(missing_yields)}. "
            f"Missing yields: {sorted(missing_weather)}."
        )

    merged["rainfall_intensity_ratio"] = merged["total_harvest_rainfall_mm"] / (
        merged["heavy_rain_days"] + 1
    )
    merged["moisture_hazard_index"] = (merged["mean_relative_humidity"] / 100.0) * merged[
        "total_harvest_rainfall_mm"
    ]
    merged["log_yield_exposure"] = np.log1p(merged["total_vulnerable_yield_mt"])

    risk = (
        0.4 * _minmax(merged["moisture_hazard_index"])
        + 0.3 * _minmax(merged["heavy_rain_days"])
        + 0.3 * _minmax(merged["log_yield_exposure"])
    )
    noise = np.random.default_rng(seed).normal(0.0, 0.015, size=len(merged))
    loss = 0.05 + (0.40 / (1.0 + np.exp(-6.0 * (risk.to_numpy() - 0.5)))) + noise
    merged[TARGET_COLUMN] = np.clip(loss, 0.05, 0.48)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged
