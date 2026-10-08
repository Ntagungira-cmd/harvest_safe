"""Fetch Season A harvest-window weather from the Open-Meteo historical archive."""

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HARVEST_START = "2024-12-15"
HARVEST_END = "2025-02-15"
HEAVY_RAIN_MM = 10.0
RAW_COLUMNS = (
    "district",
    "latitude",
    "longitude",
    "date",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "temperature_2m_max",
)
AGGREGATE_COLUMNS = (
    "district",
    "latitude",
    "longitude",
    "total_harvest_rainfall_mm",
    "heavy_rain_days",
    "mean_relative_humidity",
    "peak_temperature_c",
)

# Administrative centres (latitude, longitude) for Rwanda's 30 districts.
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "Nyarugenge": (-1.9536, 30.0588),
    "Gasabo": (-1.8841, 30.1264),
    "Kicukiro": (-1.9706, 30.1044),
    "Nyanza": (-2.3516, 29.7506),
    "Gisagara": (-2.6031, 29.8403),
    "Nyaruguru": (-2.6972, 29.5264),
    "Huye": (-2.5967, 29.7394),
    "Nyamagabe": (-2.4667, 29.5667),
    "Ruhango": (-2.2211, 29.7736),
    "Muhanga": (-2.0844, 29.7531),
    "Kamonyi": (-2.0000, 29.9010),
    "Karongi": (-2.0603, 29.3478),
    "Rutsiro": (-1.9417, 29.3250),
    "Rubavu": (-1.7028, 29.2564),
    "Nyabihu": (-1.6531, 29.5056),
    "Ngororero": (-1.8639, 29.6167),
    "Rusizi": (-2.4842, 28.9075),
    "Nyamasheke": (-2.3403, 29.1467),
    "Rulindo": (-1.7417, 29.9917),
    "Gakenke": (-1.6983, 29.7850),
    "Musanze": (-1.4998, 29.6350),
    "Burera": (-1.4780, 29.8350),
    "Gicumbi": (-1.5763, 30.0675),
    "Rwamagana": (-1.9486, 30.4347),
    "Nyagatare": (-1.2928, 30.3250),
    "Gatsibo": (-1.5958, 30.4286),
    "Kayonza": (-1.8833, 30.4500),
    "Kirehe": (-2.2639, 30.6531),
    "Ngoma": (-2.1597, 30.5428),
    "Bugesera": (-2.1450, 30.0930),
}


class ArchiveQueryError(requests.RequestException):
    """The archive answered, but rejected the query (do not retry)."""


def _request_json(params: dict, retries: int = 3) -> dict:
    """GET the archive endpoint, retrying transient network failures."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise ArchiveQueryError(payload.get("reason", "Open-Meteo error"))
            return payload
        except ArchiveQueryError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
    raise requests.RequestException(
        f"Open-Meteo request failed after {retries} attempts: {last_error}"
    ) from last_error


def _daily_humidity(hourly: dict) -> dict[str, float]:
    """Mean hourly relative humidity onto calendar dates."""
    times = hourly.get("time") or []
    humidity = hourly.get("relative_humidity_2m") or []
    buckets: dict[str, list[float]] = {}
    for stamp, value in zip(times, humidity):
        if value is None:
            continue
        buckets.setdefault(str(stamp)[:10], []).append(float(value))
    return {day: sum(values) / len(values) for day, values in buckets.items() if values}


def fetch_district_daily(district: str, latitude: float, longitude: float) -> pd.DataFrame:
    """Download one district's harvest-window daily series.

    Daily ``precipitation_sum`` and ``temperature_2m_max`` come from the archive
    daily endpoint. ``relative_humidity_2m_mean`` is requested as a daily
    variable and, if the archive rejects it, computed from hourly
    ``relative_humidity_2m``.
    """
    base = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": HARVEST_START,
        "end_date": HARVEST_END,
        "timezone": "Africa/Kigali",
    }
    try:
        payload = _request_json(
            {
                **base,
                "daily": "precipitation_sum,relative_humidity_2m_mean,temperature_2m_max",
            }
        )
        daily = payload["daily"]
        humidity = daily["relative_humidity_2m_mean"]
    except (requests.RequestException, KeyError):
        payload = _request_json(
            {
                **base,
                "daily": "precipitation_sum,temperature_2m_max",
                "hourly": "relative_humidity_2m",
            }
        )
        daily = payload["daily"]
        by_day = _daily_humidity(payload.get("hourly", {}))
        humidity = [by_day.get(day) for day in daily["time"]]

    frame = pd.DataFrame(
        {
            "district": district,
            "latitude": latitude,
            "longitude": longitude,
            "date": daily["time"],
            "precipitation_sum": daily["precipitation_sum"],
            "relative_humidity_2m_mean": humidity,
            "temperature_2m_max": daily["temperature_2m_max"],
        }
    )
    return frame.loc[:, list(RAW_COLUMNS)]


def aggregate_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily series to one harvest-window row per district."""
    records = []
    for district, group in daily.groupby("district", sort=False):
        rain = pd.to_numeric(group["precipitation_sum"], errors="coerce").fillna(0.0)
        humidity = pd.to_numeric(group["relative_humidity_2m_mean"], errors="coerce")
        temperature = pd.to_numeric(group["temperature_2m_max"], errors="coerce")
        records.append(
            {
                "district": district,
                "latitude": float(group["latitude"].iloc[0]),
                "longitude": float(group["longitude"].iloc[0]),
                "total_harvest_rainfall_mm": float(rain.sum()),
                "heavy_rain_days": int((rain > HEAVY_RAIN_MM).sum()),
                "mean_relative_humidity": float(humidity.mean()),
                "peak_temperature_c": float(temperature.max()),
            }
        )
    aggregated = pd.DataFrame(records, columns=list(AGGREGATE_COLUMNS))
    if len(aggregated) != 30:
        raise AssertionError(f"Expected weather for 30 districts, found {len(aggregated)}")
    if aggregated[list(AGGREGATE_COLUMNS[3:])].isna().any().any():
        raise ValueError("Weather aggregation produced missing values")
    return aggregated


def load_or_fetch_weather(
    raw_csv: str | Path,
    refresh: bool = False,
    pause_seconds: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(daily, aggregates)``, reusing ``raw_csv`` unless ``refresh``."""
    raw_csv = Path(raw_csv)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)

    if raw_csv.exists() and not refresh:
        daily = pd.read_csv(raw_csv)
        missing = [column for column in RAW_COLUMNS if column not in daily.columns]
        if missing:
            raise ValueError(f"Cached weather file {raw_csv} is missing columns: {missing}")
        print(f"  reusing cached weather: {raw_csv} ({len(daily)} daily rows)")
        return daily, aggregate_weather(daily)

    if len(DISTRICT_COORDS) != 30:
        raise AssertionError(f"Coordinate map must contain 30 districts, found {len(DISTRICT_COORDS)}")

    frames = []
    for district, (latitude, longitude) in tqdm(
        DISTRICT_COORDS.items(),
        desc="Open-Meteo districts",
        unit="district",
    ):
        frames.append(fetch_district_daily(district, latitude, longitude))
        time.sleep(pause_seconds)

    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(raw_csv, index=False)
    print(f"  wrote daily weather: {raw_csv} ({len(daily)} rows)")
    return daily, aggregate_weather(daily)
