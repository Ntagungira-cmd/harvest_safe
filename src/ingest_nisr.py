"""Ingest NISR SAS 2025 Table 24 district crop production (metric tons)."""

import warnings
from pathlib import Path

import pandas as pd

# Official 30 districts. Provincial headers and national totals are excluded.
OFFICIAL_DISTRICTS = (
    "Nyarugenge",
    "Gasabo",
    "Kicukiro",
    "Nyanza",
    "Gisagara",
    "Nyaruguru",
    "Huye",
    "Nyamagabe",
    "Ruhango",
    "Muhanga",
    "Kamonyi",
    "Karongi",
    "Rutsiro",
    "Rubavu",
    "Nyabihu",
    "Ngororero",
    "Rusizi",
    "Nyamasheke",
    "Rulindo",
    "Gakenke",
    "Musanze",
    "Burera",
    "Gicumbi",
    "Rwamagana",
    "Nyagatare",
    "Gatsibo",
    "Kayonza",
    "Kirehe",
    "Ngoma",
    "Bugesera",
)

YIELD_COLUMNS = (
    "district",
    "maize_yield_mt",
    "bean_yield_mt",
    "total_vulnerable_yield_mt",
)


def load_nisr_yields(path: str | Path) -> pd.DataFrame:
    """Read Table 24 and return maize and bean production for all 30 districts.

    The sheet header sits on Excel row 3 (``header=2``). Rows below the
    district block (``S/Total``, ``2024A``, ``Change``, source note) are
    dropped by matching against the official district list.
    """
    path = Path(path)
    # This workbook stores a defined-name print area that openpyxl cannot apply.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Print area cannot be set")
        raw = pd.read_excel(path, sheet_name="Table 24", header=2, usecols="B:AC")
    raw.columns = [str(column).strip() for column in raw.columns]

    missing = {"District", "Maize", "Bean"} - set(raw.columns)
    if missing:
        raise ValueError(f"Table 24 is missing expected columns: {sorted(missing)}")

    frame = raw[["District", "Maize", "Bean"]].copy()
    frame["District"] = (
        frame["District"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    )

    official = {name.title() for name in OFFICIAL_DISTRICTS}
    frame = frame[frame["District"].isin(official)].copy()
    frame = frame.rename(
        columns={
            "District": "district",
            "Maize": "maize_yield_mt",
            "Bean": "bean_yield_mt",
        }
    )
    for column in ("maize_yield_mt", "bean_yield_mt"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[["maize_yield_mt", "bean_yield_mt"]].isna().any().any():
        bad = frame.loc[frame[["maize_yield_mt", "bean_yield_mt"]].isna().any(axis=1), "district"]
        raise ValueError(f"Non-numeric yields for districts: {bad.tolist()}")

    frame["total_vulnerable_yield_mt"] = frame["maize_yield_mt"] + frame["bean_yield_mt"]
    frame = frame.loc[:, list(YIELD_COLUMNS)].reset_index(drop=True)

    found = set(frame["district"])
    if found != official or len(frame) != 30:
        raise AssertionError(
            f"Expected 30 official districts, found {len(frame)}. "
            f"Missing: {sorted(official - found)}. Extra: {sorted(found - official)}."
        )
    return frame
