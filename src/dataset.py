"""Scale harvest features and persist train/validation arrays."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

BATCH_SIZE = 4


def prepare_datasets(
    features_csv: str | Path,
    train_path: str | Path,
    val_path: str | Path,
    scaler_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Fit a scaler on the training split only and save both array datasets."""
    frame = pd.read_csv(features_csv)
    missing = [column for column in (*FEATURE_COLUMNS, TARGET_COLUMN) if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")

    matrix = frame.loc[:, list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN]
    x_train, x_val, y_train, y_val = train_test_split(
        matrix,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_val_scaled = scaler.transform(x_val).astype(np.float32)
    y_train_col = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val_col = y_val.to_numpy(dtype=np.float32).reshape(-1, 1)
    feature_names = np.array(FEATURE_COLUMNS)

    train_path = Path(train_path)
    val_path = Path(val_path)
    scaler_path = Path(scaler_path)
    for path in (train_path, val_path, scaler_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(train_path, features=x_train_scaled, target=y_train_col, feature_names=feature_names)
    np.savez(val_path, features=x_val_scaled, target=y_val_col, feature_names=feature_names)
    joblib.dump(scaler, scaler_path)

    return {
        "n_train": int(x_train_scaled.shape[0]),
        "n_val": int(x_val_scaled.shape[0]),
        "n_features": int(x_train_scaled.shape[1]),
        "train_shape": tuple(x_train_scaled.shape),
        "val_shape": tuple(x_val_scaled.shape),
        "target_train_shape": tuple(y_train_col.shape),
        "target_val_shape": tuple(y_val_col.shape),
        "scaler_path": str(scaler_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    """Load one ``.npz`` split written by :func:`prepare_datasets`."""
    payload = np.load(path)
    return {
        "features": payload["features"],
        "target": payload["target"],
        "feature_names": payload["feature_names"],
    }
