"""End-to-end HarvestSafe pipeline: ingest, weather, features, 1-epoch train."""

import argparse
from pathlib import Path

from src.dataset import prepare_datasets
from src.feature_engineering import TARGET_COLUMN, build_feature_table
from src.fetch_weather import HARVEST_END, HARVEST_START, load_or_fetch_weather
from src.ingest_nisr import load_nisr_yields
from src.model import build_harvest_loss_predictor, trainable_parameter_count
from src.train import run_training

ROOT = Path(__file__).resolve().parent
RAW_XLSX = ROOT / "data" / "raw" / "SAS 2025_Annual Tables.xlsx"
WEATHER_CSV = ROOT / "data" / "raw" / "rwanda_historical_weather.csv"
FEATURES_CSV = ROOT / "data" / "processed" / "merged_harvest_features.csv"
TRAIN_NPZ = ROOT / "data" / "processed" / "train_dataset.npz"
VAL_NPZ = ROOT / "data" / "processed" / "val_dataset.npz"
SCALER_PKL = ROOT / "data" / "processed" / "scaler.pkl"


def _banner(step: str, title: str) -> None:
    print(f"\n[{step}] {title}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HarvestSafe verification pipeline.")
    parser.add_argument(
        "--refresh-weather",
        action="store_true",
        help="Ignore the cached Open-Meteo CSV and query the archive again.",
    )
    args = parser.parse_args()

    _banner("1/4", "NISR ingestion (SAS 2025, Table 24)")
    yields = load_nisr_yields(RAW_XLSX)
    print(f"  source: {RAW_XLSX}")
    print(f"  rows: {len(yields)}  columns: {list(yields.columns)}")
    print(
        "  vulnerable yield (maize+bean, MT): "
        f"min={yields['total_vulnerable_yield_mt'].min():.1f}  "
        f"max={yields['total_vulnerable_yield_mt'].max():.1f}"
    )

    _banner("2/4", f"Open-Meteo harvest window {HARVEST_START} to {HARVEST_END}")
    _daily, weather = load_or_fetch_weather(WEATHER_CSV, refresh=args.refresh_weather)
    print(f"  districts: {len(weather)}")
    print(
        "  rainfall mm: "
        f"min={weather['total_harvest_rainfall_mm'].min():.1f}  "
        f"mean={weather['total_harvest_rainfall_mm'].mean():.1f}  "
        f"max={weather['total_harvest_rainfall_mm'].max():.1f}"
    )
    print(
        "  heavy-rain days: "
        f"min={weather['heavy_rain_days'].min()}  "
        f"mean={weather['heavy_rain_days'].mean():.1f}  "
        f"max={weather['heavy_rain_days'].max()}"
    )
    print(
        "  mean humidity %: "
        f"min={weather['mean_relative_humidity'].min():.1f}  "
        f"max={weather['mean_relative_humidity'].max():.1f}"
    )

    _banner("3/4", "Feature engineering and array datasets")
    features = build_feature_table(yields, weather, FEATURES_CSV)
    print(f"  merged rows: {len(features)}  wrote {FEATURES_CSV}")
    print(
        f"  {TARGET_COLUMN}: "
        f"min={features[TARGET_COLUMN].min():.4f}  "
        f"mean={features[TARGET_COLUMN].mean():.4f}  "
        f"max={features[TARGET_COLUMN].max():.4f}"
    )
    dataset_info = prepare_datasets(FEATURES_CSV, TRAIN_NPZ, VAL_NPZ, SCALER_PKL)
    print(
        f"  train features {dataset_info['train_shape']}  "
        f"target {dataset_info['target_train_shape']}"
    )
    print(
        f"  val features {dataset_info['val_shape']}  "
        f"target {dataset_info['target_val_shape']}"
    )
    print(f"  scaler: {dataset_info['scaler_path']}")

    _banner("4/4", "HarvestLossPredictor, 1-epoch TensorBoard run")
    n_parameters = trainable_parameter_count(build_harvest_loss_predictor())
    print("  framework: TensorFlow/Keras")
    print("  architecture: 7 -> 32 -> 16 -> 1 (sigmoid scaled to 0.50)")
    print(f"  trainable parameters: {n_parameters}")
    metrics = run_training(TRAIN_NPZ, VAL_NPZ)
    print(f"  tensorboard: {metrics['log_dir']}")
    print(f"  checkpoint: {metrics['checkpoint_path']}")
    print(
        "METRICS "
        f"train_mse={metrics['train_mse']:.6f} "
        f"val_mse={metrics['val_mse']:.6f} "
        f"train_mae={metrics['train_mae']:.6f} "
        f"val_mae={metrics['val_mae']:.6f} "
        f"n_parameters={metrics['n_parameters']} "
        f"train_batches={metrics['train_batches']}"
    )


if __name__ == "__main__":
    main()
