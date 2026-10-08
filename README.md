# HarvestSafe

District-level post-harvest loss prediction for Rwanda's Season A maize and bean harvest. The pipeline reads NISR SAS 2025 Table 24, joins real Open-Meteo harvest-window weather for all 30 districts, builds a literature-bounded loss target, and trains a small Keras network for one verification epoch with TensorBoard logging.

The loss label is a reproducible synthetic proxy. NISR does not publish district-level post-harvest loss, and a single epoch on 24 training districts only checks that the pipeline runs.

## Layout

```
data/raw/SAS 2025_Annual Tables.xlsx     NISR workbook (Table 24)
data/raw/rwanda_historical_weather.csv   Daily archive weather (fetched)
data/processed/merged_harvest_features.csv
data/processed/train_dataset.npz
data/processed/val_dataset.npz
data/processed/scaler.pkl
src/                                    Ingest, weather, features, model, train
runs/harvestsafe_experiment_1/          TensorBoard event file
runs/harvestsafe_model_epoch1.keras     Keras checkpoint after 1 epoch
runs/harvestsafe_metrics.json           MSE, MAE, and parameter count
notebooks/pipeline_demo.ipynb           Read-only walkthrough of the artifacts
docs/PROPOSAL.md                        Formative proposal
main.py                                 End-to-end entry point
```

## Setup

Python 3.11 is required (`tensorflow` and `pandas` wheels). From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the NISR workbook at `data/raw/SAS 2025_Annual Tables.xlsx` if it is not already there.

## Run

```bash
python main.py
tensorboard --logdir=runs
```

`main.py` prints row counts, tensor shapes, and the one-epoch MSE and MAE. The first weather fetch calls the Open-Meteo archive once per district and then caches `data/raw/rwanda_historical_weather.csv`. Pass `--refresh-weather` to query the archive again.

The demo notebook loads those artifacts. It does not retrain the model:

```bash
jupyter notebook notebooks/pipeline_demo.ipynb
```

## Model

`HarvestLossPredictor` is a Keras model. It maps 7 standardized features through `Dense(32) → BatchNormalization → LeakyReLU(0.1) → Dropout(0.2) → Dense(16) → LeakyReLU(0.1) → Dropout(0.1) → Dense(1)`, then scales a sigmoid into `[0, 0.50]`. Training uses AdamW (`lr=0.003`, `weight_decay=1e-4`) and mean squared error. Mean absolute error is logged only. The verification run is fixed at 1 epoch.
