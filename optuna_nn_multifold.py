"""Optuna search for TCN over all 3 holdout folds (Q1 2023, 2024, 2025)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import optuna
import pandas as pd

from solution import DATE_COL, TARGET, CAPACITY_MW, SEED, TRAIN_PATH, REPAIR_COL
import nn_model

PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
FOLD_YEARS = [2023, 2024, 2025]


def metric_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.clip(y_pred, 0.0, CAPACITY_MW))))


def objective(trial, train_df):
    epochs = trial.suggest_int("epochs", 18, 40)
    batch = trial.suggest_categorical("batch", [128, 256, 512])
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.08, 0.28)

    fold_maes = []
    for yr in FOLD_YEARS:
        hold_mask = (train_df[DATE_COL].dt.year == yr) & (train_df[DATE_COL].dt.month <= 3)
        fit = train_df.loc[~hold_mask].copy()
        hold = train_df.loc[hold_mask].copy()
        y_hold = hold[TARGET].astype(float).to_numpy()
        pred = nn_model.fit_nn(
            fit, hold.drop(columns=[TARGET]),
            target_col=TARGET, date_col=DATE_COL, repair_col=REPAIR_COL,
            capacity_mw=CAPACITY_MW, seed=SEED,
            epochs=epochs, batch=batch, lr=lr, dropout=dropout, verbose=False,
        )
        fold_maes.append(metric_mae(y_hold, pred))
    return float(np.mean(fold_maes))


def main(n_trials: int):
    train = pd.read_csv(TRAIN_PATH)
    train[DATE_COL] = pd.to_datetime(train[DATE_COL])

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="tcn_multifold")
    study.optimize(lambda t: objective(t, train), n_trials=n_trials, show_progress_bar=False)

    print(f"BEST [tcn multifold] MAE: {study.best_value:.4f}")
    print(f"BEST params: {study.best_params}")

    if PARAMS_PATH.exists():
        existing = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing["tcn"] = {
        "params": study.best_params,
        "best_mae": float(study.best_value),
        "n_trials": n_trials,
        "tuned_on": "multifold (Q1/2023, 2024, 2025)",
    }
    PARAMS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {PARAMS_PATH}")


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(trials)
