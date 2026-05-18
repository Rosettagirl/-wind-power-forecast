"""Multi-fold Optuna for CatBoost and LightGBM with current feature set."""
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.impute import SimpleImputer

from solution import (
    DATE_COL, TARGET, CAPACITY_MW, SEED, TRAIN_PATH,
    make_xy, to_catboost_xy, to_numeric_xy, clip,
)

PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
FOLD_YEARS = [2023, 2024, 2025]


def metric_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.clip(y_pred, 0.0, CAPACITY_MW))))


def objective_lgbm(trial, fold_data):
    params = {
        "objective": "regression_l1",
        "n_estimators": 3000,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    fold_maes = []
    iters = []
    for x_fit, y_fit, x_hold, y_hold in fold_data:
        model = LGBMRegressor(**params)
        model.fit(x_fit, y_fit, eval_set=[(x_hold, y_hold)], eval_metric="mae",
                  callbacks=[early_stopping(stopping_rounds=80, verbose=False)])
        pred = clip(model.predict(x_hold, num_iteration=model.best_iteration_))
        fold_maes.append(metric_mae(y_hold, pred))
        iters.append(int(model.best_iteration_ or params["n_estimators"]))
    trial.set_user_attr("best_iteration", int(np.mean(iters)))
    return float(np.mean(fold_maes))


def objective_cat(trial, fold_data):
    params = {
        "loss_function": "MAE",
        "iterations": 3000,
        "depth": trial.suggest_int("depth", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "eval_metric": "MAE",
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
        "early_stopping_rounds": 80,
    }
    fold_maes = []
    iters = []
    for x_fit, y_fit, x_hold, y_hold, cat_idx in fold_data:
        model = CatBoostRegressor(**params)
        model.fit(x_fit, y_fit, cat_features=cat_idx, eval_set=(x_hold, y_hold))
        pred = clip(model.predict(x_hold))
        fold_maes.append(metric_mae(y_hold, pred))
        iters.append(int(model.get_best_iteration() or params["iterations"]))
    trial.set_user_attr("best_iteration", int(np.mean(iters)))
    return float(np.mean(fold_maes))


def prepare_folds_numeric(train_df):
    """Pre-compute (x_fit, y_fit, x_hold, y_hold) for each fold for LGBM/XGB."""
    out = []
    for yr in FOLD_YEARS:
        hold_mask = (train_df[DATE_COL].dt.year == yr) & (train_df[DATE_COL].dt.month <= 3)
        fit_df = train_df.loc[~hold_mask].copy()
        hold_df = train_df.loc[hold_mask].copy()
        y_hold = hold_df[TARGET].astype(float).to_numpy()
        x_fit, y_fit, x_hold = make_xy(fit_df, hold_df.drop(columns=[TARGET]))
        x_fit, x_hold = to_numeric_xy(x_fit, x_hold)
        imputer = SimpleImputer(strategy="median")
        x_fit = imputer.fit_transform(x_fit)
        x_hold = imputer.transform(x_hold)
        out.append((x_fit, y_fit, x_hold, y_hold))
    return out


def prepare_folds_catboost(train_df):
    """Pre-compute (x_fit, y_fit, x_hold, y_hold, cat_idx) for each fold for CatBoost."""
    out = []
    for yr in FOLD_YEARS:
        hold_mask = (train_df[DATE_COL].dt.year == yr) & (train_df[DATE_COL].dt.month <= 3)
        fit_df = train_df.loc[~hold_mask].copy()
        hold_df = train_df.loc[hold_mask].copy()
        y_hold = hold_df[TARGET].astype(float).to_numpy()
        x_fit, y_fit, x_hold = make_xy(fit_df, hold_df.drop(columns=[TARGET]))
        x_fit, x_hold, cat_idx = to_catboost_xy(x_fit, x_hold)
        out.append((x_fit, y_fit, x_hold, y_hold, cat_idx))
    return out


def main(model_type: str, n_trials: int):
    train = pd.read_csv(TRAIN_PATH)
    train[DATE_COL] = pd.to_datetime(train[DATE_COL])

    print(f"preparing folds for {model_type}...")
    if model_type == "lightgbm":
        folds = prepare_folds_numeric(train)
        obj = lambda t: objective_lgbm(t, folds)
    elif model_type == "catboost":
        folds = prepare_folds_catboost(train)
        obj = lambda t: objective_cat(t, folds)
    else:
        raise ValueError(model_type)
    print(f"folds ready, starting optuna ({n_trials} trials)")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{model_type}_multifold")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    print(f"BEST [{model_type} multifold] MAE: {study.best_value:.4f}")
    print(f"BEST params: {study.best_params}")
    print(f"BEST iter: {study.best_trial.user_attrs.get('best_iteration')}")

    if PARAMS_PATH.exists():
        existing = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing[model_type] = {
        "params": study.best_params,
        "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        "best_mae": float(study.best_value),
        "n_trials": n_trials,
        "tuned_on": "multifold (Q1/2023, 2024, 2025) + OpenMeteo extras",
    }
    PARAMS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {PARAMS_PATH}")


if __name__ == "__main__":
    mt = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    main(mt, trials)
