from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dataset"
TRAIN_PATH = DATA_DIR / "train_dataset.csv"
VALID_PATH = DATA_DIR / "valid_features.csv"
SUBMISSION_PATH = ROOT / "submission.csv"
REPORT_PATH = ROOT / "report.json"
BEST_PARAMS_PATH = ROOT / "best_params.json"
OPENMETEO_EXTRA_PATH = ROOT / "openmeteo_extra.csv"

if BEST_PARAMS_PATH.exists():
    BEST_PARAMS = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
else:
    BEST_PARAMS = {}

if OPENMETEO_EXTRA_PATH.exists():
    _OM_EXTRA = pd.read_csv(OPENMETEO_EXTRA_PATH)
    _OM_EXTRA["time"] = pd.to_datetime(_OM_EXTRA["time"])
    OM_EXTRA_COLS = [c for c in _OM_EXTRA.columns if c != "time"]
else:
    _OM_EXTRA = None
    OM_EXTRA_COLS = []

DATE_COL = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET = "Выработка. Результирующий расчет"
REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"
N_TURBINES = 26
TURBINE_MW = 3.465
CAPACITY_MW = N_TURBINES * TURBINE_MW  # 90.09
SEED = 20260511
N_SEEDS = 1  # seed-ensemble disabled: LB regressed 8.655 -> 8.664 with 3 seeds
SEEDS = [SEED + i * 101 for i in range(N_SEEDS)]
SAMPLE_WEIGHT_HALF_LIFE_YEARS = None  # disabled: hurts Q1 2025 fold
REF_YEAR = 2026  # target year (Q1)
SMOKE_TEST = os.environ.get("WIND_SMOKE_TEST") == "1"


def time_decay_weights(train_df: pd.DataFrame) -> np.ndarray | None:
    if SAMPLE_WEIGHT_HALF_LIFE_YEARS is None:
        return None
    years = pd.to_datetime(train_df[DATE_COL]).dt.year.to_numpy().astype(float)
    age = REF_YEAR - years
    return np.exp(-np.log(2) * age / SAMPLE_WEIGHT_HALF_LIFE_YEARS)


def clip(values):
    return np.clip(values, 0.0, CAPACITY_MW)


def metric_percent(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) / CAPACITY_MW * 100.0))


def scores(y_true, y_pred):
    y_pred = clip(y_pred)
    return {
        "mae_mw": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mw": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "metric_percent": metric_percent(y_true, y_pred),
    }


def power_curve(speed: np.ndarray) -> np.ndarray:
    """Siemens Gamesa SG 3.4-132 approximation per turbine, MW."""
    v = np.asarray(speed, dtype=float)
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    p = np.zeros_like(v)
    ramp = (v >= cut_in) & (v < rated)
    full = (v >= rated) & (v < cut_out)
    # smooth cubic between cut_in and rated
    x = np.clip((v - cut_in) / (rated - cut_in), 0.0, 1.0)
    p_ramp = TURBINE_MW * (x ** 3 * (10 - 15 * x + 6 * x ** 2))  # smoothstep^? actually use saturating cubic
    # simpler & physically motivated: P ∝ v^3 between cut_in..rated
    p_ramp_v3 = TURBINE_MW * ((v - cut_in) / (rated - cut_in)) ** 3
    p[ramp] = np.clip(p_ramp_v3[ramp], 0.0, TURBINE_MW)
    p[full] = TURBINE_MW
    # v >= cut_out → 0 (already)
    return p


def add_features(df: pd.DataFrame, sorted_by_time: bool = True) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[DATE_COL])
    out["_dt"] = dt
    # merge OpenMeteo extra variables by timestamp
    if _OM_EXTRA is not None:
        extra = _OM_EXTRA.set_index("time")
        for col in OM_EXTRA_COLS:
            if col not in out.columns:
                out[col] = dt.map(extra[col]).astype(float)
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["hour_of_day"] = dt.dt.hour
    out["dayofyear"] = dt.dt.dayofyear
    out["dayofweek"] = dt.dt.dayofweek
    out["is_q1"] = (out["month"] <= 3).astype(int)
    out["is_winter"] = out["month"].isin([12, 1, 2]).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366)

    speed_cols = ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]
    for col in speed_cols:
        out[f"{col}_sq"] = out[col] ** 2
        out[f"{col}_cube"] = out[col] ** 3
        out[f"{col}_bin025"] = (out[col] * 4).round().astype("Int64").astype(str)
        out[f"{col}_bin050"] = (out[col] * 2).round().astype("Int64").astype(str)

    out["weighted_speed"] = (
        0.68 * out["wind_speed_80m"]
        + 0.24 * out["wind_speed_120m"]
        + 0.08 * out["wind_gusts_10m"]
    )
    out["weighted_speed_sq"] = out["weighted_speed"] ** 2
    out["weighted_speed_cube"] = out["weighted_speed"] ** 3
    out["weighted_speed_bin025"] = (out["weighted_speed"] * 4).round().astype("Int64").astype(str)
    out["weighted_speed_bin050"] = (out["weighted_speed"] * 2).round().astype("Int64").astype(str)

    out["speed_80_10_diff"] = out["wind_speed_80m"] - out["wind_speed_10m"]
    out["speed_120_80_diff"] = out["wind_speed_120m"] - out["wind_speed_80m"]
    out["speed_180_120_diff"] = out["wind_speed_180m"] - out["wind_speed_120m"]
    out["gust_speed10_diff"] = out["wind_gusts_10m"] - out["wind_speed_10m"]
    out["speed_120_80_ratio"] = out["wind_speed_120m"] / out["wind_speed_80m"].clip(lower=0.1)
    out["gust_speed80_ratio"] = out["wind_gusts_10m"] / out["wind_speed_80m"].clip(lower=0.1)

    # wind shear exponent estimate (log-log between 10m and 80m)
    out["shear_10_80"] = np.log(out["wind_speed_80m"].clip(lower=0.1) / out["wind_speed_10m"].clip(lower=0.1)) / np.log(80 / 10)

    dir_cols = [
        "wind_direction_10m",
        "wind_direction_80m",
        "wind_direction_120m",
        "wind_direction_180m",
    ]
    for col in dir_cols:
        degrees = (out[col] * 1000.0) % 360.0
        radians = np.deg2rad(degrees)
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
        out[f"{col}_sector12"] = (degrees // 30).astype("Int64").astype(str)
        out[f"{col}_sector24"] = (degrees // 15).astype("Int64").astype(str)

    out["available_turbines"] = N_TURBINES - out[REPAIR_COL]
    out["available_capacity"] = out["available_turbines"] * TURBINE_MW
    out["weighted_speed_x_available"] = out["weighted_speed"] * out["available_turbines"]

    # air density (kg/m^3) at hub from p (hPa) and T (°C); ρ = p / (R·T_K), R = 287.05
    t_k = out["temperature_80m"].fillna(out["temperature_80m"].median()) + 273.15
    p_pa = out["pressure_msl"].fillna(out["pressure_msl"].median()) * 100.0
    out["air_density"] = p_pa / (287.05 * t_k)
    out["air_density_ratio"] = out["air_density"] / 1.225  # vs ISA standard

    # physical power-curve features per turbine, scaled to fleet
    for col in ["wind_speed_80m", "wind_speed_120m", "weighted_speed"]:
        pc_per_turbine = power_curve(out[col].fillna(out[col].median()).to_numpy())
        out[f"pc_{col}"] = pc_per_turbine * out["available_turbines"]
        out[f"pc_{col}_norm"] = out[f"pc_{col}"] / CAPACITY_MW
        # density-corrected: power scales linearly with air density
        out[f"pc_{col}_rho"] = out[f"pc_{col}"] * out["air_density_ratio"]

    # density-adjusted kinetic energy proxy (ρ·V³)
    out["rho_v3_80m"] = out["air_density"] * out["wind_speed_80m"] ** 3
    out["rho_v3_weighted"] = out["air_density"] * out["weighted_speed"] ** 3

    out["temp_diff_80_120"] = out["temperature_80m"] - out["temperature_120m"]
    out["precip_total"] = out["rain"] + out["showers"] + out["snowfall"]

    # rolling meteo features (no target leak — only meteo)
    if sorted_by_time:
        out = out.sort_values("_dt").reset_index(drop=True)
        for col, win in [
            ("wind_speed_80m", 3), ("wind_speed_80m", 6),
            ("wind_speed_120m", 3), ("weighted_speed", 3),
            ("weighted_speed", 6),
        ]:
            out[f"{col}_roll{win}_mean"] = out[col].rolling(win, min_periods=1).mean()
            out[f"{col}_roll{win}_std"] = out[col].rolling(win, min_periods=1).std().fillna(0.0)
        out[f"weighted_speed_diff1"] = out["weighted_speed"].diff().fillna(0.0)
        out[f"weighted_speed_diff3"] = out["weighted_speed"].diff(3).fillna(0.0)

    cat_base = ["month", "hour_of_day", REPAIR_COL, "dayofweek", "is_q1"]
    for col in cat_base:
        out[f"{col}_cat"] = out[col].astype("Int64").astype(str)

    return out.drop(columns=[DATE_COL, "_dt"], errors="ignore")


def make_xy(train_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Time-aware feature build. Returns x_train, y_train, x_pred (aligned to original input order)."""
    n_tr = len(train_df)
    combined = pd.concat(
        [train_df.drop(columns=[TARGET], errors="ignore"), pred_df.drop(columns=[TARGET], errors="ignore")],
        ignore_index=True,
    )
    combined["_orig_pos"] = np.arange(len(combined))
    dt = pd.to_datetime(combined[DATE_COL])
    order = dt.sort_values(kind="mergesort").index
    sorted_df = combined.loc[order].reset_index(drop=True)
    feats_sorted = add_features(sorted_df, sorted_by_time=True)
    feats_sorted["_orig_pos"] = sorted_df["_orig_pos"].to_numpy()
    feats = feats_sorted.sort_values("_orig_pos").reset_index(drop=True).drop(columns=["_orig_pos"])

    y_train = train_df[TARGET].astype(float).to_numpy()
    x_train = feats.iloc[:n_tr].reset_index(drop=True)
    x_pred = feats.iloc[n_tr:].reset_index(drop=True)
    return x_train, y_train, x_pred


def to_catboost_xy(x_train, x_pred):
    cat_cols = [c for c in x_train.columns if not pd.api.types.is_numeric_dtype(x_train[c])]
    x_train = x_train.copy()
    x_pred = x_pred.copy()
    for col in cat_cols:
        x_train[col] = x_train[col].fillna("nan").astype(str)
        x_pred[col] = x_pred[col].fillna("nan").astype(str)
    cat_idx = [x_train.columns.get_loc(c) for c in cat_cols]
    return x_train, x_pred, cat_idx


def to_numeric_xy(x_train, x_pred):
    cat_cols = [c for c in x_train.columns if not pd.api.types.is_numeric_dtype(x_train[c])]
    x_train = x_train.copy()
    x_pred = x_pred.copy()
    for col in cat_cols:
        codes = pd.Categorical(pd.concat([x_train[col], x_pred[col]], axis=0).fillna("nan").astype(str))
        x_train[col] = codes.codes[: len(x_train)]
        x_pred[col] = codes.codes[len(x_train) :]
    return x_train, x_pred


def _default_iters(name: str, fallback: int) -> int:
    if SMOKE_TEST:
        return {
            "catboost": 120,
            "xgboost": 120,
            "lightgbm": 120,
        }.get(name, min(fallback, 120))
    info = BEST_PARAMS.get(name)
    if not info:
        return fallback
    bi = info.get("best_iteration")
    if not bi:
        return fallback
    # add small safety margin since we refit on more data
    return int(bi * 1.15)


def fit_catboost(train_df, pred_df):
    x_train, y_train, x_pred = make_xy(train_df, pred_df)
    x_train, x_pred, cat_idx = to_catboost_xy(x_train, x_pred)
    w = time_decay_weights(train_df)
    tuned = BEST_PARAMS.get("catboost", {}).get("params", {})
    preds = []
    for s in SEEDS:
        params = {
            "loss_function": "MAE",
            "iterations": _default_iters("catboost", 2200),
            "depth": tuned.get("depth", 8),
            "learning_rate": tuned.get("learning_rate", 0.028),
            "l2_leaf_reg": tuned.get("l2_leaf_reg", 8.0),
            "random_strength": tuned.get("random_strength", 1.0),
            "bagging_temperature": tuned.get("bagging_temperature", 1.0),
            "eval_metric": "MAE",
            "random_seed": s,
            "verbose": 0,
            "allow_writing_files": False,
        }
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train, cat_features=cat_idx, sample_weight=w)
        preds.append(model.predict(x_pred))
    return clip(np.mean(preds, axis=0))


def fit_xgb(train_df, pred_df):
    x_train, y_train, x_pred = make_xy(train_df, pred_df)
    x_train, x_pred = to_numeric_xy(x_train, x_pred)
    w = time_decay_weights(train_df)
    tuned = BEST_PARAMS.get("xgboost", {}).get("params", {})
    preds = []
    for s in SEEDS:
        params = {
            "objective": "reg:absoluteerror",
            "n_estimators": _default_iters("xgboost", 2000),
            "max_depth": tuned.get("max_depth", 7),
            "learning_rate": tuned.get("learning_rate", 0.022),
            "subsample": tuned.get("subsample", 0.85),
            "colsample_bytree": tuned.get("colsample_bytree", 0.85),
            "reg_lambda": tuned.get("reg_lambda", 4.0),
            "reg_alpha": tuned.get("reg_alpha", 0.05),
            "min_child_weight": tuned.get("min_child_weight", 1.0),
            "random_state": s,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        model = XGBRegressor(**params)
        pipe = make_pipeline(SimpleImputer(strategy="median"), model)
        fit_kwargs = {} if w is None else {"xgbregressor__sample_weight": w}
        pipe.fit(x_train, y_train, **fit_kwargs)
        preds.append(pipe.predict(x_pred))
    return clip(np.mean(preds, axis=0))


def fit_lgbm(train_df, pred_df):
    x_train, y_train, x_pred = make_xy(train_df, pred_df)
    x_train, x_pred = to_numeric_xy(x_train, x_pred)
    w = time_decay_weights(train_df)
    tuned = BEST_PARAMS.get("lightgbm", {}).get("params", {})
    preds = []
    for s in SEEDS:
        params = {
            "objective": "regression_l1",
            "n_estimators": _default_iters("lightgbm", 2500),
            "max_depth": tuned.get("max_depth", -1),
            "num_leaves": tuned.get("num_leaves", 63),
            "learning_rate": tuned.get("learning_rate", 0.025),
            "subsample": tuned.get("subsample", 0.85),
            "subsample_freq": 1,
            "colsample_bytree": tuned.get("colsample_bytree", 0.85),
            "reg_lambda": tuned.get("reg_lambda", 4.0),
            "reg_alpha": tuned.get("reg_alpha", 0.05),
            "min_child_samples": tuned.get("min_child_samples", 30),
            "random_state": s,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        model.fit(x_train, y_train, sample_weight=w)
        preds.append(model.predict(x_pred))
    return clip(np.mean(preds, axis=0))


def fit_tcn(train_df, pred_df):
    from nn_model import fit_tcn as _fit_tcn
    tuned = BEST_PARAMS.get("tcn", {}).get("params", {})
    return _fit_tcn(
        train_df, pred_df,
        target_col=TARGET, date_col=DATE_COL, repair_col=REPAIR_COL,
        capacity_mw=CAPACITY_MW, seed=SEED,
        epochs=2 if SMOKE_TEST else tuned.get("epochs", 30),
        batch=tuned.get("batch", 256),
        lr=tuned.get("lr", 1e-3),
        dropout=tuned.get("dropout", 0.15),
        verbose=False,
    )


MODELS = {
    "catboost": fit_catboost,
    "lightgbm": fit_lgbm,
    "xgboost": fit_xgb,
    "tcn": fit_tcn,
}


def optimize_weights(preds_per_fold, y_per_fold):
    """Find non-negative weights summing to 1 that minimize MAE across concatenated folds."""
    names = list(preds_per_fold[0].keys())
    P = np.concatenate([np.stack([fold[n] for n in names], axis=1) for fold in preds_per_fold], axis=0)
    Y = np.concatenate(y_per_fold)

    def loss(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        if s <= 0:
            return 1e9
        w = w / s
        pred = clip(P @ w)
        return np.mean(np.abs(Y - pred))

    best = None
    starts = [np.ones(len(names)) / len(names)]
    # also seed with single-model starts
    for i in range(len(names)):
        e = np.zeros(len(names)); e[i] = 1.0
        starts.append(e + 0.01)
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})
        if best is None or r.fun < best.fun:
            best = r
    w = np.clip(best.x, 0, None)
    w = w / w.sum()
    return dict(zip(names, w.tolist())), float(best.fun)


META_RAW_COLS = [
    "pc_weighted_speed",
    "pc_weighted_speed_rho",
    "pc_wind_speed_80m",
    "available_capacity",
    "air_density_ratio",
    "rho_v3_weighted",
    "weighted_speed",
    "hour_sin",
    "hour_cos",
]

RESIDUAL_RAW_COLS = [
    "month",
    "hour_of_day",
    "dayofweek",
    REPAIR_COL,
    "weighted_speed",
    "weighted_speed_sq",
    "weighted_speed_cube",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_gusts_10m",
    "pc_weighted_speed",
    "pc_weighted_speed_rho",
    "pc_wind_speed_80m",
    "pc_wind_speed_120m",
    "available_turbines",
    "available_capacity",
    "weighted_speed_x_available",
    "air_density_ratio",
    "rho_v3_weighted",
    "speed_80_10_diff",
    "speed_120_80_diff",
    "speed_180_120_diff",
    "gust_speed10_diff",
    "speed_120_80_ratio",
    "gust_speed80_ratio",
    "shear_10_80",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def build_meta_X(model_preds: dict, raw_feats: pd.DataFrame, model_names: list) -> np.ndarray:
    base = np.stack([model_preds[n] for n in model_names], axis=1)
    raw = raw_feats[META_RAW_COLS].to_numpy(dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([base, raw], axis=1)


def build_residual_X(model_preds: dict, blend_pred: np.ndarray, raw_feats: pd.DataFrame, model_names: list) -> pd.DataFrame:
    """Features for a small second-stage model that predicts blend residuals."""
    x = pd.DataFrame({"blend_pred": np.asarray(blend_pred, dtype=float)})
    base = np.stack([model_preds[n] for n in model_names], axis=1)
    for i, name in enumerate(model_names):
        x[f"pred_{name}"] = base[:, i]
        x[f"diff_{name}_blend"] = base[:, i] - x["blend_pred"]
    x["pred_min"] = base.min(axis=1)
    x["pred_max"] = base.max(axis=1)
    x["pred_spread"] = x["pred_max"] - x["pred_min"]
    x["pred_std"] = base.std(axis=1)

    for col in RESIDUAL_RAW_COLS:
        if col in raw_feats.columns:
            x[col] = pd.to_numeric(raw_feats[col], errors="coerce")

    if "weighted_speed" in raw_feats.columns:
        speed = pd.to_numeric(raw_feats["weighted_speed"], errors="coerce")
        x["weighted_speed_bin050"] = np.floor(speed * 2.0) / 2.0
        x["weighted_speed_bin100"] = np.floor(speed)
    return x.replace([np.inf, -np.inf], np.nan)


def main():
    train = pd.read_csv(TRAIN_PATH)
    valid = pd.read_csv(VALID_PATH)
    train[DATE_COL] = pd.to_datetime(train[DATE_COL])
    valid[DATE_COL] = pd.to_datetime(valid[DATE_COL])
    if SMOKE_TEST:
        print("[SMOKE] Fast validation mode is active: two folds, short models, smoke output files.")

    # one-time master features (train and valid aligned with input order)
    x_train_full, _, x_valid_full = make_xy(train, valid)

    fold_years = [2024, 2025] if SMOKE_TEST else [2023, 2024, 2025]
    report = {"cv": {}, "models": list(MODELS.keys())}

    preds_per_fold = []
    y_per_fold = []
    raw_per_fold = []
    fold_masks = []
    repair_per_fold = []

    # Q1 2026 has only repair=3 (68.5%) and repair=4 (31.5%)
    valid_repair_max = int(valid[REPAIR_COL].max())  # = 4

    for yr in fold_years:
        hold_mask = (train[DATE_COL].dt.year == yr) & (train[DATE_COL].dt.month <= 3)
        fit = train.loc[~hold_mask].copy()
        hold = train.loc[hold_mask].copy()
        y_hold = hold[TARGET].astype(float).to_numpy()
        repair_hold = hold[REPAIR_COL].to_numpy()
        fold_pred = {}
        fold_scores = {}
        for name, fn in MODELS.items():
            print(f"[CV {yr}] {name}")
            p = fn(fit, hold.drop(columns=[TARGET]))
            fold_pred[name] = p
            fold_scores[name] = scores(y_hold, p)
            print(f"  {name}: {fold_scores[name]}")
        preds_per_fold.append(fold_pred)
        y_per_fold.append(y_hold)
        raw_per_fold.append(x_train_full.loc[hold_mask.values].reset_index(drop=True))
        fold_masks.append(hold_mask.values)
        repair_per_fold.append(repair_hold)
        report["cv"][str(yr)] = fold_scores

    # Stratified blend: weight optimization on rows where repair <= valid_repair_max
    # This matches Q1 2026 distribution (3, 4 dominant; no 5,6,7).
    preds_filtered, y_filtered = [], []
    for fp, y, rp in zip(preds_per_fold, y_per_fold, repair_per_fold):
        mask = rp <= valid_repair_max
        preds_filtered.append({k: v[mask] for k, v in fp.items()})
        y_filtered.append(y[mask])
        print(f"  stratified fold: kept {int(mask.sum())}/{len(y)} rows (repair <= {valid_repair_max})")
    weights, blend_mae = optimize_weights(preds_filtered, y_filtered)
    print("optimized weights (stratified):", weights, "blend MAE:", blend_mae)

    # also compute unstratified for comparison
    w_unstrat, m_unstrat = optimize_weights(preds_per_fold, y_per_fold)
    print(f"  [for reference] unstratified weights: {w_unstrat} MAE {m_unstrat:.4f}")
    report["unstratified_weights"] = w_unstrat
    report["unstratified_blend_mae"] = m_unstrat
    report["valid_repair_max"] = valid_repair_max

    # === ABLATION: try dropping each single model ===
    from itertools import combinations
    all_names = list(MODELS.keys())
    print("\n=== ABLATION (leave-one-model-out at blend level) ===")
    for drop in [None] + list(all_names):
        keep = [n for n in all_names if n != drop]
        if len(keep) < 2:
            continue
        sub_preds = [{n: fp[n] for n in keep} for fp in preds_per_fold]
        w, m = optimize_weights(sub_preds, y_per_fold)
        per_fold = []
        for sp, yh in zip(sub_preds, y_per_fold):
            blended = clip(sum(w[n] * sp[n] for n in w))
            per_fold.append(float(np.mean(np.abs(yh - blended))))
        label = "all models" if drop is None else f"drop {drop}"
        print(f"  {label}: blend MAE {m:.4f} | per-fold {[f'{x:.4f}' for x in per_fold]} | w={ {k: round(v, 3) for k, v in w.items()} }")
    print("=== END ABLATION ===\n")

    blend_by_fold = {}
    for yr, fold_pred, y_hold in zip(fold_years, preds_per_fold, y_per_fold):
        blended = clip(sum(weights[n] * fold_pred[n] for n in weights))
        blend_by_fold[str(yr)] = scores(y_hold, blended)
    report["blend_weights"] = weights
    report["blend_cv"] = blend_by_fold
    report["blend_overall_mae_mw"] = blend_mae

    # ---------- Stacking layer ----------
    model_names = list(MODELS.keys())
    Xs_thin, Xs_wide, ys = [], [], []
    for fold_pred, raw, y in zip(preds_per_fold, raw_per_fold, y_per_fold):
        thin = np.stack([fold_pred[n] for n in model_names], axis=1)
        Xs_thin.append(thin)
        Xs_wide.append(build_meta_X(fold_pred, raw, model_names))
        ys.append(y)
    y_meta = np.concatenate(ys, axis=0)

    def loo_cv(Xs_list):
        Xc = np.concatenate(Xs_list, axis=0)
        per_fold = {}
        for i, yr in enumerate(fold_years):
            fold_idx = np.concatenate([np.full(len(ys[j]), j) for j in range(len(ys))])
            train_mask = fold_idx != i
            meta = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs", fit_intercept=True)
            meta.fit(Xc[train_mask], y_meta[train_mask])
            pred_i = clip(meta.predict(Xs_list[i]))
            per_fold[str(yr)] = scores(ys[i], pred_i)
        return per_fold

    meta_thin_cv = loo_cv(Xs_thin)
    meta_wide_cv = loo_cv(Xs_wide)
    for yr in fold_years:
        print(f"[META-thin {yr}] {meta_thin_cv[str(yr)]}")
        print(f"[META-wide {yr}] {meta_wide_cv[str(yr)]}")

    meta_thin_full = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs", fit_intercept=True)
    meta_thin_full.fit(np.concatenate(Xs_thin, axis=0), y_meta)
    meta_wide_full = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs", fit_intercept=True)
    meta_wide_full.fit(np.concatenate(Xs_wide, axis=0), y_meta)

    report["meta_thin_cv"] = meta_thin_cv
    report["meta_wide_cv"] = meta_wide_cv
    report["meta_thin_coefs"] = {
        **{n: float(c) for n, c in zip(model_names, meta_thin_full.coef_)},
        "intercept": float(meta_thin_full.intercept_),
    }
    report["meta_wide_coefs"] = {
        **{n: float(c) for n, c in zip(model_names, meta_wide_full.coef_[: len(model_names)])},
        **{c: float(v) for c, v in zip(META_RAW_COLS, meta_wide_full.coef_[len(model_names):])},
        "intercept": float(meta_wide_full.intercept_),
    }

    # ---------- Per-month bias correction ----------
    # gather month per OOF row
    months_per_fold = [pd.to_datetime(train.loc[fold_masks[i], DATE_COL]).dt.month.to_numpy() for i in range(len(fold_years))]
    months_all = np.concatenate(months_per_fold)
    y_all = y_meta

    def per_month_bias(pred_per_fold_list):
        preds_all = np.concatenate(pred_per_fold_list)
        resid = y_all - preds_all
        bias = {int(m): float(np.median(resid[months_all == m])) for m in np.unique(months_all)}
        return bias

    def apply_bias(preds, months, bias):
        out = preds.copy().astype(float)
        for m, b in bias.items():
            out[months == m] += b
        return clip(out)

    def bias_frame(raw, y, pred):
        speed = pd.to_numeric(raw["weighted_speed"], errors="coerce")
        frame = pd.DataFrame({
            "y": np.asarray(y, dtype=float),
            "pred": np.asarray(pred, dtype=float),
            "month": pd.to_numeric(raw["month"], errors="coerce").astype("Int64").astype(str),
            "repair": pd.to_numeric(raw[REPAIR_COL], errors="coerce").astype("Int64").astype(str),
            "hour": pd.to_numeric(raw["hour_of_day"], errors="coerce").astype("Int64").astype(str),
            "speed_bin": pd.cut(
                speed,
                bins=[0, 3, 5, 7, 9, 11, 13, 16, 100],
                labels=["0_3", "3_5", "5_7", "7_9", "9_11", "11_13", "13_16", "16_plus"],
                include_lowest=True,
            ).astype(str),
        })
        frame["resid"] = frame["y"] - frame["pred"]
        return frame

    def raw_bias_features(raw):
        speed = pd.to_numeric(raw["weighted_speed"], errors="coerce")
        return pd.DataFrame({
            "month": pd.to_numeric(raw["month"], errors="coerce").astype("Int64").astype(str),
            "repair": pd.to_numeric(raw[REPAIR_COL], errors="coerce").astype("Int64").astype(str),
            "hour": pd.to_numeric(raw["hour_of_day"], errors="coerce").astype("Int64").astype(str),
            "speed_bin": pd.cut(
                speed,
                bins=[0, 3, 5, 7, 9, 11, 13, 16, 100],
                labels=["0_3", "3_5", "5_7", "7_9", "9_11", "11_13", "13_16", "16_plus"],
                include_lowest=True,
            ).astype(str),
        })

    def fit_group_bias(frames, levels, shrink=80.0, min_rows=25):
        train_bias = pd.concat(frames, ignore_index=True)
        global_bias = float(train_bias["resid"].median())
        model = {"global": global_bias, "levels": []}
        for cols in levels:
            grouped = (
                train_bias.groupby(cols, dropna=False)["resid"]
                .agg(rows="size", median="median")
                .reset_index()
            )
            grouped = grouped[grouped["rows"] >= min_rows].copy()
            table = {}
            for row in grouped.itertuples(index=False):
                vals = [str(getattr(row, col)) for col in cols]
                key = "\x1f".join(vals)
                rows = float(getattr(row, "rows"))
                med = float(getattr(row, "median"))
                strength = rows / (rows + shrink)
                table[key] = global_bias + strength * (med - global_bias)
            model["levels"].append({"cols": list(cols), "table": table})
        return model

    def apply_group_bias(preds, raw, model):
        feats = raw_bias_features(raw)
        correction = np.full(len(feats), float(model["global"]), dtype=float)
        filled = np.zeros(len(feats), dtype=bool)
        for level in model["levels"]:
            cols = level["cols"]
            table = level["table"]
            keys = feats[cols].astype(str).agg("\x1f".join, axis=1)
            mapped = keys.map(table)
            mask = mapped.notna().to_numpy() & ~filled
            correction[mask] = mapped.to_numpy(dtype=float)[mask]
            filled |= mask
        return clip(np.asarray(preds, dtype=float) + correction)

    def loo_group_bias(per_fold_preds, levels, shrink=80.0, min_rows=25):
        frames = [bias_frame(raw, y, pred) for raw, y, pred in zip(raw_per_fold, ys, per_fold_preds)]
        corrected = []
        for i in range(len(frames)):
            model = fit_group_bias([frames[j] for j in range(len(frames)) if j != i], levels, shrink=shrink, min_rows=min_rows)
            corrected.append(apply_group_bias(per_fold_preds[i], raw_per_fold[i], model))
        full_model = fit_group_bias(frames, levels, shrink=shrink, min_rows=min_rows)
        avg = float(np.mean([np.mean(np.abs(ys[i] - corrected[i])) for i in range(len(ys))]))
        return corrected, full_model, avg

    # candidate predictions per fold for each strategy
    blend_per_fold = []
    for fold_pred in preds_per_fold:
        blend_per_fold.append(clip(sum(weights[n] * fold_pred[n] for n in weights)))
    Xs_thin_concat = np.concatenate(Xs_thin, axis=0)
    Xs_wide_concat = np.concatenate(Xs_wide, axis=0)
    # for honest CV, use the loo meta models — but we only have the held-out predictions per fold (already in meta_thin_cv etc.). Recompute:
    thin_per_fold, wide_per_fold = [], []
    for i in range(len(fold_years)):
        fold_idx = np.concatenate([np.full(len(ys[j]), j) for j in range(len(ys))])
        train_mask = fold_idx != i
        mt = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs", fit_intercept=True)
        mt.fit(Xs_thin_concat[train_mask], y_meta[train_mask])
        thin_per_fold.append(clip(mt.predict(Xs_thin[i])))
        mw = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs", fit_intercept=True)
        mw.fit(Xs_wide_concat[train_mask], y_meta[train_mask])
        wide_per_fold.append(clip(mw.predict(Xs_wide[i])))

    # ---------- Residual correction on top of blend ----------
    X_res_per_fold = [
        build_residual_X(fold_pred, blend_pred, raw, model_names)
        for fold_pred, blend_pred, raw in zip(preds_per_fold, blend_per_fold, raw_per_fold)
    ]
    y_res_per_fold = [y - pred for y, pred in zip(ys, blend_per_fold)]

    def make_residual_model(seed=SEED):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                objective="regression_l1",
                n_estimators=80 if SMOKE_TEST else 700,
                learning_rate=0.025,
                num_leaves=15,
                max_depth=4,
                min_child_samples=90,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_lambda=8.0,
                reg_alpha=2.0,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
        )

    residual_per_fold = []
    residual_cv = {}
    for i, yr in enumerate(fold_years):
        X_fit_res = pd.concat([X_res_per_fold[j] for j in range(len(fold_years)) if j != i], ignore_index=True)
        y_fit_res = np.concatenate([y_res_per_fold[j] for j in range(len(fold_years)) if j != i])
        model_res = make_residual_model(SEED + i)
        model_res.fit(X_fit_res, y_fit_res)
        correction = np.clip(model_res.predict(X_res_per_fold[i]), -12.0, 12.0)
        pred_i = clip(blend_per_fold[i] + correction)
        residual_per_fold.append(pred_i)
        residual_cv[str(yr)] = scores(ys[i], pred_i)
        print(f"[RESIDUAL {yr}] {residual_cv[str(yr)]}")

    X_res_all = pd.concat(X_res_per_fold, ignore_index=True)
    y_res_all = np.concatenate(y_res_per_fold)
    residual_model_full = make_residual_model(SEED)
    residual_model_full.fit(X_res_all, y_res_all)
    report["residual_cv"] = residual_cv

    # ---------- Regime-aware bias correction ----------
    regime_bias_specs = {
        "blend+bias_speed": [["speed_bin"]],
        "blend+bias_repair_speed": [["repair", "speed_bin"], ["speed_bin"]],
        "blend+bias_month_speed": [["month", "speed_bin"], ["speed_bin"], ["month"]],
        "blend+bias_month_repair_speed": [["month", "repair", "speed_bin"], ["repair", "speed_bin"], ["speed_bin"], ["month"]],
    }
    regime_bias_results = {}
    for label, levels in regime_bias_specs.items():
        corrected, model, avg = loo_group_bias(blend_per_fold, levels, shrink=120.0, min_rows=35)
        regime_bias_results[label] = {"fold_preds": corrected, "model": model, "avg_mae": avg}
        print(f"{label} CV avg MAE: {avg:.4f}")
    blend_month_bias_folds, blend_month_bias_model, blend_month_bias_avg = loo_group_bias(
        blend_per_fold, [["month"]], shrink=0.0, min_rows=20
    )
    report["regime_bias"] = {
        label: {
            "avg_mae_mw": info["avg_mae"],
            "levels": regime_bias_specs[label],
        }
        for label, info in regime_bias_results.items()
    }

    bias_blend = per_month_bias(blend_per_fold)
    bias_thin = per_month_bias(thin_per_fold)
    bias_wide = per_month_bias(wide_per_fold)
    bias_residual = per_month_bias(residual_per_fold)
    print("bias blend:", bias_blend)
    print("bias thin :", bias_thin)
    print("bias wide :", bias_wide)
    print("bias residual:", bias_residual)

    # ---------- Isotonic calibration on blend (LOO-fold) ----------
    iso_per_fold = []
    for i in range(len(fold_years)):
        train_b = np.concatenate([blend_per_fold[j] for j in range(len(fold_years)) if j != i])
        train_y = np.concatenate([ys[j] for j in range(len(fold_years)) if j != i])
        iso_i = IsotonicRegression(y_min=0.0, y_max=CAPACITY_MW, out_of_bounds="clip")
        iso_i.fit(train_b, train_y)
        iso_per_fold.append(clip(iso_i.predict(blend_per_fold[i])))
    iso_avg = float(np.mean([np.mean(np.abs(ys[i] - iso_per_fold[i])) for i in range(len(fold_years))]))
    iso_full = IsotonicRegression(y_min=0.0, y_max=CAPACITY_MW, out_of_bounds="clip")
    iso_full.fit(np.concatenate(blend_per_fold), y_meta)
    print(f"blend+iso CV avg MAE: {iso_avg:.4f}")

    def eval_with_bias(per_fold_preds, bias):
        avg = 0.0
        for i, yr in enumerate(fold_years):
            corrected = apply_bias(per_fold_preds[i], months_per_fold[i], bias)
            avg += float(np.mean(np.abs(ys[i] - corrected)))
        return avg / len(fold_years)

    blend_bias_avg = blend_month_bias_avg
    thin_bias_avg = eval_with_bias(thin_per_fold, bias_thin)
    wide_bias_avg = eval_with_bias(wide_per_fold, bias_wide)
    residual_bias_avg = eval_with_bias(residual_per_fold, bias_residual)

    blend_avg = float(np.mean([v["mae_mw"] for v in blend_by_fold.values()]))
    thin_avg = float(np.mean([v["mae_mw"] for v in meta_thin_cv.values()]))
    wide_avg = float(np.mean([v["mae_mw"] for v in meta_wide_cv.values()]))
    residual_avg = float(np.mean([v["mae_mw"] for v in residual_cv.values()]))
    options = {
        "blend": blend_avg,
        "blend+bias": blend_bias_avg,
        **{label: info["avg_mae"] for label, info in regime_bias_results.items()},
        "blend+residual": residual_avg,
        "blend+residual+bias": residual_bias_avg,
        "blend+iso": iso_avg,
        "meta_thin": thin_avg,
        "meta_thin+bias": thin_bias_avg,
        "meta_wide": wide_avg,
        "meta_wide+bias": wide_bias_avg,
    }
    chosen = min(options, key=lambda k: options[k])
    report["selection"] = {**{f"{k}_avg_mae_mw": v for k, v in options.items()}, "chosen": chosen}
    report["bias"] = {"blend": bias_blend, "meta_thin": bias_thin, "meta_wide": bias_wide, "residual": bias_residual}
    for k, v in options.items():
        print(f"  {k}: {v:.4f}")
    print(f"chosen -> {chosen}")

    fold_candidates = {
        "blend": blend_per_fold,
        "blend+bias": blend_month_bias_folds,
        **{label: info["fold_preds"] for label, info in regime_bias_results.items()},
        "blend+residual": residual_per_fold,
        "blend+residual+bias": [apply_bias(residual_per_fold[i], months_per_fold[i], bias_residual) for i in range(len(fold_years))],
        "blend+iso": iso_per_fold,
        "meta_thin": thin_per_fold,
        "meta_thin+bias": [apply_bias(thin_per_fold[i], months_per_fold[i], bias_thin) for i in range(len(fold_years))],
        "meta_wide": wide_per_fold,
        "meta_wide+bias": [apply_bias(wide_per_fold[i], months_per_fold[i], bias_wide) for i in range(len(fold_years))],
    }

    def error_slices(per_fold_preds):
        frames = []
        for yr, raw, y, pred in zip(fold_years, raw_per_fold, ys, per_fold_preds):
            frame = pd.DataFrame({
                "fold_year": yr,
                "y": y,
                "pred": pred,
                "abs_err": np.abs(y - pred),
                "month": pd.to_numeric(raw["month"], errors="coerce"),
                "hour": pd.to_numeric(raw["hour_of_day"], errors="coerce"),
                "repair": pd.to_numeric(raw[REPAIR_COL], errors="coerce"),
                "weighted_speed": pd.to_numeric(raw["weighted_speed"], errors="coerce"),
            })
            frames.append(frame)
        err = pd.concat(frames, ignore_index=True)
        err["speed_bin"] = pd.cut(err["weighted_speed"], bins=[0, 3, 5, 7, 9, 11, 13, 16, 100], include_lowest=True)
        err["hour_bin"] = pd.cut(err["hour"], bins=[-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])

        def agg(col, top=None):
            g = (
                err.groupby(col, dropna=False)["abs_err"]
                .agg(rows="size", mae_mw="mean")
                .reset_index()
            )
            g = g[g["rows"] >= 20].copy()
            g[col] = g[col].astype(str)
            g["mae_mw"] = g["mae_mw"].astype(float)
            g["rows"] = g["rows"].astype(int)
            if top is not None:
                g = g.sort_values("mae_mw", ascending=False).head(top)
            return g.to_dict(orient="records")

        return {
            "overall_mae_mw": float(err["abs_err"].mean()),
            "by_fold_year": agg("fold_year"),
            "by_month": agg("month"),
            "by_repair": agg("repair"),
            "by_hour_bin": agg("hour_bin"),
            "worst_speed_bins": agg("speed_bin", top=8),
        }

    report["error_analysis"] = {
        "chosen_strategy": chosen,
        "chosen": error_slices(fold_candidates[chosen]),
        "blend": error_slices(blend_per_fold),
    }

    # ---------- Final fit on full train ----------
    final_preds = {}
    for name, fn in MODELS.items():
        print(f"[FINAL] {name}")
        final_preds[name] = fn(train, valid)

    blend_final = clip(sum(weights[n] * final_preds[n] for n in weights))
    thin_valid = np.stack([final_preds[n] for n in model_names], axis=1)
    wide_valid = build_meta_X(final_preds, x_valid_full, model_names)
    thin_final = clip(meta_thin_full.predict(thin_valid))
    wide_final = clip(meta_wide_full.predict(wide_valid))
    residual_valid_X = build_residual_X(final_preds, blend_final, x_valid_full, model_names)
    residual_correction = np.clip(residual_model_full.predict(residual_valid_X), -12.0, 12.0)
    residual_final = clip(blend_final + residual_correction)

    valid_months = pd.to_datetime(valid[DATE_COL]).dt.month.to_numpy()
    candidates = {
        "blend": blend_final,
        "blend+bias": apply_group_bias(blend_final, x_valid_full, blend_month_bias_model),
        **{
            label: apply_group_bias(blend_final, x_valid_full, info["model"])
            for label, info in regime_bias_results.items()
        },
        "blend+residual": residual_final,
        "blend+residual+bias": apply_bias(residual_final, valid_months, bias_residual),
        "blend+iso": clip(iso_full.predict(blend_final)),
        "meta_thin": thin_final,
        "meta_thin+bias": apply_bias(thin_final, valid_months, bias_thin),
        "meta_wide": wide_final,
        "meta_wide+bias": apply_bias(wide_final, valid_months, bias_wide),
    }
    final = candidates[chosen]

    submission_path = ROOT / "submission_smoke.csv" if SMOKE_TEST else SUBMISSION_PATH
    report_path = ROOT / "report_smoke.json" if SMOKE_TEST else REPORT_PATH
    pd.DataFrame({TARGET: final}).to_csv(submission_path, index=False)

    report["submission"] = {
        "path": str(submission_path),
        "rows": int(len(final)),
        "min": float(final.min()),
        "max": float(final.max()),
        "mean": float(final.mean()),
        "strategy": report["selection"]["chosen"],
    }
    if SMOKE_TEST:
        report["smoke_test"] = True
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
