"""Compact TCN model for wind power forecasting.

Takes 24-hour window of key meteo features, predicts power at the last hour.
Designed to plug into solution.py as a 3rd base model alongside CatBoost + LightGBM.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WINDOW = 24  # hours of context

OPENMETEO_EXTRA_COLS = [
    "relative_humidity_2m", "dew_point_2m", "surface_pressure",
    "cloud_cover_mid", "cloud_cover_high", "shortwave_radiation",
    "vapour_pressure_deficit", "soil_temperature_0_to_7cm",
    "is_day", "weather_code",
]

FEAT_COLS = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_gusts_10m",
    "wind_direction_80m_sin", "wind_direction_80m_cos",
    "wind_direction_120m_sin", "wind_direction_120m_cos",
    "temperature_80m", "pressure_msl",
    "available_turbines", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "air_density_ratio", "pc_weighted_speed_norm",
]  # OpenMeteo extras tried but hurt TCN on LB — reverted

_OM_EXTRA_PATH = Path(__file__).resolve().parent / "openmeteo_extra.csv"
if _OM_EXTRA_PATH.exists():
    _OM_EXTRA = pd.read_csv(_OM_EXTRA_PATH)
    _OM_EXTRA["time"] = pd.to_datetime(_OM_EXTRA["time"])
    _OM_EXTRA = _OM_EXTRA.set_index("time")
else:
    _OM_EXTRA = None


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=0)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # causal padding on left
        h = F.pad(x, (self.pad, 0))
        h = F.relu(self.conv1(h))
        h = F.pad(h, (self.pad, 0))
        h = self.conv2(h)
        h = self.drop(h)
        return F.relu(h + self.proj(x))


class TCN(nn.Module):
    def __init__(self, n_features: int, channels: List[int] = (48, 48, 48), kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, ch, kernel, dilation=2 ** i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        h = self.tcn(x.transpose(1, 2))
        # take last time step
        h = h[:, :, -1]
        return self.head(h).squeeze(-1)


def _prepare_features(df: pd.DataFrame, date_col: str, repair_col: str) -> pd.DataFrame:
    """Build the TCN feature set on a time-sorted concatenated frame."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["_dt"] = dt
    out = out.sort_values("_dt").reset_index(drop=True)

    # merge OpenMeteo extra variables
    if _OM_EXTRA is not None:
        dt_sorted = pd.to_datetime(out[date_col])
        for col in OPENMETEO_EXTRA_COLS:
            if col not in out.columns and col in _OM_EXTRA.columns:
                out[col] = dt_sorted.map(_OM_EXTRA[col]).astype(float).fillna(0.0)

    out["hour"] = dt.dt.hour
    out["month"] = dt.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    for col in ["wind_direction_80m", "wind_direction_120m"]:
        deg = (out[col] * 1000.0) % 360.0
        rad = np.deg2rad(deg)
        out[f"{col}_sin"] = np.sin(rad)
        out[f"{col}_cos"] = np.cos(rad)

    out["available_turbines"] = 26 - out[repair_col]

    # air density
    t_k = out["temperature_80m"].fillna(out["temperature_80m"].median()) + 273.15
    p_pa = out["pressure_msl"].fillna(out["pressure_msl"].median()) * 100.0
    out["air_density"] = p_pa / (287.05 * t_k)
    out["air_density_ratio"] = out["air_density"] / 1.225

    # simple power curve normalized
    def pc(v):
        v = np.clip(v, 0, None)
        ramp = np.clip((v - 3.0) / 9.0, 0.0, 1.0) ** 3
        out_p = np.where(v < 25, np.minimum(ramp, 1.0), 0.0)
        return out_p
    out["pc_weighted_speed_norm"] = pc(out["wind_speed_80m"].fillna(out["wind_speed_80m"].median()))

    return out


def _build_windows(feats: pd.DataFrame, n_train: int, window: int = WINDOW):
    """Return (X_seq, y_train, x_pred_seq, train_mask).
    X_seq is shape (n_train_valid, window, n_features) for rows with full history.
    """
    X = feats[FEAT_COLS].astype(float).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_total = len(X)
    # build windows: for each row i with i >= window-1, take rows [i-window+1 : i+1]
    # too memory-heavy if eager. Use strided trick.
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(X, window_shape=window, axis=0)  # (n_total - window + 1, n_features, window)
    sw = sw.transpose(0, 2, 1).copy()  # (n_valid, window, n_features)
    # row index for sample i corresponds to original index i + window - 1
    start_idx = window - 1
    return sw, start_idx


def fit_tcn(train_df, pred_df, **kwargs):
    return fit_nn(train_df, pred_df, **kwargs)


def fit_nn(train_df: pd.DataFrame, pred_df: pd.DataFrame, target_col: str, date_col: str, repair_col: str,
           capacity_mw: float, seed: int = 42, epochs: int = 30, batch: int = 256,
           lr: float = 1e-3, dropout: float = 0.15, verbose: bool = False, window: int | None = None) -> np.ndarray:
    """Train TCN on train_df, predict for pred_df rows."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tr = len(train_df)
    n_pr = len(pred_df)
    combined = pd.concat(
        [train_df.drop(columns=[target_col], errors="ignore"), pred_df.drop(columns=[target_col], errors="ignore")],
        ignore_index=True,
    )
    combined["_orig_pos"] = np.arange(len(combined))
    feats = _prepare_features(combined, date_col=date_col, repair_col=repair_col)
    # feats is sorted by _dt; _orig_pos preserved
    orig_pos = feats["_orig_pos"].to_numpy()

    win = window if window is not None else WINDOW
    sw, start_idx = _build_windows(feats, n_tr, window=win)
    # sw row k corresponds to feats row (start_idx + k) i.e. orig_pos[start_idx + k]
    sw_orig_pos = orig_pos[start_idx:]
    n_samples = len(sw)

    # Map train rows: those with orig_pos < n_tr in feats sorted order
    is_train = sw_orig_pos < n_tr
    # y values aligned with sw rows
    target_full = combined.iloc[orig_pos[start_idx:]][target_col].to_numpy() if target_col in combined.columns else np.full(n_samples, np.nan)
    # but target only exists for train rows (rows from train_df)
    # combined was train_df + pred_df concatenated; pred_df rows have no target column.
    # Build y per sample using train_df[target] indexed by orig_pos
    train_targets = train_df[target_col].astype(float).to_numpy()
    y_per_sample = np.full(n_samples, np.nan)
    for i in range(n_samples):
        op = sw_orig_pos[i]
        if op < n_tr:
            y_per_sample[i] = train_targets[op]

    train_mask = is_train & ~np.isnan(y_per_sample)
    pred_mask = sw_orig_pos >= n_tr  # rows that belong to pred_df

    X_train = torch.tensor(sw[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y_per_sample[train_mask], dtype=torch.float32)
    X_pred = torch.tensor(sw[pred_mask], dtype=torch.float32)
    pred_orig_pos = sw_orig_pos[pred_mask] - n_tr  # row index in pred_df

    # standardize per-feature using train stats
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True).clamp(min=1e-3)
    X_train = (X_train - mean) / std
    X_pred = (X_pred - mean) / std

    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    model = TCN(n_features=len(FEAT_COLS), channels=(48, 48, 48), kernel=3, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = F.l1_loss(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            n += len(yb)
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  ep{ep}: train MAE {total / max(n, 1):.4f}")

    # predict
    model.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(X_pred), 512):
            chunk = X_pred[i:i + 512]
            all_preds.append(model(chunk).numpy())
        pred_y = np.concatenate(all_preds)

    # assemble final predictions, aligned to pred_df row order
    final = np.full(n_pr, np.nan)
    final[pred_orig_pos] = pred_y
    # fill any missing (rows in pred_df that didn't have full 24h history because they were too early)
    # this happens only for the very first few rows if pred_df is contiguous after train_df
    if np.isnan(final).any():
        global_mean = float(np.nanmean(pred_y)) if not np.all(np.isnan(pred_y)) else 0.0
        final = np.where(np.isnan(final), global_mean, final)
    return np.clip(final, 0.0, capacity_mw)
