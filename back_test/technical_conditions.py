"""Causal daily technical features and declarative buy conditions.

Compute on full history; the replay engine shifts these completed-bar features
one session before consulting them for a fill. No LLM or network calls here.
"""
from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class BuyConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_volume_ratio: float = Field(default=0, ge=0, allow_inf_nan=False)
    macd: Literal["off", "bullish", "rising"] = "off"
    kdj: Literal["off", "bullish", "cross_up"] = "off"
    boll: Literal["off", "above_mid", "inside_bands"] = "off"
    hold_above_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    close_hold_days: int = Field(default=2, ge=1, le=10)


PROFILES = {
    "price_only": {},
    "volume_macd": {"min_volume_ratio": 1.0, "macd": "bullish"},
    "volume_macd_kdj_boll": {"min_volume_ratio": 1.0, "macd": "bullish", "kdj": "bullish", "boll": "above_mid"},
}


def validate_execution_conditions(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {"entry", "add"}:
        raise ValueError("execution_conditions must contain only entry/add objects")
    return {role: BuyConditions.model_validate(conditions).model_dump() for role, conditions in value.items()}


def technical_features(history: pd.DataFrame) -> pd.DataFrame:
    """MACD(12,26,9), KDJ(9,3,3; K/D seed 50), BOLL(20,2; ddof=0).

    Require 60 consecutive valid OHLC bars for these execution features. This
    is an explicit warm-up convention, not a claim of exact TA-Lib parity.
    Volume ratio uses today's volume / preceding 20 bars' average (excludes
    today's volume from the denominator).
    """
    df = history.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    data = df.reindex(columns=["Open", "High", "Low", "Close", "Volume"]).apply(pd.to_numeric, errors="coerce")
    data = data.where(data.apply(lambda col: col.map(lambda v: pd.notna(v) and math.isfinite(v))))
    valid = (data[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    valid &= data.High.ge(data[["Open", "Close", "Low"]].max(axis=1)) & data.Low.le(data[["Open", "Close"]].min(axis=1))
    close = data.Close.where(valid)
    dif = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    lo = data.Low.where(valid).rolling(9).min()
    hi = data.High.where(valid).rolling(9).max()
    rsv = (100 * (close - lo) / (hi - lo)).where(hi != lo, 50.0).where(lo.notna() & hi.notna())
    ks, ds = [], []
    k = d = 50.0
    for value in rsv:
        if pd.isna(value):
            k = d = 50.0
            ks.append(float("nan")); ds.append(float("nan"))
        else:
            k = 2 / 3 * k + value / 3
            d = 2 / 3 * d + k / 3
            ks.append(k); ds.append(d)
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std(ddof=0)
    volume = data.Volume.where(data.Volume >= 0)
    average = volume.shift(1).rolling(20, min_periods=20).mean()
    out = pd.DataFrame({"Date":df.Date, "close":close, "macd":dif, "macd_signal":dea,
                        "macd_hist":dif-dea, "kdj_k":ks, "kdj_d":ds,
                        "boll_mid":mid, "boll_upper":mid+2*sd, "boll_lower":mid-2*sd,
                        "volume_ratio":(volume/average).where(average > 0)})
    for days in range(1, 11):
        out[f"close_min_{days}"] = close.rolling(days, min_periods=days).min()
    out["kdj_j"] = 3*out.kdj_k - 2*out.kdj_d
    out["macd_hist_previous"] = out.macd_hist.shift(1)
    out["kdj_cross_up"] = ((out.kdj_k > out.kdj_d) & (out.kdj_k.shift(1) <= out.kdj_d.shift(1))).astype(float)
    warm = valid.rolling(60, min_periods=60).sum().eq(60)
    out.loc[~warm, out.columns != "Date"] = float("nan")
    return out


def conditions_pass(conditions: dict, features: dict | None) -> tuple[bool, str]:
    c = BuyConditions.model_validate(conditions)
    checks = []
    if c.min_volume_ratio > 0:
        checks.append(("volume_ratio", lambda x: x >= c.min_volume_ratio))
    if c.macd == "bullish":
        checks.append(("macd_hist", lambda x: x > 0))
    elif c.macd == "rising":
        checks.extend([("macd_hist", lambda x: True), ("macd_hist_previous", lambda x: True)])
    if c.kdj != "off":
        checks.extend([("kdj_k", lambda x: True), ("kdj_d", lambda x: True)])
        if c.kdj == "cross_up":
            checks.append(("kdj_cross_up", lambda x: x == 1))
    if c.boll != "off":
        checks.extend([("close", lambda x: True), ("boll_mid", lambda x: True)])
        if c.boll == "inside_bands":
            checks.extend([("boll_lower", lambda x: True), ("boll_upper", lambda x: True)])
    if c.hold_above_price is not None:
        checks.append((f"close_min_{c.close_hold_days}", lambda x: x >= c.hold_above_price))
    f = features or {}
    for name, test in checks:
        value = f.get(name)
        if value is None or not math.isfinite(value):
            return False, "missing:" + name
        if not test(value):
            return False, "unconfirmed:" + name
    if c.macd == "rising" and f["macd_hist"] <= f["macd_hist_previous"]:
        return False, "unconfirmed:macd_rising"
    if c.kdj != "off" and f["kdj_k"] <= f["kdj_d"]:
        return False, "unconfirmed:kdj"
    if c.boll == "above_mid" and f["close"] < f["boll_mid"]:
        return False, "unconfirmed:boll_mid"
    if c.boll == "inside_bands" and not f["boll_lower"] <= f["close"] <= f["boll_upper"]:
        return False, "unconfirmed:boll_bands"
    return True, "confirmed"
