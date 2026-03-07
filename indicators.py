"""
Индикаторы для МТС: EMA, VWAP, ATR, RSI, ADX.
Используются в бэктестах для сигналов входа и выхода.
"""
import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """Экспоненциальная скользящая средняя."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI). Значение 0-100."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def vwap_from_ohlcv(df: pd.DataFrame) -> pd.Series:
    """
    VWAP по типичной цене (H+L+C)/3 и объёму.
    Для дневного VWAP обычно считают накопленный от начала дня;
    здесь — скользящий VWAP на окне (упрощение для бэктеста).
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    return (typical * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)


def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """VWAP сброс в начале каждого дня (если индекс — datetime)."""
    chunks = []
    for _, g in df.groupby(df.index.date, sort=False):
        v = g["volume"]
        t = (g["high"] + g["low"] + g["close"]) / 3.0
        vwap_g = (t * v).cumsum() / v.cumsum().replace(0, np.nan)
        chunks.append(vwap_g)
    if not chunks:
        return pd.Series(index=df.index, dtype=float)
    values = pd.concat(chunks).to_numpy()
    return pd.Series(values, index=df.index, dtype=float)


def add_indicators(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 60,
    ema_trend: int = 100,
    use_daily_vwap: bool = True,
    atr_period: int = 14,
    rsi_period: int = 14,
    vol_ma_period: int = 20,
    adx_period: int = 14,
    slope_period: int = 3,
) -> pd.DataFrame:
    """
    Добавляет к OHLCV колонки:
    ema_fast, ema_slow, ema_trend, vwap, atr, rsi, vol_ma, adx,
    ema_fast_slope (для фильтра направления).
    """
    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    df["ema_trend"] = ema(df["close"], ema_trend)
    if use_daily_vwap and hasattr(df.index, "date"):
        df["vwap"] = vwap_daily(df)
    else:
        df["vwap"] = vwap_from_ohlcv(df)
    if atr_period > 0:
        df["atr"] = atr(df, atr_period)
    if rsi_period > 0:
        df["rsi"] = rsi(df["close"], rsi_period)
    if vol_ma_period > 0:
        df["vol_ma"] = df["volume"].rolling(vol_ma_period, min_periods=1).mean()
    if adx_period > 0:
        df["adx"] = adx(df, adx_period)
    if slope_period > 0:
        df["ema_fast_slope"] = ema_slope(df["ema_fast"], slope_period)
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).
    ADX > 20 — трендовый рынок, ADX < 20 — боковик.
    Возвращает только ADX (без +DI/-DI).
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed with Wilder (EWM)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def ema_slope(series: pd.Series, period: int = 3) -> pd.Series:
    """
    Наклон EMA: разность текущего и period баров назад, нормированная на значение.
    Положительный → растёт, отрицательный → падает.
    """
    return (series - series.shift(period)) / series.shift(period).replace(0, np.nan)


def long_signal(row) -> bool:
    """Сигнал на покупку: быстрая EMA выше медленной и цена выше VWAP."""
    return row["ema_fast"] > row["ema_slow"] and row["close"] > row["vwap"]


def short_signal(row) -> bool:
    """Сигнал на продажу: быстрая EMA ниже медленной и цена ниже VWAP."""
    return row["ema_fast"] < row["ema_slow"] and row["close"] < row["vwap"]


def signal_long_prev_short(df: pd.DataFrame, i: int) -> bool:
    """Пересечение вверх: сейчас лонг, на предыдущем баре был шорт."""
    if i < 1:
        return False
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    return long_signal(curr) and (short_signal(prev) or prev["ema_fast"] <= prev["ema_slow"] and prev["close"] <= prev["vwap"])


def signal_short_prev_long(df: pd.DataFrame, i: int) -> bool:
    """Пересечение вниз: сейчас шорт, на предыдущем баре был лонг."""
    if i < 1:
        return False
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    return short_signal(curr) and (long_signal(prev) or (prev["ema_fast"] >= prev["ema_slow"] and prev["close"] >= prev["vwap"]))
