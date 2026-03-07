"""
Загрузка исторических данных для бэктестов.
Приоритет: официальный Finam API v1 (GET /v1/instruments/{symbol}/bars),
символ в формате ticker@mic (SBER@MISX, YDEX@MISX). Запасной вариант — Yahoo.
Результат: data/YNDX_15m.csv, data/YNDX_1h.csv.
"""
import io
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.getLogger("finam").setLevel(logging.WARNING)

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent

FINAM_TIMEFRAME = {
    "15m": "TIME_FRAME_M15",
    "1h": "TIME_FRAME_H1",
    "1d": "TIME_FRAME_D",
}
FINAM_CHUNK_DAYS = {"15m": 30, "1h": 30, "1d": 365}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def load_dotenv():
    try:
        from dotenv import load_dotenv as _load
        _load(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def get_jwt(secret: str) -> str | None:
    """Получить JWT по secret (POST /v1/sessions). Нужен для Bearer-авторизации bars."""
    try:
        r = requests.post(
            "https://api.finam.ru/v1/sessions",
            json={"secret": secret.strip()},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("token") or data.get("accessToken")
    except Exception:
        return None


def download_finam_api_v1(api_key: str, symbol: str, interval: str,
                           from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Finam API v1: GET /v1/instruments/{symbol}/bars.
    Требуется Authorization: Bearer <JWT> (JWT получается из secret через POST /v1/sessions).
    symbol — ticker@mic (например SBER@MISX, YDEX@MISX).
    """
    if not api_key or not api_key.strip():
        return pd.DataFrame()
    tf = FINAM_TIMEFRAME.get(interval)
    if tf is None:
        return pd.DataFrame()
    jwt = get_jwt(api_key)
    if not jwt:
        logging.warning("Finam API: failed to get JWT (check FINAM_API_KEY in .env).")
        return pd.DataFrame()
    chunk_days = FINAM_CHUNK_DAYS.get(interval, 30)
    headers = {"Authorization": f"Bearer {jwt}", "Accept": "application/json"}
    all_rows = []
    start = from_date
    end = min(to_date, datetime.now())
    while start < end:
        chunk_end = min(start + timedelta(days=chunk_days), end)
        params = {
            "timeframe": tf,
            "interval.startTime": start.strftime("%Y-%m-%dT00:00:00Z"),
            "interval.endTime": chunk_end.strftime("%Y-%m-%dT23:59:59Z"),
        }
        url = f"https://api.finam.ru/v1/instruments/{symbol}/bars"
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code != 200:
                try:
                    err = r.json()
                    msg = err.get("message") or err.get("error") or r.text[:200]
                except Exception:
                    msg = r.text[:200]
                logging.warning("Finam bars %s: HTTP %s — %s", symbol, r.status_code, msg)
                return pd.DataFrame()
            data = r.json()
        except Exception as e:
            logging.warning("Finam bars %s: request failed — %s", symbol, e)
            return pd.DataFrame()
        bars = data.get("bars") or []
        for b in bars:
            ts = b.get("timestamp")
            o = b.get("open") or {}
            h = b.get("high") or {}
            lo = b.get("low") or {}
            c = b.get("close") or {}
            v = b.get("volume") or {}
            try:
                open_v = float((o.get("value") if isinstance(o, dict) else o) or 0)
                high_v = float((h.get("value") if isinstance(h, dict) else h) or open_v)
                low_v = float((lo.get("value") if isinstance(lo, dict) else lo) or open_v)
                close_v = float((c.get("value") if isinstance(c, dict) else c) or open_v)
                vol_v = int(float((v.get("value") if isinstance(v, dict) else v) or 0))
            except (TypeError, ValueError):
                continue
            all_rows.append({
                "datetime": pd.to_datetime(ts),
                "open": open_v, "high": high_v, "low": low_v, "close": close_v,
                "volume": vol_v,
            })
        start = chunk_end
        if not bars:
            if not all_rows:
                logging.warning("Finam bars %s %s: API 200 but no bars in range %s–%s",
                                symbol, interval, params.get("interval.startTime"), params.get("interval.endTime"))
            break
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows).set_index("datetime").sort_index()


def download_finam_library(ticker: str, interval: str,
                           from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Загрузка через finam-export (как в https://github.com/ffeast/finam-export и smart-lab).
    Важно: market берём из результата lookup — Market(contract.market), не только SHARES.
    """
    import time
    try:
        from finam import Exporter, Market, Timeframe
    except ImportError:
        return pd.DataFrame()

    tf_map = {"15m": Timeframe.MINUTES15, "1h": Timeframe.HOURLY}
    tf = tf_map.get(interval)
    if tf is None:
        return pd.DataFrame()

    exporter = Exporter()
    start_date = from_date.date()
    end_date = min(to_date.date(), datetime.now().date())

    # lookup: сначала по code, если пусто — по name (как в статьях)
    table = None
    for lookup_kw in [{"code": ticker, "market": Market.SHARES},
                      {"name": ticker, "market": Market.SHARES}]:
        try:
            table = exporter.lookup(**lookup_kw)
            if table is not None and len(table) > 0:
                break
        except Exception:
            continue
    if table is None or len(table) == 0:
        return pd.DataFrame()

    # как в скрипте finam-download: contract из первой строки, market из контракта
    contract = table.reset_index().iloc[0]
    asset_id = int(contract["id"])
    market = Market(int(contract["market"]))

    try:
        time.sleep(1)
        raw = exporter.download(
            asset_id,
            market,
            start_date=start_date,
            end_date=end_date,
            timeframe=tf,
            delay=2,
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = [c for c in raw.columns if "DATE" in c.upper()][:1]
    time_col = [c for c in raw.columns if "TIME" in c.upper()][:1]
    open_col = [c for c in raw.columns if "OPEN" in c.upper()][:1]
    high_col = [c for c in raw.columns if "HIGH" in c.upper()][:1]
    low_col = [c for c in raw.columns if "LOW" in c.upper()][:1]
    close_col = [c for c in raw.columns if "CLOSE" in c.upper()][:1]
    vol_col = [c for c in raw.columns if "VOL" in c.upper()][:1]
    if not (date_col and time_col and open_col and close_col):
        return pd.DataFrame()

    raw["datetime"] = pd.to_datetime(
        raw[date_col[0]].astype(str) + " " + raw[time_col[0]].astype(str).str.zfill(8),
        format="%Y%m%d %H:%M:%S",
        errors="coerce",
    )
    raw = raw.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    out = pd.DataFrame(index=raw.index)
    out["open"] = pd.to_numeric(raw[open_col[0]], errors="coerce")
    out["high"] = pd.to_numeric(raw[high_col[0]], errors="coerce") if high_col else out["open"]
    out["low"] = pd.to_numeric(raw[low_col[0]], errors="coerce") if low_col else out["open"]
    out["close"] = pd.to_numeric(raw[close_col[0]], errors="coerce")
    out["volume"] = pd.to_numeric(raw[vol_col[0]], errors="coerce").fillna(0).astype(int) if vol_col else 0
    out = out.dropna(subset=["open", "close"])
    return out


def load_manual_finam_txt(path: Path) -> pd.DataFrame | None:
    """
    Читает выгрузку Finam в формате <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>.
    DATE = YYMMDD (250801 → 2025-08-01), TIME = HHMMSS.
    Возвращает DataFrame с индексом datetime и колонками open, high, low, close, volume.
    """
    for enc in ("utf-8", "cp1251", "cp866", None):
        try:
            raw = pd.read_csv(path, sep=",", skipinitialspace=True, encoding=enc)
            break
        except Exception:
            continue
    else:
        return None
    raw.columns = [str(c).strip().strip("<>").strip("\ufeff") for c in raw.columns]
    need = {"PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"}
    if not need.issubset(raw.columns):
        return None
    raw = raw.dropna(subset=["OPEN", "CLOSE"], how="any")
    if raw.empty:
        return None
    # DATE: YYMMDD → 20YY-MM-DD
    date_s = raw["DATE"].astype(int).astype(str).str.zfill(6)
    raw["_dt_date"] = "20" + date_s.str[:2] + "-" + date_s.str[2:4] + "-" + date_s.str[4:6]
    # TIME: HHMMSS
    time_s = raw["TIME"].astype(int).astype(str).str.zfill(6)
    raw["_dt_time"] = time_s.str[:2] + ":" + time_s.str[2:4] + ":" + time_s.str[4:6]
    raw["datetime"] = pd.to_datetime(raw["_dt_date"] + " " + raw["_dt_time"], errors="coerce")
    raw = raw.dropna(subset=["datetime"]).drop(columns=["_dt_date", "_dt_time"])
    idx = raw["datetime"].values
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["OPEN"], errors="coerce").values,
            "high": pd.to_numeric(raw["HIGH"], errors="coerce").values,
            "low": pd.to_numeric(raw["LOW"], errors="coerce").values,
            "close": pd.to_numeric(raw["CLOSE"], errors="coerce").values,
            "volume": pd.to_numeric(raw["VOL"], errors="coerce").fillna(0).astype(int).values,
        },
        index=pd.DatetimeIndex(idx),
    ).sort_index()
    return out.dropna(subset=["open", "close"])


def find_manual_finam_files() -> dict[str, Path]:
    """
    Сканирует корень проекта на наличие YDEX*.txt в формате Finam.
    По колонке PER определяет таймфрейм: 15 → 15m, 60 → 1h.
    Возвращает {"15m": Path или None, "1h": Path или None}.
    """
    result: dict[str, Path] = {"15m": None, "1h": None}
    for path in PROJECT_ROOT.glob("YDEX*.txt"):
        try:
            head = pd.read_csv(path, sep=",", nrows=2)
        except Exception:
            continue
        head.columns = [c.strip().strip("<>") for c in head.columns]
        if "PER" not in head.columns or len(head) < 2:
            continue
        try:
            per = int(head["PER"].iloc[1])
        except (ValueError, TypeError):
            continue
        if per == 15 and result["15m"] is None:
            result["15m"] = path
        elif per == 60 and result["1h"] is None:
            result["1h"] = path
    return result


def make_fallback_data(interval: str, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """Тестовые данные (случайное блуждание), чтобы бэктест запустился."""
    import numpy as np
    np.random.seed(42)
    n_bars = 500 if interval == "15m" else 400
    freq = "15min" if interval == "15m" else "1h"
    start = pd.Timestamp(from_date).normalize()
    idx = pd.date_range(start=start, periods=n_bars, freq=freq)
    returns = np.random.randn(n_bars).cumsum() * 0.002
    close = 100 * np.exp(returns)
    open_ = np.roll(close, 1)
    open_[0] = 100
    high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    volume = (np.random.rand(n_bars) * 1_000_000 + 100_000).astype(int)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=idx).sort_index()


def download_yahoo_direct(symbol: str, interval: str,
                           from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """Yahoo Chart API через requests (без yfinance)."""
    yf_interval = "15m" if interval == "15m" else "60m"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": yf_interval,
        "period1": int(from_date.timestamp()),
        "period2": int(to_date.timestamp()),
        "includePrePost": "false",
    }
    for verify in [True, False]:
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30, verify=verify)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            continue
    else:
        return pd.DataFrame()
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()
    ch = result[0]
    timestamps = ch.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame()
    q = (ch.get("indicators") or {}).get("quote") or [{}]
    q = q[0]
    df = pd.DataFrame({
        "open": q.get("open") or [None] * len(timestamps),
        "high": q.get("high") or [None] * len(timestamps),
        "low": q.get("low") or [None] * len(timestamps),
        "close": q.get("close") or [None] * len(timestamps),
        "volume": q.get("volume") or [0] * len(timestamps),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))
    try:
        df.index = df.index.tz_convert("Europe/Moscow").tz_localize(None)
    except Exception:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["open", "close"])
    df["volume"] = df["volume"].fillna(0).astype(int)
    return df


def main():
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    api_key = os.environ.get("FINAM_API_KEY", "").strip()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not api_key:
        print("Hint: set FINAM_API_KEY in .env for Finam API v1 (ticker@mic).")

    to_date = datetime.now()
    from_date = to_date - timedelta(days=730)

    output_symbol = "YNDX"
    timeframes = [("15m", "15m"), ("1h", "1h")]

    for name, interval in timeframes:
        df = pd.DataFrame()

        # 1. Сначала официальный Finam API v1 (актуальные данные)
        if api_key:
            for symbol in ["YDEX@MISX", "YNDX@MISX", "SBER@MISX"]:
                print(f"Finam API v1: {symbol} {name}...", end=" ", flush=True)
                try:
                    df = download_finam_api_v1(api_key, symbol, interval, from_date, to_date)
                except Exception as e:
                    print(f"error: {e}")
                    df = pd.DataFrame()
                    continue
                if not df.empty:
                    print(f"OK ({len(df)} rows)")
                    break
                print("empty.")

        # 2. Ручные файлы Finam (YDEX*.txt в корне проекта), если API не дал данных
        if df.empty:
            manual = find_manual_finam_files()
            manual_path = manual.get(interval)
            if manual_path is not None:
                print(f"Manual Finam file: {manual_path.name} {name}...", end=" ", flush=True)
                df = load_manual_finam_txt(manual_path)
                if df is not None and not df.empty:
                    print(f"OK ({len(df)} rows)")
                else:
                    df = pd.DataFrame()
                    print("empty or error.")

        # 3. finam-export (export.finam.ru)
        if df.empty:
            for ticker in ["SBER", "YDEX", "YNDX", "GAZP"]:
                print(f"Finam (finam-export): {ticker} {name}...", end=" ", flush=True)
                try:
                    df = download_finam_library(ticker, interval, from_date, to_date)
                except Exception as e:
                    print(f"error: {e}")
                    df = pd.DataFrame()
                    continue
                if not df.empty:
                    print(f"OK ({len(df)} rows)")
                    break
                print("empty.")

        if df.empty:
            for symbol in ["YDEX.ME", "YNDX.ME", "SBER.ME", "GAZP.ME"]:
                print(f"Yahoo: {symbol} {name}...", end=" ", flush=True)
                df = download_yahoo_direct(symbol, interval, from_date, to_date)
                if not df.empty:
                    print(f"OK ({len(df)} rows)")
                    break
                print("empty.")

        if df.empty:
            print(f"Daily SBER.ME...", end=" ", flush=True)
            df = download_yahoo_direct("SBER.ME", "1d", from_date, to_date)
            if not df.empty:
                print(f"OK ({len(df)} rows)")
            else:
                print("empty.")

        if df.empty:
            print(f"No data for {name}. Add YDEX*.txt (PER=15 or 60) to project root or configure API.")
            continue

        out_path = DATA_DIR / f"{output_symbol}_{name}.csv"
        df.to_csv(out_path)
        print(f"Saved: {out_path.relative_to(PROJECT_ROOT)} ({len(df)} rows)")

    print("\nDone. Run: python backtest_isolated.py")


if __name__ == "__main__":
    main()
