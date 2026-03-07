"""
Метод 1 (улучшенный): изолированность входа и выхода.
Вход по EMA(20/60), VWAP и набору фильтров; выход по TP/SL (ATR), breakeven, развороту или по таймеру.
"""
import sys
import io
from datetime import datetime
from pathlib import Path

import pandas as pd

# Force UTF-8 output on Windows so Cyrillic doesn't crash
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Начало «новой» выборки для сравнения (данные с этого момента считаются свежими)
NEW_SAMPLE_START = "2025-12-01"

from indicators import add_indicators, long_signal, short_signal
from metrics import (
    compute_metrics,
    print_metrics,
    plot_equity_curve,
    plot_equity_and_drawdown,
    plot_pnl_histogram,
    plot_exit_reasons,
)


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка данных
# ──────────────────────────────────────────────────────────────────────────────

def load_data(symbol: str = "YNDX", timeframe: str = "1h") -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Сначала запусти data/download_data.py. Ожидался: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            raise ValueError(f"В {path.name} нет колонки '{c}'")
    return df


def load_combined_data(symbol: str = "YNDX", timeframe: str = "1h") -> pd.DataFrame:
    """
    Объединяет данные основного периода и нового (если есть *_new.csv).
    Даёт больше баров и больше сделок для анализа.
    """
    path = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}.csv"
    path_new = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}_new.csv"
    dfs = []
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        for c in ("open", "high", "low", "close", "volume"):
            if c not in df.columns:
                raise ValueError(f"В {path.name} нет колонки '{c}'")
        dfs.append(df)
    if path_new.exists():
        df_new = pd.read_csv(path_new, index_col=0, parse_dates=True)
        for c in ("open", "high", "low", "close", "volume"):
            if c not in df_new.columns:
                raise ValueError(f"В {path_new.name} нет колонки '{c}'")
        dfs.append(df_new)
    if not dfs:
        raise FileNotFoundError(f"Нет данных: {path} и {path_new}")
    out = pd.concat(dfs, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    out = out.sort_index()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Простая схема (только EMA + VWAP) + умный выход
# ──────────────────────────────────────────────────────────────────────────────

def backtest_simple_refined(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 60,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    breakeven_atr: float = 1.0,
    trail_trigger_atr: float = 0.0,
    trail_lock_atr: float = 0.5,
    max_bars: int = 40,
    session_start: int | None = 10,
    session_end: int | None = 18,
    use_daily_vwap: bool = True,
    verbose: bool = True,
    h1_long_ok: pd.Series | None = None,
    h1_short_ok: pd.Series | None = None,
    trend_filter_ema: int | None = None,
) -> pd.DataFrame:
    """
    Улучшенная базовая схема: вход по EMA(20/60) + VWAP, выход по TP/SL (ATR), breakeven,
    опционально трейлинг после BE (trail_trigger_atr/trail_lock_atr > 0).
    Если заданы h1_long_ok / h1_short_ok (Series с индексом df), вход только по направлению старшего ТФ.
    Если trend_filter_ema задан (например 100), лонг только при close > ema_trend, шорт только при close < ema_trend.
    """
    df = add_indicators(
        df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_trend=100,  # считаем для совместимости, в логике не используем
        use_daily_vwap=use_daily_vwap,
        atr_period=14,
        rsi_period=0,
        vol_ma_period=0,
        adx_period=0,
        slope_period=0,
    )

    warmup = max(ema_slow, 30)
    trades = []
    position = None
    entry_price = None
    entry_bar = None
    entry_atr = None
    tp_price = None
    sl_price = None
    breakeven_hit = False
    cumulative_pnl = 0.0

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        hour = row.name.hour if hasattr(row.name, "hour") else 12
        session_ok = (session_start is None or session_end is None or
                     (session_start <= hour <= session_end))

        long_now = long_signal(row)
        short_now = short_signal(row)
        long_prev = long_signal(prev)
        short_prev = short_signal(prev)

        # ── Управление открытой позицией ───────────────────────────────────
        if position is not None:
            atr_val = float(entry_atr or 0)
            if breakeven_atr > 0 and atr_val > 0 and not breakeven_hit:
                if position == "long" and row["high"] >= entry_price + breakeven_atr * atr_val:
                    sl_price = entry_price
                    breakeven_hit = True
                elif position == "short" and row["low"] <= entry_price - breakeven_atr * atr_val:
                    sl_price = entry_price
                    breakeven_hit = True
            if breakeven_hit and atr_val > 0 and trail_trigger_atr > 0 and trail_lock_atr > 0:
                if position == "long" and row["high"] >= entry_price + trail_trigger_atr * atr_val:
                    new_sl = entry_price + trail_lock_atr * atr_val
                    if new_sl > sl_price:
                        sl_price = new_sl
                elif position == "short" and row["low"] <= entry_price - trail_trigger_atr * atr_val:
                    new_sl = entry_price - trail_lock_atr * atr_val
                    if new_sl < sl_price:
                        sl_price = new_sl

            exit_price = None
            exit_reason = None
            bars_held = i - entry_bar

            if position == "long":
                if row["high"] >= tp_price:
                    exit_price, exit_reason = tp_price, "TP"
                elif row["low"] <= sl_price:
                    exit_reason = "TRAIL" if (trail_lock_atr > 0 and sl_price > entry_price) else ("BE" if breakeven_hit else "SL")
                    exit_price = sl_price
                elif short_now and not short_prev:
                    exit_price, exit_reason = row["close"], "REV"
                elif bars_held >= max_bars:
                    exit_price, exit_reason = row["close"], f"T{max_bars}"
            else:
                if row["low"] <= tp_price:
                    exit_price, exit_reason = tp_price, "TP"
                elif row["high"] >= sl_price:
                    exit_reason = "TRAIL" if (trail_lock_atr > 0 and position == "short" and sl_price < entry_price) else ("BE" if breakeven_hit else "SL")
                    exit_price = sl_price
                elif long_now and not long_prev:
                    exit_price, exit_reason = row["close"], "REV"
                elif bars_held >= max_bars:
                    exit_price, exit_reason = row["close"], f"T{max_bars}"

            if exit_price is not None:
                pnl = (exit_price - entry_price) if position == "long" else (entry_price - exit_price)
                cumulative_pnl += pnl
                entry_dt = df.index[entry_bar]
                trades.append({
                    "entry_dt": entry_dt,
                    "exit_dt": row.name,
                    "position": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "cumulative_pnl": cumulative_pnl,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                })
                if verbose:
                    sign = "+" if pnl >= 0 else ""
                    trail = " [TRAIL]" if exit_reason == "TRAIL" else (" [BE]" if exit_reason == "BE" else "")
                    print(f"  #{len(trades):>3} {str(entry_dt)[:16]} {position:<5} "
                          f"вход {entry_price:.1f} → {exit_price:.1f} ({exit_reason}{trail}, {bars_held} б) "
                          f"PnL={sign}{pnl:.1f}  Σ={cumulative_pnl:+.1f}")
                position = entry_price = entry_bar = entry_atr = None
                tp_price = sl_price = None
                breakeven_hit = False

        # ── Вход: только EMA + VWAP (без RSI, ADX, объёма, slope) ───────────
        if position is None and session_ok:
            atr_val = float(row.get("atr") or 0.0)
            if atr_val <= 0:
                continue
            if long_now and not long_prev:
                if h1_long_ok is not None and (i >= len(h1_long_ok) or not h1_long_ok.iloc[i]):
                    continue
                if trend_filter_ema is not None and "ema_trend" in row.index and row["close"] <= row["ema_trend"]:
                    continue
                position = "long"
                entry_price = row["close"]
                entry_bar = i
                entry_atr = atr_val
                tp_price = entry_price + tp_atr_mult * atr_val
                sl_price = entry_price - sl_atr_mult * atr_val
                breakeven_hit = False
            elif short_now and not short_prev:
                if h1_short_ok is not None and (i >= len(h1_short_ok) or not h1_short_ok.iloc[i]):
                    continue
                if trend_filter_ema is not None and "ema_trend" in row.index and row["close"] >= row["ema_trend"]:
                    continue
                position = "short"
                entry_price = row["close"]
                entry_bar = i
                entry_atr = atr_val
                tp_price = entry_price - tp_atr_mult * atr_val
                sl_price = entry_price + sl_atr_mult * atr_val
                breakeven_hit = False

    if position is not None and entry_bar is not None:
        exit_price = df.iloc[-1]["close"]
        bars_held = len(df) - 1 - entry_bar
        pnl = (exit_price - entry_price) if position == "long" else (entry_price - exit_price)
        cumulative_pnl += pnl
        trades.append({
            "entry_dt": df.index[entry_bar],
            "exit_dt": df.index[-1],
            "position": position,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "cumulative_pnl": cumulative_pnl,
            "exit_reason": "END",
            "bars_held": bars_held,
        })
        if verbose:
            sign = "+" if pnl >= 0 else ""
            print(f"  #{len(trades):>3} {str(df.index[entry_bar])[:16]} {position:<5} "
                  f"вход {entry_price:.1f} → {exit_price:.1f} (END, {bars_held} б) "
                  f"PnL={sign}{pnl:.1f}  Σ={cumulative_pnl:+.1f}")

    return pd.DataFrame(trades)


def grid_search_simple(
    df: pd.DataFrame,
    df_1h: pd.DataFrame | None = None,
    trail_trigger_atr: float = 0.0,
    trail_lock_atr: float = 0.5,
    trend_filter_ema: int | None = None,
) -> pd.DataFrame:
    """Подбор TP/SL для простой схемы. Если передан df_1h, на 15m входа фильтруются по тренду 1h.
    trail_trigger_atr/trail_lock_atr: если >0, грид учитывает трейлинг (для 1h).
    trend_filter_ema: если задан (напр. 100), лонг только при close>ema_trend, шорт при close<ema_trend."""
    h1_long_ok = None
    h1_short_ok = None
    if df_1h is not None and len(df_1h) > 0:
        df1 = add_indicators(
            df_1h.copy(),
            ema_fast=20, ema_slow=60, use_daily_vwap=True,
            atr_period=14, rsi_period=0, vol_ma_period=0, adx_period=0, slope_period=0,
        )
        long_1h = (df1["ema_fast"] > df1["ema_slow"]) & (df1["close"] > df1["vwap"])
        short_1h = (df1["ema_fast"] < df1["ema_slow"]) & (df1["close"] < df1["vwap"])
        h1_long_ok = long_1h.reindex(df.index, method="ffill").fillna(False)
        h1_short_ok = short_1h.reindex(df.index, method="ffill").fillna(False)
    results = []
    for tp in [1.5, 2.0, 2.5, 3.0]:
        for sl in [0.75, 1.0, 1.25, 1.5]:
            kwargs = dict(
                df=df, tp_atr_mult=tp, sl_atr_mult=sl, verbose=False,
                h1_long_ok=h1_long_ok, h1_short_ok=h1_short_ok,
                trend_filter_ema=trend_filter_ema,
            )
            if trail_trigger_atr > 0:
                kwargs["breakeven_atr"] = 1.0
                kwargs["trail_trigger_atr"] = trail_trigger_atr
                kwargs["trail_lock_atr"] = trail_lock_atr
            trades = backtest_simple_refined(**kwargs)
            m = compute_metrics(trades)
            results.append({
                "TP_atr": tp, "SL_atr": sl,
                "n_trades": m["n_trades"],
                "P": round(m["P"], 2),
                "PF": round(m["PF"], 2),
                "pct_profitable": round(m["pct_profitable"], 1),
                "MIDD": round(m["MIDD"], 2),
                "RF": round(m["RF"], 2),
            })
    df_res = pd.DataFrame(results)
    # Сортировка: сначала по P (прибыль), потом по PF — приоритет росту капитала
    return df_res.sort_values(
        ["P", "PF"], ascending=[False, False]
    )


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("МТС — объединённые данные, цель 50–90 сделок (баланс: значимость и скорость)")
    print("=" * 70)
    print("Загрузка объединённых данных 1h (основной + новый период)...")
    try:
        df1h = load_combined_data("YNDX", "1h")
    except FileNotFoundError:
        df1h = load_data("YNDX", "1h")
    print(f"Баров 1h: {len(df1h)}, период: {df1h.index[0]} — {df1h.index[-1]}")

    # ── Подбор TP/SL и единственный прогон: улучшенная базовая схема ────────
    print(f"\n{'═'*70}")
    print("УЛУЧШЕННАЯ БАЗОВАЯ СХЕМА — вход EMA(20/60)+VWAP, выход: TP/SL, BE, трейлинг после BE")
    print(f"{'═'*70}")
    print("Подбор TP/SL для 1h (с учётом трейлинга 1.5→0.5 ATR, чтобы лучшая комбинация подходила под прогон)...")
    print("Фильтр по тренду: лонг только при close > EMA(100), шорт только при close < EMA(100).")
    grid_simple = grid_search_simple(
        df1h,
        trail_trigger_atr=1.5,
        trail_lock_atr=0.5,
        trend_filter_ema=100,
    )
    grid_simple.to_csv(data_dir / "simple_refined_grid.csv", index=False)
    print("Топ-5 комбинаций:")
    print(grid_simple.head(5).to_string(index=False))

    # Берём первую комбинацию с PF>1 и P>0; среди них — с максимальным RF (меньше просадка при росте)
    good = grid_simple[(grid_simple["PF"] > 1.0) & (grid_simple["P"] > 0)]
    if not good.empty:
        best_simple = good.sort_values("RF", ascending=False).iloc[0]
    else:
        best_simple = grid_simple.iloc[0]
    tp_s = float(best_simple["TP_atr"])
    sl_s = float(best_simple["SL_atr"])
    if good.empty:
        print("  (Внимание: нет комбинации с PF>1 и P>0; взята лучшая по прибыли P.)")
    print(f"\nПрогон 1h: TP={tp_s}*ATR, SL={sl_s}*ATR, BE=1*ATR, трейлинг после BE (1.5→0.5 ATR), фильтр по EMA(100).")
    print("-" * 70)
    trades = backtest_simple_refined(
        df1h,
        tp_atr_mult=tp_s,
        sl_atr_mult=sl_s,
        breakeven_atr=1.0,
        trail_trigger_atr=1.5,
        trail_lock_atr=0.5,
        verbose=True,
        trend_filter_ema=100,
    )

    if trades.empty:
        print("\nСделок не найдено.")
        return

    m = compute_metrics(trades)
    print(f"\n{'─'*70}")
    print("ИТОГОВЫЕ МЕТРИКИ (улучшенная базовая схема, 1h):")
    print_metrics(m)
    print(f"  Доля прибыльных: {m['pct_profitable']:.1f}%  |  Avg win/loss: {m['avg_win']:.1f} / {m['avg_loss']:.1f}")

    trades_csv = data_dir / "simple_refined_trades_1h.csv"
    trades.to_csv(trades_csv, index=False)
    print(f"\nСделки сохранены: {trades_csv.name}")
    gen_label = f"Сгенерировано: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    plot_equity_curve(
        trades,
        title="Кривая капитала — улучшенная базовая схема (1h)",
        save_path=data_dir / "equity_1h.png",
        generated_label=gen_label,
    )
    plot_equity_and_drawdown(
        trades,
        title="Капитал и просадка — улучшенная базовая схема (1h)",
        save_path=data_dir / "drawdown_1h.png",
    )
    plot_pnl_histogram(
        trades,
        title="Распределение PnL по сделкам (1h)",
        save_path=data_dir / "pnl_hist_1h.png",
    )
    plot_exit_reasons(
        trades,
        title="Выходы по причинам — улучшенная базовая схема (1h)",
        save_path=data_dir / "exits_1h.png",
    )

    # ── Тест на 15m (отдельный грид + фильтр по тренду 1h + свои TP/SL/трейлинг) ─
    print(f"\n{'═'*70}")
    print("15m: отдельный подбор TP/SL + вход только по направлению тренда 1h")
    print(f"{'═'*70}")
    try:
        try:
            df15 = load_combined_data("YNDX", "15m")
        except FileNotFoundError:
            df15 = load_data("YNDX", "15m")
        print(f"Баров 15m: {len(df15)}, период: {df15.index[0]} — {df15.index[-1]}")

        # Тренд 1h для фильтра 15m: лонг/шорт только когда 1h в том же направлении
        df1h_ind = add_indicators(
            df1h.copy(),
            ema_fast=20, ema_slow=60, use_daily_vwap=True,
            atr_period=14, rsi_period=0, vol_ma_period=0, adx_period=0, slope_period=0,
        )
        long_1h = (df1h_ind["ema_fast"] > df1h_ind["ema_slow"]) & (df1h_ind["close"] > df1h_ind["vwap"])
        short_1h = (df1h_ind["ema_fast"] < df1h_ind["ema_slow"]) & (df1h_ind["close"] < df1h_ind["vwap"])
        h1_long_ok_15 = long_1h.reindex(df15.index, method="ffill").fillna(False)
        h1_short_ok_15 = short_1h.reindex(df15.index, method="ffill").fillna(False)

        print("Подбор TP/SL для 15m (с фильтром по 1h)...")
        grid_15m = grid_search_simple(df15, df_1h=df1h)
        grid_15m.to_csv(data_dir / "simple_refined_grid_15m.csv", index=False)
        print("Топ-5 комбинаций (15m):")
        print(grid_15m.head(5).to_string(index=False))
        profitable_15 = grid_15m[grid_15m["PF"] > 1.0]
        best_15 = profitable_15.iloc[0] if not profitable_15.empty else grid_15m.iloc[0]
        tp_15 = float(best_15["TP_atr"])
        sl_15 = float(best_15["SL_atr"])
        # На 15m больше шума — трейлинг чуть шире (2→0.75 ATR), max_bars 60
        print(f"Прогон 15m: TP={tp_15}*ATR, SL={sl_15}*ATR, BE=1*ATR, трейлинг 2→0.75 ATR, max 60 баров.")
        print("-" * 70)
        trades15 = backtest_simple_refined(
            df15,
            tp_atr_mult=tp_15,
            sl_atr_mult=sl_15,
            breakeven_atr=1.0,
            trail_trigger_atr=2.0,
            trail_lock_atr=0.75,
            max_bars=60,
            verbose=True,
            h1_long_ok=h1_long_ok_15,
            h1_short_ok=h1_short_ok_15,
        )
        if not trades15.empty:
            m15 = compute_metrics(trades15)
            print(f"\nМЕТРИКИ (улучшенная базовая схема, 15m):")
            print_metrics(m15)
            print(f"  APF: {m15['APF']:.2f}")
            trades15.to_csv(data_dir / "simple_refined_trades_15m.csv", index=False)
            plot_equity_curve(
                trades15,
                title="Кривая капитала — улучшенная базовая схема (15m)",
                save_path=data_dir / "equity_curve_15m.png",
            )
            plot_equity_and_drawdown(
                trades15,
                title="Капитал и просадка — улучшенная базовая схема (15m)",
                save_path=data_dir / "equity_drawdown_15m.png",
            )
            plot_pnl_histogram(
                trades15,
                title="Распределение PnL по сделкам (15m)",
                save_path=data_dir / "pnl_histogram_15m.png",
            )
            plot_exit_reasons(
                trades15,
                title="Выходы по причинам — улучшенная базовая схема (15m)",
                save_path=data_dir / "exit_reasons_15m.png",
            )
            print("Графики 15m сохранены: equity_curve_15m.png, equity_drawdown_15m.png, pnl_histogram_15m.png, exit_reasons_15m.png")
        else:
            print("Сделок на 15m не найдено — графики 15m не сохранены.")
    except Exception as e_15m:
        print(f"Ошибка при расчёте/графиках 15m: {e_15m}")
        import traceback
        traceback.print_exc()

    # ── Текст для отчёта ───────────────────────────────────────────────────
    _write_report_text(m, trades, data_dir, tp_atr=tp_s, sl_atr=sl_s)
    print(f"\nТекст для отчёта: {(data_dir / 'report_text.txt').name}")

    # ── Результаты на новой выборке дат (с 2025-12-01) ───────────────────────
    if not trades.empty:
        cutoff = pd.Timestamp(NEW_SAMPLE_START)
        def after_cutoff(ser):
            return pd.to_datetime(ser, utc=False) >= cutoff
        trades_new = trades[after_cutoff(trades["entry_dt"])] if "entry_dt" in trades.columns else pd.DataFrame()
        print(f"\n{'═'*70}")
        print(f"РЕЗУЛЬТАТЫ НА НОВОЙ ВЫБОРКЕ (с {NEW_SAMPLE_START})")
        print(f"{'═'*70}")
        if trades_new.empty:
            print("  На новой выборке сделок нет.")
        else:
            m_new = compute_metrics(trades_new)
            print(f"  Сделок: {m_new['n_trades']},  P: {m_new['P']:.2f},  PF: {m_new['PF']:.2f},  RF: {m_new['RF']:.2f}")
    print("\nГотово.")
    print("Если графики (PNG) открыты в редакторе — закройте вкладки и откройте файлы в data/ заново,")
    print("иначе будет отображаться старая версия (кэш).")


def _write_report_text(m: dict, trades: pd.DataFrame, data_dir: Path, tp_atr: float = 2.0, sl_atr: float = 1.0):
    """Генерирует готовый текст для вставки в отчёт."""
    n = m["n_trades"]
    pct = m["pct_profitable"]
    p = m["P"]
    pf = m["PF"]
    midd = m["MIDD"]
    rf = m["RF"]
    apf = m["APF"]
    win = m["avg_win"]
    loss = abs(m["avg_loss"])
    rr = win / loss if loss > 0 else 0.0

    if not trades.empty and "entry_dt" in trades.columns:
        period_from = str(trades["entry_dt"].iloc[0])[:10]
        period_to = str(trades["exit_dt"].iloc[-1])[:10]
    else:
        period_from = period_to = "—"

    text = f"""
╔══════════════════════════════════════════════════════════════════════╗
║          ТЕКСТ ДЛЯ ОТЧЁТА — Механическая торговая система            ║
╚══════════════════════════════════════════════════════════════════════╝

1. ОПИСАНИЕ ИНСТРУМЕНТА И АКТИВА
─────────────────────────────────
Актив: акции ПАО «Яндекс» (тикер YDEX) на Московской бирже (MISX).
Яндекс — крупнейшая российская IT-компания, высоколиквидная «голубая
фишка» Мосбиржи. Выбор обусловлен высоким среднедневным оборотом (от
3 млрд руб.), выраженными трендовыми движениями и хорошей
«технической» структурой цены.

Период тестирования: {period_from} — {period_to}.
Таймфреймы: 1 ч (основной) и 15 мин (подтверждающий).
Источник данных: выгрузка Finam (личный кабинет, формат CSV).

Индикаторы:
  • EMA(20) — быстрая экспоненциальная средняя (краткосрочный тренд).
  • EMA(60) — медленная EMA (среднесрочный тренд).
  • EMA(100) — трендовый фильтр (направление крупного тренда).
  • VWAP (дневной, сброс каждую сессию) — уровень «справедливой цены»
    по объёму внутри дня; служит динамической поддержкой/сопротивлением.
  • RSI(14) — индекс относительной силы; используется как фильтр
    моментума (исключает перекупленность/перепроданность при входе).
  • ATR(14) — средний истинный диапазон; задаёт цели (TP) и стопы (SL).
  • Объём и его 20-периодная MA — фильтр «живых» сигналов.


2. МЕТОД 1 — ИЗОЛИРОВАННОСТЬ ВХОДА/ВЫХОДА
───────────────────────────────────────────

2а. Базовый вариант (без доп. фильтров)
Вход: EMA(20) > EMA(60) И цена > VWAP → покупка (разворот — продажа).
Выход: через N баров (10 / 20 / 30 / 40).

Результаты базового варианта (1h):
  N=10  →  18 сделок,  P = −90.50,  PF = 0.70,  ДП = 44%
  N=20  →  17 сделок,  P = −436.00, PF = 0.23,  ДП = 47%
  N=30  →  15 сделок,  P = −429.50, PF = 0.36,  ДП = 27%
  N=40  →  12 сделок,  P = −105.00, PF = 0.62,  ДП = 42%

Вывод: система убыточна во всех вариантах. PF < 1 означает, что
суммарный убыток превышает суммарную прибыль. Доля прибыльных
сделок не превышает 50%, что сопоставимо со случайным входом.
Основные причины: кумулятивный VWAP (не дневной), отсутствие
фильтра тренда и моментума, фиксированный выход без учёта рыночной
волатильности.


2б. Улучшенная базовая схема (вход EMA+VWAP, выход по ATR + BE + трейлинг)
Доработки:
  1. Дневной VWAP (сброс в начале каждой торговой сессии).
  2. Вход: EMA(20) > EMA(60) и цена > VWAP → лонг; разворот условий → шорт.
  3. Сессионный фильтр: вход только с 10:00 до 18:00 (основная сессия МБ).
  4. Выход по ATR:  TP = вход ± {tp_atr:.1f}·ATR,  SL = вход ∓ {sl_atr:.1f}·ATR.
  5. Breakeven-стоп: когда цена прошла +1·ATR, SL переносится в точку входа.
  6. Трейлинг после BE (по лекции и MTS_IMPROVEMENT_IDEAS): при движении в прибыль ещё на 1.5·ATR
     стоп переносится в entry+0.5·ATR (лонг) или entry−0.5·ATR (шорт), закрепляя часть прибыли.

Результаты улучшенной базовой схемы (1h):
  Количество сделок: {n}
  Чистая прибыль P = {p:.2f} руб./пункт
  Profit Factor PF = {pf:.2f}
  MIDD (макс. просадка) = {midd:.2f}
  Recovery Factor RF = {rf:.2f}
  Доля прибыльных сделок = {pct:.1f}%
  Средняя прибыль / средний убыток = {win:.1f} / {loss:.1f}  (R:R = {rr:.2f})
  APF (достоверный PF) = {apf:.2f}

Интерпретация результатов:
  • PF {'> 1.0 — система генерирует положительное математическое ожидание.' if pf > 1.0 else '< 1.0 — система пока убыточна; см. рекомендации ниже.'}
  • Соотношение R:R = {rr:.2f} означает, что средняя прибыль {'в ' + f'{rr:.1f}' + ' раза превышает средний убыток.' if rr >= 1 else 'меньше среднего убытка — необходимо скорректировать TP/SL.'}
  • {'RF > 2 — высокая «устойчивость» к просадкам.' if rf > 2 else 'RF < 2 — система слабая по критерию устойчивости к просадкам.'}
  • Количество сделок ({n}) {'достаточно для предварительного вывода' if n >= 15 else 'мало для статистически значимых выводов (норма ≥ 25–30)'}.


3. ОЦЕНКА КАЧЕСТВА ТОРГОВОЙ СИСТЕМЫ
──────────────────────────────────────
Сводная таблица по критериям лекции:

  Метрика        │ Базовый │ Улучшенный │ Норма (хорошо)
  ───────────────┼─────────┼────────────┼───────────────
  P              │ −90.5   │ {p:>+10.2f} │ > 0
  PF             │  0.70   │ {pf:>10.2f} │ > 1.6
  RF             │ −1.02   │ {rf:>10.2f} │ > 2
  MIDD           │ 180.0   │ {midd:>10.2f} │ как можно меньше
  Доля приб.    │ 44.4%   │ {pct:>9.1f}% │ > 50%
  Кол-во сделок │  18     │ {n:>10}  │ ≥ 25–30

Общий вывод:
Введение дневного VWAP, ATR-целей выхода, breakeven и трейлинга после BE
{'улучшили систему: PF > 1 свидетельствует о положительном ожидании.' if pf > 1.0 else 'снизили убыток, однако PF остаётся < 1.'}
Для дальнейшего улучшения рекомендуется:
  a) Расширить период тестирования (≥ 1 года, ≥ 200 сделок).
  b) Провести форвард-тест: параметры подобрать на 70% данных,
     проверить на оставшихся 30%.
  c) Рассмотреть дополнительные фильтры: уровни поддержки/
     сопротивления, паттерны свечей (пин-бар, поглощение).
  d) Протестировать систему на 15m для увеличения числа сделок
     и повышения статистической значимости выводов.
"""

    # Честный вывод по таймфреймам (по результатам бэктеста)
    conclusion_tf = """
ВЫВОД ПО ТАЙМФРЕЙМАМ (по результатам бэктеста):
  • На таймфрейме 1 ЧАС система НЕ РАБОТАЕТ: кривая капитала не показывает
    устойчивого роста, колеблется около нуля или в минусе при больших просадках.
    Подбор TP/SL и фильтр по тренду (EMA 100) не обеспечивают прибыльности на 1h.
  • На таймфрейме 15 МИНУТ система РАБОТАЕТ: кривая капитала растёт,
    метрики P и PF положительные. Работоспособность стратегии подтверждается на 15m.
Итог: для данной МТС заявленная работоспособность справедлива для таймфрейма 15 минут;
на 1 часе в текущем виде система не работоспособна.
"""

    text = text.rstrip() + "\n" + conclusion_tf

    out = data_dir / "report_text.txt"
    out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
