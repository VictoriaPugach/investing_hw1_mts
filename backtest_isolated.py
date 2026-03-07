"""
Метод 1: изолированность входа/выхода.
Вход — по правилам (EMA 20/60 + VWAP). Выход — через фиксированное число баров (10, 20, 30, 40).
Цель: найти интервалы выхода с большей долей прибыльных сделок.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators import add_indicators, long_signal, short_signal
from metrics import compute_metrics, print_metrics, plot_equity_curve


def load_data(symbol: str = "YNDX", timeframe: str = "1h") -> pd.DataFrame:
    path_15 = PROJECT_ROOT / "data" / f"{symbol}_15m.csv"
    path_1h = PROJECT_ROOT / "data" / f"{symbol}_1h.csv"
    path = path_1h if timeframe == "1h" else path_15
    if not path.exists():
        raise FileNotFoundError(f"Сначала запусти data/download_data.py. Ожидался файл: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"В {path} нет колонки {c}")
    return df


def backtest_isolated_exit(
    df: pd.DataFrame,
    exit_bars: int,
    ema_fast: int = 20,
    ema_slow: int = 60,
    atr_min_pct: float | None = None,
    use_stop_atr: float | None = None,
    use_daily_vwap: bool = False,
) -> pd.DataFrame:
    """
    Вход: при смене сигнала с шорт на лонг — покупка; с лонг на шорт — продажа.
    Выход: ровно через exit_bars баров (изолированный выход).
    use_daily_vwap=False — кумулятивный VWAP (как в оригинальном методе 1 для отчёта).
    """
    atr_period = 14
    df = add_indicators(df, ema_fast=ema_fast, ema_slow=ema_slow, atr_period=atr_period, use_daily_vwap=use_daily_vwap)
    if atr_min_pct is not None and "atr" in df.columns:
        atr_median = df["atr"].rolling(50, min_periods=atr_period).median()
        df["_atr_ok"] = df["atr"] >= (atr_median * atr_min_pct)
    else:
        df["_atr_ok"] = True
    trades = []
    position = None  # "long" | "short" | None
    entry_price = None
    entry_bar = None
    entry_atr = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        long_now = long_signal(row)
        short_now = short_signal(row)
        long_prev = long_signal(prev)
        short_prev = short_signal(prev)

        # Досрочный выход по стопу (ATR)
        if position is not None and entry_bar is not None and use_stop_atr and entry_atr is not None:
            exit_price = row["close"]
            if position == "long":
                pnl = exit_price - entry_price
                stop_triggered = pnl <= -use_stop_atr * entry_atr
            else:
                pnl = entry_price - exit_price
                stop_triggered = pnl <= -use_stop_atr * entry_atr
            if stop_triggered:
                trades.append({"pnl": pnl, "entry_bar": entry_bar, "exit_bar": i, "position": position})
                position = None
                entry_price = None
                entry_bar = None
                entry_atr = None
                continue

        # Выход по истечении N баров
        if position is not None and entry_bar is not None:
            bars_held = i - entry_bar
            if bars_held >= exit_bars:
                exit_price = row["close"]
                if position == "long":
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price
                trades.append({"pnl": pnl, "entry_bar": entry_bar, "exit_bar": i, "position": position})
                position = None
                entry_price = None
                entry_bar = None
                entry_atr = None

        # Вход по пересечению (фильтр ATR только при входе)
        if position is None and row.get("_atr_ok", True):
            if long_now and not long_prev:
                position = "long"
                entry_price = row["close"]
                entry_bar = i
                entry_atr = row.get("atr", None)
            elif short_now and not short_prev:
                position = "short"
                entry_price = row["close"]
                entry_bar = i
                entry_atr = row.get("atr", None)

    # Закрыть последнюю позицию в конце данных
    if position is not None and entry_bar is not None:
        exit_price = df.iloc[-1]["close"]
        if position == "long":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        trades.append({"pnl": pnl, "entry_bar": entry_bar, "exit_bar": len(df) - 1, "position": position})

    return pd.DataFrame(trades)


def main():
    symbol = "YNDX"
    timeframe = "1h"
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Загрузка данных...")
    df = load_data(symbol, timeframe)
    # Базовый вариант: кумулятивный VWAP (как в лекции), чтобы результаты не «уплывали»
    df = add_indicators(df, use_daily_vwap=False)
    print(f"Баров: {len(df)}, период: {df.index[0]} — {df.index[-1]}")

    exit_intervals_baseline = [10, 20, 30, 40]

    # ---- БАЗОВЫЙ ВАРИАНТ (для отчёта — сохраняем как историю) ----
    print("\n[Базовый вариант — для отчёта] Изолированность выхода (выход через N баров)")
    print("-" * 60)
    results_baseline = []
    for n in exit_intervals_baseline:
        trades = backtest_isolated_exit(df, exit_bars=n, use_daily_vwap=False)
        m = compute_metrics(trades)
        pct = m["pct_profitable"]
        results_baseline.append({
            "exit_bars": n,
            "n_trades": m["n_trades"],
            "P": m["P"],
            "PF": m["PF"],
            "pct_profitable": pct,
            "MIDD": m["MIDD"],
        })
        print(f"Выход через {n} баров: сделок {m['n_trades']}, P={m['P']:.2f}, PF={m['PF']:.2f}, доля прибыльных={pct:.1f}%")

    res_baseline = pd.DataFrame(results_baseline)
    baseline_csv = data_dir / "method1_baseline_for_report.csv"
    res_baseline.to_csv(baseline_csv, index=False)
    with open(data_dir / "report_baseline_summary.txt", "w", encoding="utf-8") as f:
        f.write("Метод 1 — базовый вариант (для отчёта)\n")
        f.write("Параметры: выход через 10, 20, 30, 40 баров; вход по EMA 20/60 + VWAP.\n")
        f.write("Данные: 1h, период из data/YNDX_1h.csv\n")
        f.write("Таблица результатов: method1_baseline_for_report.csv\n")
    print(f"\nБазовые результаты для отчёта сохранены: {baseline_csv.name}")

    best_idx = res_baseline["pct_profitable"].idxmax()
    best_n = int(res_baseline.loc[best_idx, "exit_bars"])
    print(f"Лучший интервал выхода по доле прибыльных (базовый): {best_n} баров")
    trades_best = backtest_isolated_exit(df, exit_bars=best_n, use_daily_vwap=False)
    print_metrics(compute_metrics(trades_best))
    plot_equity_curve(trades_best, title="Кривая капитала (метод 1, базовый)", save_path=data_dir / "equity_curve_method1_baseline.png")

    # ---- УЛУЧШЕННЫЙ ВАРИАНТ (фильтр ATR + стоп по ATR) ----
    print("\n[Улучшенный вариант] Вход при ATR >= 0.5*median(ATR), стоп 1.5*ATR, те же интервалы выхода")
    print("-" * 60)
    results_improved = []
    for n in exit_intervals_baseline:
        trades = backtest_isolated_exit(df, exit_bars=n, atr_min_pct=0.5, use_stop_atr=1.5, use_daily_vwap=False)
        m = compute_metrics(trades)
        pct = m["pct_profitable"]
        results_improved.append({
            "exit_bars": n,
            "n_trades": m["n_trades"],
            "P": m["P"],
            "PF": m["PF"],
            "pct_profitable": pct,
            "MIDD": m["MIDD"],
        })
        print(f"Выход через {n} баров: сделок {m['n_trades']}, P={m['P']:.2f}, PF={m['PF']:.2f}, доля прибыльных={pct:.1f}%")

    res_improved = pd.DataFrame(results_improved)
    improved_csv = data_dir / "method1_improved_results.csv"
    res_improved.to_csv(improved_csv, index=False)
    print(f"\nУлучшенные результаты сохранены: {improved_csv.name}")

    if not res_improved.empty and int(res_improved["n_trades"].sum()) > 0:
        best_imp_idx = res_improved["pct_profitable"].idxmax()
        best_imp_n = int(res_improved.loc[best_imp_idx, "exit_bars"])
        trades_imp = backtest_isolated_exit(df, exit_bars=best_imp_n, atr_min_pct=0.5, use_stop_atr=1.5, use_daily_vwap=False)
        print_metrics(compute_metrics(trades_imp))
        plot_equity_curve(trades_imp, title="Кривая капитала (метод 1, улучшенный)", save_path=data_dir / "equity_curve_method1_improved.png")
    else:
        print("(Улучшенный вариант: сделок нет или мало — график не строится)")

    # Совместимость: основная таблица для Excel — как раньше
    res_df = res_baseline
    out_csv = data_dir / "method1_isolated_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nТаблица для Excel: {out_csv.name}")
    print("\nГотово. В отчёте используй: method1_baseline_for_report.csv (исходные параметры), method1_improved_results.csv (улучшенный вариант).")


if __name__ == "__main__":
    main()
