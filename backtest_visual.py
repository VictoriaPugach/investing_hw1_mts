"""
Метод 2: визуально-графический анализ.
Решётчатый перебор параметров (периоды EMA), построение графика результата от параметров.
Оценка устойчивости: много ли прибыльных комбинаций.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics import compute_metrics
from backtest_isolated import load_data
from backtest_expert import backtest_simple_refined


def run_grid(
    df: pd.DataFrame,
    ema_fast_range: list = None,
    ema_slow_range: list = None,
) -> pd.DataFrame:
    """Решётчатый перебор EMA для улучшенной рабочей схемы 1h."""
    if ema_fast_range is None:
        ema_fast_range = [10, 15, 20, 25, 30]
    if ema_slow_range is None:
        ema_slow_range = [40, 50, 60, 70, 80]
    rows = []
    for fast in ema_fast_range:
        for slow in ema_slow_range:
            if slow <= fast:
                continue
            trades = backtest_simple_refined(
                df=df,
                ema_fast=fast,
                ema_slow=slow,
                tp_atr_mult=1.5,
                sl_atr_mult=1.5,
                breakeven_atr=1.0,
                trail_trigger_atr=1.5,
                trail_lock_atr=0.5,
                max_bars=40,
                session_start=10,
                session_end=17,
                use_daily_vwap=True,
                verbose=False,
                trend_filter_ema=100,
                adx_min=18,
                rsi_long_min=52,
                rsi_short_max=48,
                vol_ratio_min=0.9,
                slope_min=0.0002,
            )
            m = compute_metrics(trades)
            rows.append({
                "ema_fast": fast,
                "ema_slow": slow,
                "n_trades": m["n_trades"],
                "P": m["P"],
                "PF": m["PF"],
                "MIDD": m["MIDD"],
                "RF": m["RF"],
                "pct_profitable": m["pct_profitable"],
            })
    return pd.DataFrame(rows)


def main():
    symbol = "YNDX"
    timeframe = "1h"
    print("Загрузка данных...")
    df = load_data(symbol, timeframe)
    print(f"Баров: {len(df)}")

    res_df = run_grid(df)
    res_df = res_df[res_df["n_trades"] >= 5]  # отсечь слишком мало сделок

    out_csv = PROJECT_ROOT / "data" / "method2_visual_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Таблица сохранена: {out_csv}")

    # Доля прибыльных тестов (устойчивость по лекции)
    n_total = len(res_df)
    n_profitable = (res_df["P"] > 0).sum()
    pct_profitable_tests = n_profitable / n_total * 100 if n_total else 0
    print(f"\nМетод 2: визуально-графический анализ")
    print(f"Всего комбинаций параметров: {n_total}, прибыльных: {n_profitable} ({pct_profitable_tests:.1f}%)")
    if pct_profitable_tests >= 20:
        print("Устойчивость: не менее 20% тестов прибыльные — система считается устойчивой (по лекции).")
    else:
        print("Устойчивость: менее 20% тестов прибыльные — стоит доработать параметры.")

    # График: прибыль от параметра (один параметр — ema_fast при фикс. ema_slow)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    slow_ref = 60
    sub = res_df[res_df["ema_slow"] == slow_ref].sort_values("ema_fast")
    if not sub.empty:
        axes[0].bar(sub["ema_fast"].astype(str), sub["P"], color=["green" if p > 0 else "red" for p in sub["P"]])
        axes[0].set_xlabel("EMA быстрая (период)")
        axes[0].set_ylabel("Чистая прибыль P")
        axes[0].set_title(f"Прибыль при EMA медленная = {slow_ref}")
        axes[0].tick_params(axis="x", rotation=15)
    axes[1].scatter(res_df["ema_fast"], res_df["ema_slow"], c=res_df["P"], cmap="RdYlGn", s=80)
    axes[1].set_xlabel("EMA быстрая")
    axes[1].set_ylabel("EMA медленная")
    axes[1].set_title("Прибыль по решётке параметров (цвет)")
    plt.tight_layout()
    plot_path = PROJECT_ROOT / "data" / "method2_visual_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"График сохранён: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
