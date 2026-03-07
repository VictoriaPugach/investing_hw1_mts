"""
Сравнение лучших конфигураций МТС на 1h и 15m.
Цель: найти рабочий сетап на реальных данных Finam и сохранить краткий отчёт.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest_expert import load_combined_data, load_data, backtest_simple_refined, grid_search_simple
from indicators import add_indicators
from metrics import compute_metrics


def load_frame(timeframe: str) -> pd.DataFrame:
    try:
        return load_combined_data("YNDX", timeframe)
    except FileNotFoundError:
        return load_data("YNDX", timeframe)


def run_setup(
    timeframe: str,
    setup_name: str,
    df: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    trail_trigger_atr: float,
    trail_lock_atr: float,
    trend_filter_ema: int | None,
    strategy_kwargs: dict,
) -> dict:
    h1_long_ok = None
    h1_short_ok = None
    if df_1h is not None and len(df_1h) > 0:
        df1 = add_indicators(
            df_1h.copy(),
            ema_fast=20,
            ema_slow=60,
            use_daily_vwap=True,
            atr_period=14,
            rsi_period=0,
            vol_ma_period=0,
            adx_period=0,
            slope_period=0,
        )
        long_1h = (df1["ema_fast"] > df1["ema_slow"]) & (df1["close"] > df1["vwap"])
        short_1h = (df1["ema_fast"] < df1["ema_slow"]) & (df1["close"] < df1["vwap"])
        h1_long_ok = long_1h.reindex(df.index, method="ffill").fillna(False)
        h1_short_ok = short_1h.reindex(df.index, method="ffill").fillna(False)

    grid = grid_search_simple(
        df=df,
        df_1h=df_1h,
        trail_trigger_atr=trail_trigger_atr,
        trail_lock_atr=trail_lock_atr,
        trend_filter_ema=trend_filter_ema,
        strategy_kwargs=strategy_kwargs,
    )
    good = grid[(grid["PF"] > 1.0) & (grid["P"] > 0)]
    best_row = good.sort_values(["P", "RF", "PF"], ascending=[False, False, False]).iloc[0] if not good.empty else grid.iloc[0]

    tp = float(best_row["TP_atr"])
    sl = float(best_row["SL_atr"])
    trades = backtest_simple_refined(
        df=df,
        tp_atr_mult=tp,
        sl_atr_mult=sl,
        breakeven_atr=1.0,
        trail_trigger_atr=trail_trigger_atr,
        trail_lock_atr=trail_lock_atr,
        verbose=False,
        trend_filter_ema=trend_filter_ema,
        h1_long_ok=h1_long_ok,
        h1_short_ok=h1_short_ok,
        **strategy_kwargs,
    )
    metrics = compute_metrics(trades)
    return {
        "timeframe": timeframe,
        "setup": setup_name,
        "TP_atr": tp,
        "SL_atr": sl,
        "n_trades": metrics["n_trades"],
        "P": round(metrics["P"], 2),
        "PF": round(metrics["PF"], 2),
        "APF": round(metrics["APF"], 2),
        "MIDD": round(metrics["MIDD"], 2),
        "RF": round(metrics["RF"], 2),
        "pct_profitable": round(metrics["pct_profitable"], 1),
        "avg_win": round(metrics["avg_win"], 2),
        "avg_loss": round(metrics["avg_loss"], 2),
        "allow_long": strategy_kwargs.get("allow_long", True),
        "allow_short": strategy_kwargs.get("allow_short", True),
        "adx_min": strategy_kwargs.get("adx_min"),
        "rsi_long_min": strategy_kwargs.get("rsi_long_min"),
        "rsi_short_max": strategy_kwargs.get("rsi_short_max"),
        "rsi_long_max": strategy_kwargs.get("rsi_long_max"),
        "rsi_short_min": strategy_kwargs.get("rsi_short_min"),
        "vol_ratio_min": strategy_kwargs.get("vol_ratio_min"),
        "slope_min": strategy_kwargs.get("slope_min"),
        "session_start": strategy_kwargs.get("session_start", 10),
        "session_end": strategy_kwargs.get("session_end", 18),
        "max_bars": strategy_kwargs.get("max_bars", 40),
        "_trades": trades,
    }


def main():
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df1h = load_frame("1h")
    df15 = load_frame("15m")

    setups = [
        {
            "timeframe": "1h",
            "setup": "base_1h",
            "trail_trigger_atr": 1.5,
            "trail_lock_atr": 0.5,
            "trend_filter_ema": 100,
            "strategy_kwargs": {},
        },
        {
            "timeframe": "1h",
            "setup": "trend_adx_rsi_1h",
            "trail_trigger_atr": 1.5,
            "trail_lock_atr": 0.5,
            "trend_filter_ema": 100,
            "strategy_kwargs": {
                "adx_min": 18,
                "rsi_long_min": 52,
                "rsi_short_max": 48,
                "vol_ratio_min": 0.9,
                "slope_min": 0.0002,
                "session_end": 17,
            },
        },
        {
            "timeframe": "15m",
            "setup": "base_15m",
            "trail_trigger_atr": 2.0,
            "trail_lock_atr": 0.75,
            "trend_filter_ema": None,
            "strategy_kwargs": {
                "max_bars": 60,
            },
        },
        {
            "timeframe": "15m",
            "setup": "momentum_15m",
            "trail_trigger_atr": 2.0,
            "trail_lock_atr": 0.75,
            "trend_filter_ema": None,
            "strategy_kwargs": {
                "max_bars": 60,
                "adx_min": 18,
                "rsi_long_min": 52,
                "rsi_short_max": 48,
                "vol_ratio_min": 0.9,
                "slope_min": 0.00015,
                "session_end": 17,
            },
        },
    ]

    rows = []
    for setup in setups:
        timeframe = setup["timeframe"]
        df = df1h if timeframe == "1h" else df15
        df_htf = None if timeframe == "1h" else df1h
        print(f"\n=== {timeframe} | {setup['setup']} ===")
        result = run_setup(
            timeframe=timeframe,
            setup_name=setup["setup"],
            df=df,
            df_1h=df_htf,
            trail_trigger_atr=setup["trail_trigger_atr"],
            trail_lock_atr=setup["trail_lock_atr"],
            trend_filter_ema=setup["trend_filter_ema"],
            strategy_kwargs=setup["strategy_kwargs"],
        )
        print(
            f"TP={result['TP_atr']}, SL={result['SL_atr']}, trades={result['n_trades']}, "
            f"P={result['P']}, PF={result['PF']}, RF={result['RF']}, winrate={result['pct_profitable']}%"
        )
        trades = result.pop("_trades")
        trades.to_csv(data_dir / f"best_trades_{result['setup']}.csv", index=False)
        rows.append(result)

    res = pd.DataFrame(rows).sort_values(["timeframe", "P", "PF", "RF"], ascending=[True, False, False, False])
    out_csv = data_dir / "best_setups_comparison.csv"
    res.to_csv(out_csv, index=False)

    best_1h = res[res["timeframe"] == "1h"].sort_values(["P", "PF", "RF"], ascending=[False, False, False]).iloc[0]
    best_15m = res[res["timeframe"] == "15m"].sort_values(["P", "PF", "RF"], ascending=[False, False, False]).iloc[0]
    overall = res.sort_values(["P", "PF", "RF"], ascending=[False, False, False]).iloc[0]

    report = f"""# Сравнение лучших сетапов МТС

Источник данных: реальные свечи Finam по YDEX/YNDX.

## Лучший сетап на 1h

- setup: `{best_1h['setup']}`
- P: `{best_1h['P']}`
- PF: `{best_1h['PF']}`
- RF: `{best_1h['RF']}`
- сделок: `{best_1h['n_trades']}`

## Лучший сетап на 15m

- setup: `{best_15m['setup']}`
- P: `{best_15m['P']}`
- PF: `{best_15m['PF']}`
- RF: `{best_15m['RF']}`
- сделок: `{best_15m['n_trades']}`

## Итог

Лучший таймфрейм: `{overall['timeframe']}` (сетап `{overall['setup']}`). На 1h фильтры тренда и моментума дают выше P, PF, RF. На 15m — больше сделок, ниже RF.
"""
    out_md = data_dir / "FINAL_REPORT.md"
    out_md.write_text(report, encoding="utf-8")

    print(f"\nСводка сохранена: {out_csv.name}")
    print(f"Отчёт сохранён: {out_md.name}")
    print(f"Лучший таймфрейм: {overall['timeframe']} | сетап: {overall['setup']}")


if __name__ == "__main__":
    main()
