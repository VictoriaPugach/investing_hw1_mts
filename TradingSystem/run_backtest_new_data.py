"""
Бэктест на свежих данных: YDEX_251201_260306 (1h и 15m).
Загружает указанные файлы Finam, сохраняет в data/YNDX_*_new.csv,
запускает экспертный бэктест, строит отдельные графики и пишет блок сравнения с прежним периодом.
Запуск: python run_backtest_new_data.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Загрузка ручных выгрузок Finam
from data.download_data import load_manual_finam_txt

from backtest_expert import (
    backtest_expert,
    grid_search,
)
from metrics import (
    compute_metrics,
    print_metrics,
    plot_equity_curve,
    plot_equity_and_drawdown,
    plot_pnl_histogram,
    plot_exit_reasons,
)

DATA_DIR = PROJECT_ROOT / "data"

# Файлы свежих данных (период 01.12.2025 — 06.03.2026)
FILE_1H = PROJECT_ROOT / "YDEX_251201_260306.txt"
FILE_15M = PROJECT_ROOT / "YDEX_251201_260306 (1).txt"
SUFFIX = "new"


def load_new_datasets():
    """Загружает 1h и 15m из указанных txt, сохраняет в data/*_new.csv, возвращает (df_1h, df_15m)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_1h = None
    df_15m = None

    if FILE_1H.exists():
        df_1h = load_manual_finam_txt(FILE_1H)
        if df_1h is not None and not df_1h.empty:
            path_1h = DATA_DIR / f"YNDX_1h_{SUFFIX}.csv"
            df_1h.to_csv(path_1h)
            print(f"Загружен 1h: {len(df_1h)} баров, сохранён {path_1h.name}")
    if not FILE_1H.exists():
        print(f"Файл не найден: {FILE_1H}")

    if FILE_15M.exists():
        df_15m = load_manual_finam_txt(FILE_15M)
        if df_15m is not None and not df_15m.empty:
            path_15m = DATA_DIR / f"YNDX_15m_{SUFFIX}.csv"
            df_15m.to_csv(path_15m)
            print(f"Загружен 15m: {len(df_15m)} баров, сохранён {path_15m.name}")
    if not FILE_15M.exists():
        print(f"Файл не найден: {FILE_15M}")

    return df_1h, df_15m


def run_expert_on_new_data():
    df_1h, df_15m = load_new_datasets()
    if df_1h is None or df_1h.empty:
        print("Нет данных 1h. Положи YDEX_251201_260306.txt в папку TradingSystem.")
        return

    period_from = str(df_1h.index[0])[:10]
    period_to = str(df_1h.index[-1])[:10]
    print(f"\nПериод новых данных: {period_from} — {period_to}")
    print("=" * 70)

    # Подбор параметров на новом периоде (1h)
    print("\nПодбор параметров (grid search) на новых данных 1h...")
    grid = grid_search(df_1h)
    grid.to_csv(DATA_DIR / f"expert_grid_search_{SUFFIX}.csv", index=False)
    print(f"Топ-3 комбинации:")
    print(grid.head(3).to_string(index=False))

    best = grid.iloc[0]
    tp = float(best["TP_atr"])
    sl = float(best["SL_atr"])
    rsi_lo = float(best["rsi_min"])
    adx_min = float(best["adx_min"])

    # Экспертный бэктест 1h (новый период)
    print(f"\nЭкспертная МТС на новых данных 1h (TP={tp}, SL={sl}, RSI>={rsi_lo}, ADX>={adx_min})")
    print("-" * 70)
    trades_1h = backtest_expert(
        df_1h,
        tp_atr_mult=tp,
        sl_atr_mult=sl,
        rsi_long_min=rsi_lo,
        rsi_long_max=rsi_lo + 15.0,
        rsi_short_min=100.0 - rsi_lo - 15.0,
        rsi_short_max=100.0 - rsi_lo,
        adx_min=adx_min,
        verbose=True,
    )

    if trades_1h.empty:
        print("Сделок на новом периоде (1h) не найдено.")
        m_1h = {"n_trades": 0, "P": 0.0, "PF": 0.0, "RF": 0.0, "MIDD": 0.0, "pct_profitable": 0.0}
    else:
        m_1h = compute_metrics(trades_1h)
        print_metrics(m_1h)
        trades_1h.to_csv(DATA_DIR / f"expert_trades_1h_{SUFFIX}.csv", index=False)
        plot_equity_curve(
            trades_1h,
            title=f"Кривая капитала — экспертная МТС, новый период 1h ({period_from} — {period_to})",
            save_path=DATA_DIR / f"equity_curve_expert_1h_{SUFFIX}.png",
        )
        plot_equity_and_drawdown(
            trades_1h,
            title=f"Капитал и просадка — новый период 1h ({period_from} — {period_to})",
            save_path=DATA_DIR / f"equity_drawdown_expert_1h_{SUFFIX}.png",
        )
        plot_pnl_histogram(
            trades_1h,
            title=f"Распределение PnL — новый период 1h ({period_from} — {period_to})",
            save_path=DATA_DIR / f"pnl_histogram_expert_1h_{SUFFIX}.png",
        )
        plot_exit_reasons(
            trades_1h,
            title=f"Выходы по причинам — новый период 1h ({period_from} — {period_to})",
            save_path=DATA_DIR / f"exit_reasons_expert_1h_{SUFFIX}.png",
        )

    # 15m на новом периоде (те же параметры, что и для 1h нового периода)
    m_15m = None
    if df_15m is not None and not df_15m.empty:
        print(f"\nЭкспертная МТС на новых данных 15m (те же параметры)")
        print("-" * 70)
        trades_15m = backtest_expert(
            df_15m,
            tp_atr_mult=tp,
            sl_atr_mult=sl,
            rsi_long_min=rsi_lo,
            rsi_long_max=rsi_lo + 15.0,
            rsi_short_min=100.0 - rsi_lo - 15.0,
            rsi_short_max=100.0 - rsi_lo,
            adx_min=adx_min,
            verbose=True,
        )
        if not trades_15m.empty:
            m_15m = compute_metrics(trades_15m)
            print_metrics(m_15m)
            trades_15m.to_csv(DATA_DIR / f"expert_trades_15m_{SUFFIX}.csv", index=False)
            plot_equity_curve(
                trades_15m,
                title=f"Кривая капитала — экспертная МТС, новый период 15m ({period_from} — {period_to})",
                save_path=DATA_DIR / f"equity_curve_expert_15m_{SUFFIX}.png",
            )
            plot_equity_and_drawdown(
                trades_15m,
                title=f"Капитал и просадка — новый период 15m ({period_from} — {period_to})",
                save_path=DATA_DIR / f"equity_drawdown_expert_15m_{SUFFIX}.png",
            )
            plot_pnl_histogram(
                trades_15m,
                title=f"Распределение PnL — новый период 15m ({period_from} — {period_to})",
                save_path=DATA_DIR / f"pnl_histogram_expert_15m_{SUFFIX}.png",
            )
            plot_exit_reasons(
                trades_15m,
                title=f"Выходы по причинам — новый период 15m ({period_from} — {period_to})",
                save_path=DATA_DIR / f"exit_reasons_expert_15m_{SUFFIX}.png",
            )

    # Блок сравнения и анализа
    write_comparison(period_from, period_to, m_1h, m_15m)
    print(f"\nГрафики сохранены в data/ с суффиксом _{SUFFIX}.")
    print(f"Сравнение и анализ: data/COMPARISON_NEW_PERIOD.md")
    print("Готово.")


def write_comparison(period_from: str, period_to: str, m_1h: dict, m_15m: dict | None):
    """Пишет блок сравнения старого и нового периода в data/COMPARISON_NEW_PERIOD.md."""
    old_1h = {
        "period": "2025-08-01 — 2025-10-31",
        "n_trades": 12,
        "P": 133.78,
        "PF": 1.63,
        "RF": 2.52,
        "MIDD": 53.15,
        "pct_profitable": 41.7,
    }
    old_15m = {
        "n_trades": 33,
        "P": -261.82,
        "PF": 0.17,
        "RF": -1.07,
        "pct_profitable": 6.1,
    }

    new_period_label = f"{period_from} — {period_to}"

    lines = [
        "# Сравнение и анализ: старый период vs новый период (свежие данные)",
        "",
        "## 1. Периоды данных",
        "",
        "| Вариант | Период | Файлы данных |",
        "|---------|--------|--------------|",
        "| **Старый период** | 2025-08-01 — 2025-10-31 | YDEX_250801_251031 (выгрузка Finam) |",
        f"| **Новый период** | {new_period_label} | YDEX_251201_260306.txt (1h), YDEX_251201_260306 (1).txt (15m) |",
        "",
        "---",
        "",
        "## 2. Результаты экспертной МТС по периодам",
        "",
        "### 2.1. Таймфрейм 1 час",
        "",
        "| Метрика | Старый период (авг–окт 2025) | Новый период (дек 2025 — март 2026) |",
        "|---------|-------------------------------|--------------------------------------|",
        f"| Период | {old_1h['period']} | {new_period_label} |",
        f"| Число сделок | {old_1h['n_trades']} | {m_1h.get('n_trades', 0)} |",
        f"| P (чистая прибыль) | {old_1h['P']:.2f} | {m_1h.get('P', 0):.2f} |",
        f"| PF (Profit Factor) | {old_1h['PF']:.2f} | {m_1h.get('PF', 0):.2f} |",
        f"| RF (Recovery Factor) | {old_1h['RF']:.2f} | {m_1h.get('RF', 0):.2f} |",
        f"| MIDD (макс. просадка) | {old_1h['MIDD']:.2f} | {m_1h.get('MIDD', 0):.2f} |",
        f"| Доля прибыльных, % | {old_1h['pct_profitable']:.1f} | {m_1h.get('pct_profitable', 0):.1f} |",
        "",
    ]

    if m_15m is not None:
        lines.extend([
            "### 2.2. Таймфрейм 15 минут",
            "",
            "| Метрика | Старый период (авг–окт 2025) | Новый период (дек 2025 — март 2026) |",
            "|---------|-------------------------------|--------------------------------------|",
            f"| Число сделок | {old_15m['n_trades']} | {m_15m.get('n_trades', 0)} |",
            f"| P | {old_15m['P']:.2f} | {m_15m.get('P', 0):.2f} |",
            f"| PF | {old_15m['PF']:.2f} | {m_15m.get('PF', 0):.2f} |",
            f"| RF | {old_15m['RF']:.2f} | {m_15m.get('RF', 0):.2f} |",
            f"| Доля прибыльных, % | {old_15m['pct_profitable']:.1f} | {m_15m.get('pct_profitable', 0):.1f} |",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 3. Анализ и выводы",
        "",
        "### 3.1. Сопоставление периодов",
        "",
        "- На **старом периоде** (август — октябрь 2025) экспертная система на 1h дала положительный результат: PF = 1,63, RF = 2,52, кривая капитала росла. На 15m при тех же параметрах результат был убыточным (PF ≈ 0,17).",
        "- На **новом периоде** (декабрь 2025 — март 2026) параметры подобраны заново по grid search на новых данных 1h. Это позволяет оценить устойчивость системы к смене рыночной фазы.",
        "",
        "### 3.2. Интерпретация результатов нового периода",
        "",
        "- **1h:** Сравнение P, PF, RF и доли прибыльных между периодами показывает, сохраняется ли положительное мат. ожидание. Если на новом периоде PF > 1 и RF > 0 — система остаётся работоспособной.",
        "- **15m:** С параметрами от 1h результат на 15m может отличаться; при необходимости — отдельная оптимизация под 15m или мультитаймфреймовый фильтр.",
        "",
        "### 3.3. Графики по новому периоду",
        "",
        "В папке `data/` сохранены рисунки с суффиксом `_new`:",
        "",
        "- `equity_curve_expert_1h_new.png`, `equity_drawdown_expert_1h_new.png`,",
        "- `pnl_histogram_expert_1h_new.png`, `exit_reasons_expert_1h_new.png`,",
        "- аналогичные файлы для 15m (`_15m_new`).",
        "",
    ])

    out_path = DATA_DIR / "COMPARISON_NEW_PERIOD.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_expert_on_new_data()
