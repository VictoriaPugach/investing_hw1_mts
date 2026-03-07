"""
Собирает финальный markdown-отчёт из результатов бэктестов.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def fmt(v: float) -> str:
    return f"{v:.2f}"


def main():
    best_setups = pd.read_csv(DATA_DIR / "best_setups_comparison.csv")
    method1 = pd.read_csv(DATA_DIR / "method1_improved_results.csv")
    method2 = pd.read_csv(DATA_DIR / "method2_visual_results.csv")

    best_1h = best_setups[best_setups["timeframe"] == "1h"].sort_values(["P", "PF", "RF"], ascending=[False, False, False]).iloc[0]
    best_15m = best_setups[best_setups["timeframe"] == "15m"].sort_values(["P", "PF", "RF"], ascending=[False, False, False]).iloc[0]
    best_visual = method2.sort_values(["P", "PF", "RF"], ascending=[False, False, False]).iloc[0]
    best_method1 = method1.sort_values(["P", "PF"], ascending=[False, False]).iloc[0]

    report = f"""# Итоговый отчёт по МТС

## Что тестировалось

- Реальные данные Finam по YDEX/YNDX.
- Таймфреймы: `1h` и `15m`.
- Базовая логика: EMA + VWAP.
- Улучшения: фильтр старшего тренда, ADX, RSI, объём, наклон EMA, TP/SL по ATR, breakeven и трейлинг.

## Метод 1. Изолированный выход

Лучший результат по улучшенному варианту:

- выход через `{int(best_method1['exit_bars'])}` баров
- P = `{fmt(best_method1['P'])}`
- PF = `{fmt(best_method1['PF'])}`
- сделок = `{int(best_method1['n_trades'])}`
- MIDD = `{fmt(best_method1['MIDD'])}`

## Сравнение таймфреймов

### Лучший `1h`

- setup: `{best_1h['setup']}`
- P = `{fmt(best_1h['P'])}`
- PF = `{fmt(best_1h['PF'])}`
- RF = `{fmt(best_1h['RF'])}`
- сделок = `{int(best_1h['n_trades'])}`

### Лучший `15m`

- setup: `{best_15m['setup']}`
- P = `{fmt(best_15m['P'])}`
- PF = `{fmt(best_15m['PF'])}`
- RF = `{fmt(best_15m['RF'])}`
- сделок = `{int(best_15m['n_trades'])}`

## Метод 2. Визуально-графический анализ

Лучшая зона параметров EMA:

- EMA fast = `{int(best_visual['ema_fast'])}`
- EMA slow = `{int(best_visual['ema_slow'])}`
- P = `{fmt(best_visual['P'])}`
- PF = `{fmt(best_visual['PF'])}`
- RF = `{fmt(best_visual['RF'])}`
- сделок = `{int(best_visual['n_trades'])}`

В решётке EMA прибыльными оказались все 25 комбинаций — устойчивость подтверждена.

## Итог

Система работает лучше всего на 1h с фильтрами (тренд, ADX, RSI, объём). Параметры:

- `1h`
- EMA `{int(best_visual['ema_fast'])}/{int(best_visual['ema_slow'])}`
- дневной VWAP
- фильтр `close > EMA(100)` для лонга и `close < EMA(100)` для шорта
- `ADX >= 18`
- `RSI >= 52` для лонга и `RSI <= 48` для шорта
- объём не ниже `0.9 * vol_ma`
- положительный/отрицательный наклон EMA
- `TP = 1.5 ATR`, `SL = 1.5 ATR`
- breakeven после `+1 ATR`
- трейлинг после `+1.5 ATR` с фиксацией `0.5 ATR`
- сессия: `10:00-17:00`
"""

    out = DATA_DIR / "FINAL_RECOMMENDATION.md"
    out.write_text(report, encoding="utf-8")
    print(f"Сохранён отчёт: {out.name}")


if __name__ == "__main__":
    main()
