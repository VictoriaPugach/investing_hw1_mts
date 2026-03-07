# Механическая торговая система (МТС)

Актив: Яндекс (YDEX/YNDX), Мосбиржа. Таймфреймы: 15m, 1h. Стратегия: EMA 20/60 + VWAP, выход по TP/SL (ATR) или фиксированным N барам.

## Быстрый старт

```bash
pip install -r requirements.txt
cp .env.example .env
# Добавить FINAM_API_KEY в .env (api.finam.ru/tokens)
python data/download_data.py
python backtest_isolated.py
```

## МТС — бэктест

**Загрузка данных:** `python data/download_data.py`  
Источники по приоритету: Finam API → ручные YDEX*.txt в корне → finam-export → Yahoo. Результат: `data/YNDX_15m.csv`, `data/YNDX_1h.csv`.

**Метод 1 (изолированный выход):** `python backtest_isolated.py`  
Вход по EMA + VWAP, выход через N баров. Метрики в `data/method1_isolated_results.csv`, кривые капитала — в `data/*.png`.

**Метод 2 (сетка параметров):** `python backtest_expert.py`  
Сетка TP/SL по ATR. Таблицы и графики в `data/`.

**Визуальный перебор:** `python backtest_visual.py`  
Зависимость результата от параметров EMA.

## Finam Trade API — торговля

Нужен secret-токен с [api.finam.ru/tokens](https://api.finam.ru/tokens) в `.env` как `FINAM_API_KEY`.

**Проверка доступа:** `python finam_test_account.py` — счета, equity.

**Заявки:** Только 10:02–18:00 МСК (основная сессия Мосбиржи).

| Команда | Действие |
|---------|----------|
| `python finam_mts_trade.py --order` | Одна тестовая лимитная заявка (SBER) |
| `python finam_mts_trade.py --order --symbol YDEX@MISX --price 4500` | Заявка по Яндексу |
| `python finam_mts_trade.py --trade-5` | 5 заявок (для набора сделок) |
| `python finam_mts_trade.py --mts-signal` | Заявка по сигналу МТС (EMA+VWAP) |
| `python finam_mts_trade.py --list` | Список заявок |
| `python finam_mts_trade.py --trades` | История сделок |
| `python finam_mts_trade.py --cancel <id>` | Отмена заявки |

## Структура

```
TradingSystem/
├── data/download_data.py   — загрузка OHLCV
├── backtest_isolated.py    — метод 1 (выход по N барам)
├── backtest_expert.py      — метод 2 (TP/SL по ATR)
├── backtest_visual.py      — перебор параметров
├── finam_test_account.py   — проверка API
├── finam_mts_trade.py      — торговые скрипты
├── indicators.py, metrics.py
└── requirements.txt
```
