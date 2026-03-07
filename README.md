# investing_hw1_mts

Механическая торговая система по акциям Яндекса (YDEX). Курс «Инвестиции».

## Установка

```bash
cd investing_hw1_mts
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Создать `.env` из `.env.example`, указать `FINAM_API_KEY` (токен с api.finam.ru/tokens).

## Загрузка данных

```bash
python data/download_data.py
```

Источники: Finam API → ручные YDEX*.txt → finam-export → Yahoo. Результат: `data/YNDX_15m.csv`, `data/YNDX_1h.csv`.

## Бэктесты

```bash
python backtest_isolated.py
python backtest_best.py
python backtest_visual.py
python build_final_report.py
```

Или одной командой:

```bash
python run_backtest.py
```

Файлы результатов: `data/method1_*.csv`, `data/best_setups_comparison.csv`, `data/method2_visual_results.csv`, `data/FINAL_RECOMMENDATION.md`. Текст для вставки в отчёт: `data/REPORT_INSERT.md`.

## Торговля через Finam API

```bash
python finam_test_account.py
python finam_mts_trade.py --order
python finam_mts_trade.py --mts-signal
```

Заявки принимаются только 10:02–18:00 МСК.
