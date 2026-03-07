"""
Тест МТС через реальные методы Finam Trade API: заявки и сделки.

Использует:
  POST /v1/accounts/{accountId}/orders — создание заявки
  GET  /v1/accounts/{accountId}/orders — список заявок (если есть в API)
  GET  /v1/accounts/{accountId}/trades — история сделок

Запуск: python finam_mts_trade.py [--list] [--trades] [--order] [--trade-5] [--symbol X] [--price P] [--cancel ID]
  --list       список заявок
  --trades     история сделок
  --order      одна тестовая заявка (SBER: мин. цена 308.66, по умолчанию 309 — без исполнения)
  --trade-5    5 лимитных заявок (цена должна быть в коридоре биржи: для SBER напр. 309–321, по умолчанию 318)
  --mts-signal одна заявка по сигналу вашей МТС (данные 15m + фильтр 1h); инструмент YDEX@MISX по данным YNDX
  --symbol X   инструмент (SBER@MISX, YDEX@MISX)
  --price P   цена лимита (для SBER укладываться в коридор биржи, напр. 309–321)
  --cancel ID  отменить заявку

Важно: заявки API принимает только в основную сессию Мосбиржи — с 10:02 до 18:00 по Москве.
Команду --order запускайте после 10:02 МСК.
"""

import os
import sys
import time
from datetime import datetime, timezone, timedelta
from datetime import time as dt_time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from finam_test_account import load_dotenv, get_jwt, get_token_details, get_account_info

BASE = "https://api.finam.ru/v1"

FINAM_TIMEFRAME = {"15m": "TIME_FRAME_M15", "1h": "TIME_FRAME_H1"}


def _fetch_ohlcv_api(secret: str, symbol: str, interval: str, days: int = 10, jwt: str | None = None):
    """
    Загружает актуальные свечи через Finam API v1 (GET /v1/instruments/{symbol}/bars).
    Требуется Authorization: Bearer <JWT>. JWT получается из secret через get_jwt(secret).
    symbol — тикер@MIC (YDEX@MISX). Возвращает DataFrame или None.
    """
    import pandas as pd
    tf = FINAM_TIMEFRAME.get(interval)
    if not tf or not secret:
        return None
    token = jwt or get_jwt(secret)
    if not token:
        return None
    end = datetime.now(timezone(timedelta(hours=3)))
    start = end - timedelta(days=days)
    params = {
        "timeframe": tf,
        "interval.startTime": start.strftime("%Y-%m-%dT00:00:00Z"),
        "interval.endTime": end.strftime("%Y-%m-%dT23:59:59Z"),
    }
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        r = requests.get(f"{BASE}/instruments/{symbol}/bars", params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    bars = data.get("bars") or []
    rows = []
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
        rows.append({
            "datetime": pd.to_datetime(ts),
            "open": open_v, "high": high_v, "low": low_v, "close": close_v, "volume": vol_v,
        })
    if rows:
        return pd.DataFrame(rows).set_index("datetime").sort_index()
    return None


def _load_ohlcv(symbol: str, timeframe: str):
    """Загружает OHLCV из data/{symbol}_{timeframe}.csv (и _new если есть)."""
    try:
        import pandas as pd
        path = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}.csv"
        path_new = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}_new.csv"
        dfs = []
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            dfs.append(df)
        if path_new.exists():
            df_new = pd.read_csv(path_new, index_col=0, parse_dates=True)
            dfs.append(df_new)
        if not dfs:
            return None
        out = pd.concat(dfs, axis=0)
        out = out[~out.index.duplicated(keep="first")]
        return out.sort_index()
    except Exception:
        return None


def _get_mts_signal(df15=None, df1h=None):
    """
    Сигнал МТС на последнем баре 15m (EMA+VWAP, фильтр 1h).
    Если df15/df1h не переданы — сначала пробуем загрузить из API (в main), иначе из CSV.
    Возвращает (side, price) или (None, price).
    """
    try:
        from indicators import add_indicators, long_signal, short_signal
    except ImportError:
        return None, None
    if df15 is None:
        df15 = _load_ohlcv("YNDX", "15m")
    if df1h is None:
        df1h = _load_ohlcv("YNDX", "1h")
    if df15 is None or df1h is None or len(df15) < 2:
        return None, None
    df1h = add_indicators(df1h.copy(), ema_fast=20, ema_slow=60, use_daily_vwap=True,
        atr_period=14, rsi_period=0, vol_ma_period=0, adx_period=0, slope_period=0)
    long_1h = (df1h["ema_fast"] > df1h["ema_slow"]) & (df1h["close"] > df1h["vwap"])
    short_1h = (df1h["ema_fast"] < df1h["ema_slow"]) & (df1h["close"] < df1h["vwap"])
    h1_long_ok = long_1h.reindex(df15.index, method="ffill").fillna(False)
    h1_short_ok = short_1h.reindex(df15.index, method="ffill").fillna(False)
    df15 = add_indicators(df15.copy(), ema_fast=20, ema_slow=60, ema_trend=100,
        use_daily_vwap=True, atr_period=14, rsi_period=0, vol_ma_period=0, adx_period=0, slope_period=0)
    last, prev = df15.iloc[-1], df15.iloc[-2]
    hour = last.name.hour if hasattr(last.name, "hour") else 12
    if hour < 10 or hour > 18:
        return None, round(float(last["close"]), 2)
    long_now, short_now = long_signal(last), short_signal(last)
    long_prev, short_prev = long_signal(prev), short_signal(prev)
    price = round(float(last["close"]), 2)
    if long_now and not long_prev and h1_long_ok.iloc[-1]:
        return "long", price
    if short_now and not short_prev and h1_short_ok.iloc[-1]:
        return "short", price
    return None, price


def _headers(jwt: str) -> dict:
    return {"Authorization": jwt, "Accept": "application/json", "Content-Type": "application/json"}


def get_orders(jwt: str, account_id: str) -> list | None:
    """GET /v1/accounts/{accountId}/orders — список заявок по счёту."""
    url = f"{BASE}/accounts/{account_id}/orders"
    try:
        r = requests.get(url, headers=_headers(jwt), timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "orders" in data:
                return data["orders"]  # пустой [] тоже возвращаем как есть
            return data.get("order") or []
        return []
    except Exception as e:
        print(f"Ошибка запроса заявок: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            print("Ответ:", getattr(resp, "text", "")[:600])
        return None


def get_trades(
    jwt: str,
    account_id: str,
    limit: int = 20,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list | None:
    """GET /v1/accounts/{accountId}/trades — история сделок. API требует interval (startTime/endTime)."""
    url = f"{BASE}/accounts/{account_id}/trades"
    end_time = end_time or int(time.time())
    start_time = start_time or (end_time - 30 * 24 * 3600)  # последние 30 дней
    params = {"limit": limit, "startTime": start_time, "endTime": end_time}
    try:
        r = requests.get(url, headers=_headers(jwt), params=params, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("trades") or data.get("data") or []
    except Exception as e:
        print(f"Ошибка запроса сделок: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            print("Ответ:", getattr(resp, "text", "")[:600])
        return None


def create_order(
    jwt: str,
    account_id: str,
    symbol: str = "YDEX@MISX",
    quantity_value: str = "1",
    side: str = "SIDE_BUY",
    order_type: str = "ORDER_TYPE_LIMIT",
    time_in_force: str | None = None,
    limit_price_value: str = "3000",
) -> dict | None:
    """
    POST /v1/accounts/{accountId}/orders — создание заявки.
    Схема: symbol, quantity.value, side, type, limitPrice.value; timeInForce — по документации (UNSPECIFIED или не передавать).
    """
    url = f"{BASE}/accounts/{account_id}/orders"
    body = {
        "symbol": symbol,
        "quantity": {"value": quantity_value},
        "side": side,
        "type": order_type,
        "limitPrice": {"value": limit_price_value},
    }
    if time_in_force is not None:
        body["timeInForce"] = time_in_force
    try:
        r = requests.post(url, headers=_headers(jwt), json=body, timeout=15)
        out = None
        try:
            out = r.json()
        except Exception:
            out = {"_raw": r.text[:500]}
        if r.status_code not in (200, 201):
            print(f"Код ответа: {r.status_code}")
            print("Тело ответа:", out)
            msg = str(out.get("message", "")) if isinstance(out, dict) else ""
            if r.status_code == 400 and "10:02" in msg:
                print("Подсказка: заявки принимаются только с 10:02 по Москве. Запусти команду между 10:02 и 18:00 МСК.")
            if r.status_code == 400 and ("308.66" in msg or "не меньше" in msg):
                print("Подсказка: для SBER минимальная цена 308.66. Используй --price 309 или выше.")
            if r.status_code == 400 and ("не больше" in msg or "321" in msg):
                import re
                m = re.search(r"не больше\s+([\d.]+)", msg) or re.search(r"(\d+\.\d+)\s*\\x0c", msg)
                if m:
                    max_p = m.group(1)
                    print(f"Подсказка: максимальная цена для инструмента сейчас {max_p}. Используй --price с значением не выше (например --price {max_p}).")
                else:
                    print("Подсказка: биржа задаёт верхний предел цены. Используй --price 320 или меньше (в пределах коридора).")
            return None
        return out
    except Exception as e:
        print(f"Ошибка создания заявки: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            print("Ответ:", getattr(resp, "text", "")[:600])
        return None


def cancel_order(jwt: str, account_id: str, order_id: str) -> bool:
    """DELETE /v1/accounts/{accountId}/orders/{orderId} — отмена заявки."""
    url = f"{BASE}/accounts/{account_id}/orders/{order_id}"
    try:
        r = requests.delete(url, headers=_headers(jwt), timeout=15)
        if r.status_code in (200, 204):
            return True
        print(f"Отмена заявки: код {r.status_code}, ответ {r.text[:400]}")
        return False
    except Exception as e:
        print(f"Ошибка отмены заявки: {e}")
        return False


def main():
    load_dotenv()
    secret = (os.environ.get("FINAM_TRADE_TOKEN") or os.environ.get("FINAM_API_KEY") or "").strip()
    if not secret:
        print("Задай в .env FINAM_API_KEY или FINAM_TRADE_TOKEN.")
        sys.exit(1)

    jwt = get_jwt(secret)
    if not jwt:
        print("Не удалось получить JWT.")
        sys.exit(2)

    details = get_token_details(secret, jwt)
    if not details:
        print("Не удалось получить Token Details.")
        sys.exit(3)

    account_ids = details.get("account_ids") or details.get("accountIds") or []
    if not account_ids:
        print("Нет доступных счетов в Token Details.")
        sys.exit(4)

    account_id = str(account_ids[0]).strip()

    # --cancel ORDER_ID
    if "--cancel" in sys.argv:
        idx = sys.argv.index("--cancel")
        if idx + 1 < len(sys.argv):
            order_id = sys.argv[idx + 1]
            print(f"Отмена заявки {order_id}...")
            if cancel_order(jwt, account_id, order_id):
                print("Заявка отменена.")
            else:
                sys.exit(5)
        else:
            print("Укажи order_id: python finam_mts_trade.py --cancel <order_id>")
            sys.exit(1)
        return

    print(f"Счёт: {account_id}")
    print()

    only_list = "--list" in sys.argv
    only_trades = "--trades" in sys.argv
    do_order = "--order" in sys.argv
    do_trade_5 = "--trade-5" in sys.argv
    do_mts_signal = "--mts-signal" in sys.argv

    def _parse_price():
        if "--price" in sys.argv:
            idx = sys.argv.index("--price")
            if idx + 1 < len(sys.argv):
                return str(sys.argv[idx + 1]).strip()
        return None

    def _parse_symbol():
        if "--symbol" in sys.argv:
            idx = sys.argv.index("--symbol")
            if idx + 1 < len(sys.argv):
                return sys.argv[idx + 1].strip()
        return "SBER@MISX"

    if not only_trades and not do_order and not do_trade_5 and not do_mts_signal:
        info = get_account_info(jwt, account_id)
        if info:
            eq = info.get("equity") or {}
            cash = (info.get("cash") or [{}])[0] if info.get("cash") else {}
            print("Портфель: equity =", eq.get("value", eq), "  cash =", cash.get("units", ""), cash.get("currency_code", ""))
            print()

    if only_list or (not only_trades and not do_order and not do_trade_5 and not do_mts_signal):
        orders = get_orders(jwt, account_id)
        if orders is not None:
            print(f"Заявок по счёту: {len(orders)}")
            for o in (orders or [])[:15]:
                if isinstance(o, dict):
                    oid = o.get("order_id") or o.get("orderId") or o.get("id")
                    sym = o.get("symbol", "")
                    qty = o.get("quantity") or o.get("lots") or o.get("size") or o.get("qty")
                    state = o.get("state") or o.get("status") or o.get("orderState") or ""
                    if oid is None and qty is None and len(o) > 0:
                        print(f"  (ключи: {list(o.keys())})  {o}")
                    else:
                        print(f"  id={oid}  {sym}  qty={qty}  {state}")
                else:
                    print(" ", o)
        else:
            print("Список заявок недоступен или пуст (эндпоинт GET .../orders может отличаться).")
        print()

    if only_trades or (not only_list and not do_order and not do_trade_5 and not do_mts_signal):
        trades = get_trades(jwt, account_id)
        if trades is not None:
            print(f"Сделок (последние): {len(trades)}")
            for t in (trades or [])[:10]:
                if isinstance(t, dict):
                    print(f"  {t.get('symbol')}  {t.get('side')}  qty={t.get('size') or t.get('quantity')}  price={t.get('price')}  {t.get('timestamp', t.get('date'))}")
                else:
                    print(" ", t)
        else:
            print("История сделок недоступна или пуста.")
        print()

    if do_order:
        # Предупреждение: заявки принимаются только 10:02–18:00 МСК
        moscow = timezone(timedelta(hours=3))
        now_moscow = datetime.now(moscow).time()
        session_start = dt_time(10, 2, 0)
        session_end = dt_time(18, 0, 0)
        if now_moscow < session_start or now_moscow > session_end:
            print("Внимание: по правилам Мосбиржи заявки принимаются с 10:02 до 18:00 по Москве.")
            print(f"Сейчас по Москве: {now_moscow.strftime('%H:%M')}. Запустите --order после 10:02 МСК.")
            print()
        symbol = _parse_symbol()
        # Для SBER биржа требует цену не меньше ~308.66; 309 — заявка примется, но не исполнится (далеко от рынка)
        price = _parse_price() or ("309" if "SBER" in symbol else "3000")
        print(f"Выставление тестовой лимитной заявки: 1 лот {symbol}, покупка по {price} (без исполнения).")
        result = create_order(jwt, account_id, symbol=symbol, quantity_value="1", limit_price_value=price)
        if result:
            print("Ответ API:", result)
            oid = (result or {}).get("order_id") or (result or {}).get("id") or (result or {}).get("orderId")
            if oid and isinstance(oid, str):
                print(f"Заявка создана, order_id={oid}. Отменить: python finam_mts_trade.py --cancel {oid}")
        else:
            print("Не удалось создать заявку. Для SBER цена должна быть не меньше 308.66 (--price 309).")
    elif do_trade_5:
        # Режим «5 сделок»: выставляем 5 лимитных заявок по цене, при которой они исполнятся
        moscow = timezone(timedelta(hours=3))
        now_moscow = datetime.now(moscow).time()
        session_start = dt_time(10, 2, 0)
        session_end = dt_time(18, 0, 0)
        if now_moscow < session_start or now_moscow > session_end:
            print("Внимание: заявки принимаются с 10:02 до 18:00 по Москве.")
            print(f"Сейчас: {now_moscow.strftime('%H:%M')} МСК. Запустите после 10:02 МСК.")
            sys.exit(6)
        symbol = _parse_symbol()
        # Биржа задаёт коридор: мин и макс цена (напр. SBER 308.66–321.15). По умолчанию 318 — внутри коридора
        price = _parse_price() or ("318" if "SBER" in symbol else "3100")
        print(f"Режим --trade-5: 5 заявок по 1 лот {symbol}, покупка по {price} (цена в коридоре биржи).")
        print()
        placed = 0
        for i in range(5):
            result = create_order(jwt, account_id, symbol=symbol, quantity_value="1", limit_price_value=price)
            if result:
                placed += 1
                oid = (result or {}).get("order_id") or (result or {}).get("id") or (result or {}).get("orderId")
                print(f"  Заявка {i+1}/5 создана, order_id={oid}")
            else:
                print(f"  Заявка {i+1}/5 не принята (укажи цену в коридоре биржи: --price 318 или меньше, см. подсказку выше).")
            if i < 4:
                time.sleep(2)
        print()
        print(f"Выставлено заявок: {placed}/5. Проверить сделки: python finam_mts_trade.py --trades")
    elif do_mts_signal:
        # Одна заявка по сигналу МТС (15m + фильтр 1h). Данные: сначала через API, иначе из CSV
        moscow = timezone(timedelta(hours=3))
        now_moscow = datetime.now(moscow).time()
        if now_moscow < dt_time(10, 2, 0) or now_moscow > dt_time(18, 0, 0):
            print("Заявки принимаются с 10:02 до 18:00 МСК.")
            sys.exit(6)
        df15_api = _fetch_ohlcv_api(secret, "YDEX@MISX", "15m", days=10, jwt=jwt)
        df1h_api = _fetch_ohlcv_api(secret, "YDEX@MISX", "1h", days=10, jwt=jwt)
        used_api = False
        if df15_api is not None and not df15_api.empty and df1h_api is not None and not df1h_api.empty:
            print("Данные для МТС: загружены через Finam API (актуальные).")
            used_api = True
            side_signal, close_price = _get_mts_signal(df15=df15_api, df1h=df1h_api)
        else:
            print("Данные через API недоступны или пусты — используем CSV из data/.")
            side_signal, close_price = _get_mts_signal()
        symbol_mts = "YDEX@MISX"
        price_str = _parse_price() or (str(close_price) if close_price is not None else "3000")
        if side_signal is None:
            print("МТС: сигнала входа на последнем баре 15m нет.")
            if not used_api:
                print("Обновить данные: python data/download_data.py (или добавьте свежие YDEX*.txt в корень проекта).")
        else:
            side = "SIDE_BUY" if side_signal == "long" else "SIDE_SELL"
            print(f"МТС: сигнал {side_signal.upper()} на последнем баре 15m. Выставляю заявку: 1 лот {symbol_mts}, {side_signal} по {price_str}.")
            result = create_order(jwt, account_id, symbol=symbol_mts, quantity_value="1", side=side, limit_price_value=price_str)
            if result:
                oid = (result or {}).get("order_id") or (result or {}).get("id") or (result or {}).get("orderId")
                print(f"Заявка по МТС создана, order_id={oid}. Отменить: python finam_mts_trade.py --cancel {oid}")
            else:
                print("Заявка не принята (проверьте инструмент YDEX@MISX и цену; на демо возможен Symbol not found).")
    elif not only_list and not only_trades:
        print("Тестовая заявка: python finam_mts_trade.py --order")
        print("5 заявок в коридоре биржи: python finam_mts_trade.py --trade-5 [--price 318]")
        print("Одна заявка по сигналу МТС (15m+1h): python finam_mts_trade.py --mts-signal")

    print("\nГотово.")


if __name__ == "__main__":
    main()
