"""
Проверка работы с тестовым (демо) счётом Finam через Trade API.

Список счетов (account_id) получаем через POST /v1/sessions/details — в ответе поле accountIds.
  https://api.finam.ru/docs/rest/#tag/authservice/post/v1/sessions/details
В .env: FINAM_API_KEY= или FINAM_TRADE_TOKEN= (secret-токен с api.finam.ru/tokens).
Запуск: python finam_test_account.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv():
    try:
        from dotenv import load_dotenv as _load
        _load(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def get_jwt(secret: str) -> str | None:
    """Получить JWT по secret-токену (POST /v1/sessions)."""
    import requests
    url = "https://api.finam.ru/v1/sessions"
    try:
        r = requests.post(
            url,
            json={"secret": secret},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("token") or data.get("accessToken")
    except Exception as e:
        print(f"Ошибка получения JWT: {e}")
        return None


def get_token_details(secret: str, jwt_token: str) -> dict | None:
    """
    POST /v1/sessions/details — информация о токене, в т.ч. список счетов (accountIds).
    Документация: https://api.finam.ru/docs/rest/#tag/authservice/post/v1/sessions/details
    Тело: {"token": "<JWT>"}, заголовок Authorization: secret.
    """
    import requests
    url = "https://api.finam.ru/v1/sessions/details"
    try:
        r = requests.post(
            url,
            json={"token": jwt_token},
            headers={
                "Content-Type": "application/json",
                "Authorization": secret,
                "Accept": "application/json",
            },
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Ошибка запроса Token Details: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                print("Ответ:", getattr(resp, "text", "")[:500])
            except Exception:
                pass
        return None


def get_accounts(jwt_token: str) -> tuple[list | None, int | None]:
    """
    Пробует GET /v1/accounts.
    Возвращает (список счетов или None, status_code или None).
    При 404 эндпоинт у Finam может отсутствовать.
    """
    import requests
    url = "https://api.finam.ru/v1/accounts"
    try:
        r = requests.get(
            url,
            headers={"Authorization": jwt_token, "Accept": "application/json"},
            timeout=15,
        )
        sc = r.status_code
        if sc != 200:
            return None, sc
        data = r.json()
        if isinstance(data, list):
            return data, 200
        if isinstance(data, dict):
            out = data.get("accounts") or data.get("data") or [data]
            return out, 200
        return None, 200
    except Exception as e:
        resp = getattr(e, "response", None)
        sc = getattr(resp, "status_code", None) if resp else None
        if sc != 404:
            print(f"Ошибка запроса счетов: {e}")
            if resp is not None:
                try:
                    print("Ответ:", getattr(resp, "text", "")[:500])
                except Exception:
                    pass
        return None, sc


def get_account_info(jwt_token: str, account_id: str):
    """GET /v1/accounts/{account_id} — данные по одному счёту (портфель/инфо)."""
    import requests
    url = f"https://api.finam.ru/v1/accounts/{account_id}"
    try:
        r = requests.get(
            url,
            headers={"Authorization": jwt_token, "Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Ошибка запроса счёта {account_id}: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                print("Ответ:", getattr(resp, "text", "")[:500])
            except Exception:
                pass
        return None


def main():
    load_dotenv()
    print("Ключ берётся из .env (FINAM_TRADE_TOKEN или FINAM_API_KEY)")
    secret = os.environ.get("FINAM_TRADE_TOKEN") or os.environ.get("FINAM_API_KEY") or ""
    secret = secret.strip()
    if not secret:
        print("В .env не задан FINAM_TRADE_TOKEN и не задан FINAM_API_KEY.")
        print("Добавь в .env строку: FINAM_API_KEY=твой_токен_с_api.finam.ru/tokens")
        sys.exit(1)

    print("Получение JWT...")
    jwt_token = get_jwt(secret)
    if not jwt_token:
        print("Не удалось получить JWT. Проверьте токен в .env.")
        sys.exit(2)

    print("JWT получен. Запрос Token Details (POST /v1/sessions/details)...")
    details = get_token_details(secret, jwt_token)
    if details is not None:
        account_ids = details.get("account_ids") or details.get("accountIds") or []
        if isinstance(account_ids, list) and len(account_ids) > 0:
            print("\nДоступные счета (account_id из ответа Token Details):")
            for i, acc_id in enumerate(account_ids):
                # В ЛК формат КлФ-account_id
                display_id = f"КлФ-{acc_id}" if acc_id and not str(acc_id).startswith("КлФ") else acc_id
                print(f"  {i+1}. {acc_id}  (в ЛК: {display_id})")
            if details.get("created_at"):
                print(f"\nТокен создан: {details.get('created_at')}")
            if details.get("expires_at"):
                print(f"Истекает:      {details.get('expires_at')}")
            print("\nГотово. API доступен. Для запросов по счёту используйте accountId из списка выше.")
            # Опционально: запрос по первому счёту
            first_id = str(account_ids[0]).strip()
            if first_id:
                print(f"\nЗапрос данных по первому счёту ({first_id})...")
                info = get_account_info(jwt_token, first_id)
                if info is not None:
                    print("Данные по счёту:")
                    if isinstance(info, dict):
                        for k, v in list(info.items())[:15]:
                            print(f"  {k}: {v}")
                    else:
                        print(" ", info)
            return
        # details есть, но account_ids/accountIds пустой или нет такого поля
        print("\nОтвет Token Details получен, но account_ids пуст или отсутствует.")
        print("Ключи ответа:", list(details.keys())[:20])
    else:
        print("Не удалось получить Token Details.")

    # Запасной вариант: ручной FINAM_ACCOUNT_ID или старый GET /v1/accounts
    print("\nПробуем GET /v1/accounts...")
    accounts, status_code = get_accounts(jwt_token)

    if accounts is not None:
        print(f"\nНайдено счетов: {len(accounts)}")
        for i, acc in enumerate(accounts):
            if isinstance(acc, dict):
                name = acc.get("name") or acc.get("title") or acc.get("accountId") or acc.get("id")
                acc_id = acc.get("accountId") or acc.get("id") or acc.get("number")
                kind = acc.get("type") or acc.get("kind") or ""
                print(f"  {i+1}. {name} (id: {acc_id}, тип: {kind})")
            else:
                print(f"  {i+1}. {acc}")
        print("\nГотово. API доступен.")
        return

    account_id = (os.environ.get("FINAM_ACCOUNT_ID") or "").strip()
    if account_id:
        print("Список счетов недоступен. Запрос по счёту FINAM_ACCOUNT_ID...")
        info = get_account_info(jwt_token, account_id)
        if info is not None:
            print("\nДанные по счёту", account_id)
            if isinstance(info, dict):
                for k, v in list(info.items())[:20]:
                    print(f"  {k}: {v}")
            else:
                print(" ", info)
            print("\nГотово. API и доступ к счёту работают.")
            return
        sys.exit(3)

    if status_code == 404:
        print("\nЭндпоинт GET /v1/accounts вернул 404 — в документации API такого метода нет.")
        print("Все методы идут по пути /v1/accounts/{accountId}/... (см. https://api.finam.ru/docs/rest/ ).")
        print("Авторизация по токену и JWT при этом работают.")
        print("\nЧтобы проверить доступ к счёту, добавь в .env номер счёта из личного кабинета:")
        print("  FINAM_ACCOUNT_ID=номер_без_КлФ")
        print("Скрипт сделает GET /v1/accounts/{id} и покажет данные счёта. Документация: api.finam.ru/docs/rest")
    else:
        print("Не удалось получить список счетов.")
    sys.exit(3)


if __name__ == "__main__":
    main()
