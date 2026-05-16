#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сбор метрик производительности хостов РЕД Виртуализации через REST API.

Скрипт запускается один раз и работает непрерывно:
- аутентифицируется через OAuth2,
- каждые 60 секунд опрашивает список хостов и собирает статистики,
- записывает полученные метрики в CSV-файл для последующего анализа.
"""

import requests
import time
import csv
from datetime import datetime

requests.packages.urllib3.disable_warnings()


# ------------------------------------------------------------
# 1. НАСТРОЙКИ (измените под вашу среду)
# ------------------------------------------------------------

# Адрес Engine, логин и пароль администратора
ENGINE_URL = "https://engine.adris.local"
USERNAME = "admin@internal"
PASSWORD = "P@ssw0rd"

# Эндпоинты API
TOKEN_URL = f"{ENGINE_URL}/ovirt-engine/sso/oauth/token"
HOSTS_URL = f"{ENGINE_URL}/ovirt-engine/api/hosts"

# Область доступа (обязательно ovirt-app-api)
SCOPE = "ovirt-app-api"

# Имя выходного CSV-файла
CSV_FILE = "metrics.csv"

# Список хостов, за которыми следим (id и имя)
# Получить id можно через GET /api/hosts после настройки аутентификации
HOSTS = [
    {"id": "6b471a9f-ea73-4975-bc1b-ef86d2dd5559", "name": "redhost1.adris.local"},
    {"id": "e9aa1090-e2b8-46ac-9ab7-39e7aa5652e7", "name": "redhost2.adris.local"}
]

# Поля, которые сохраняются в CSV
# Порядок колонок: временная метка, идентификатор хоста, имя хоста,
# затем метрики CPU, памяти, swap.
FIELD_NAMES = [
    "timestamp", "host_id", "host_name",
    "cpu_user", "cpu_system", "cpu_load",
    "memory_total", "memory_used", "memory_free",
    "swap_total", "swap_used", "swap_free"
]

# Интервал опроса в секундах (60 секунд – минимальная агрегация Data Warehouse)
SLEEP_INTERVAL = 20

# ------------------------------------------------------------
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------------------------------------------

def get_token():
    """
    Получение OAuth2 токена доступа (Resource Owner Password Credentials).
    Возвращает строку access_token или выбрасывает исключение.
    """
    data = {
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
        "scope": SCOPE
    }
    # verify=False – отключаем проверку SSL-сертификата для тестового стенда.
    # В реальной эксплуатации следует указать путь к сертификату CA.
    resp = requests.post(TOKEN_URL, data=data, verify=False,
                         headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()["access_token"]

def safe_get(stats, name, default=0):
    """
    Безопасное извлечение значения статистики из словаря.
    Если ключ отсутствует или значение None – возвращает default.
    """
    value = stats.get(name, default)
    return default if value is None else value

def get_host_stats(token, host_id):
    """
    Запрашивает статистики для одного хоста по его ID.
    Возвращает словарь {имя_метрики: значение} или None при ошибке.
    """
    url = f"{ENGINE_URL}/ovirt-engine/api/hosts/{host_id}/statistics"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    try:
        resp = requests.get(url, headers=headers, verify=False, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка HTTP при запросе статистик хоста {host_id}: {e}")
        return None

    data = resp.json()
    stats = {}
    # В ответе приходит список объектов statistic, каждый с полями name и values
    for stat in data.get("statistic", []):
        name = stat.get("name")
        try:
            # Значение находится по пути values.value[0].datum
            value = stat["values"]["value"][0]["datum"]
            stats[name] = value
        except (KeyError, IndexError, TypeError):
            # Если структура не соответствует ожидаемой, пропускаем
            continue
    return stats

# ------------------------------------------------------------
# 3. ОСНОВНАЯ ЛОГИКА
# ------------------------------------------------------------

def main():
    print("="*60)
    print("СБОР МЕТРИК РЕД ВИРТУАЛИЗАЦИИ (REST API)")
    print("="*60)
    print(f"Сбор будет выполняться каждые {SLEEP_INTERVAL} секунд.")
    print(f"Контролируемые хосты: {', '.join([h['name'] for h in HOSTS])}")
    print(f"Результат сохраняется в файл: {CSV_FILE}")
    print("Для остановки нажмите Ctrl+C.\n")

    # Создаём CSV-файл и записываем заголовок
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()

    # Получаем первый токен
    try:
        token = get_token()
        print("Токен доступа получен. Будет автоматически обновляться каждые 5 минут.")
    except Exception as e:
        print(f"Критическая ошибка: не удалось получить токен. {e}")
        return

    last_token_refresh = time.time()
    total_records = 0

    # Бесконечный цикл сбора данных
    while True:
        try:
            # Обновление токена (срок жизни ~10 минут, обновляем каждые 5)
            if time.time() - last_token_refresh > 300:
                token = get_token()
                last_token_refresh = time.time()
                print(f"[{datetime.now().isoformat()}] Токен обновлён.")

            current_time = datetime.now().isoformat()

            # Опрашиваем каждый хост из списка
            for host in HOSTS:
                stats = get_host_stats(token, host["id"])
                if stats is None:
                    print(f"[{current_time}] {host['name']}: не удалось собрать статистики (пропускаем цикл)")
                    continue

                # Извлекаем нужные метрики с безопасным доступом
                row = {
                    "timestamp": current_time,
                    "host_id": host["id"],
                    "host_name": host["name"],
                    "cpu_user": safe_get(stats, "cpu.current.user"),
                    "cpu_system": safe_get(stats, "cpu.current.system"),
                    "cpu_load": safe_get(stats, "cpu.load.avg.5m"),
                    "memory_total": safe_get(stats, "memory.total"),
                    "memory_used": safe_get(stats, "memory.used"),
                    "memory_free": safe_get(stats, "memory.free"),
                    "swap_total": safe_get(stats, "swap.total"),
                    "swap_used": safe_get(stats, "swap.used"),
                    "swap_free": safe_get(stats, "swap.free"),
                }

                # Дополнительная проверка: если какой-то обязательный параметр отсутствует,
                # всё равно пишем строку (значения по умолчанию 0 уже подставлены).
                try:
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
                        writer.writerow(row)
                except IOError as e:
                    print(f"Ошибка записи в CSV: {e}. Строка будет пропущена.")
                    continue

                total_records += 1

                # Вычисляем дополнительные показатели для вывода в консоль (только статистика)
                cpu_total = row['cpu_user'] + row['cpu_system']
                mem_percent = (row['memory_used'] / row['memory_total'] * 100) if row['memory_total'] > 0 else 0
                print(f"[{current_time}] {host['name']} | CPU: {cpu_total:.1f}% | MEM: {mem_percent:.1f}% | LOAD: {row['cpu_load']} | всего записей: {total_records}")

            # Пауза до следующего цикла сбора
            time.sleep(SLEEP_INTERVAL)

        except KeyboardInterrupt:
            print("\nСбор метрик остановлен пользователем.")
            print(f"Всего сохранено записей: {total_records}")
            break
        except Exception as e:
            # Не фатальная ошибка – логируем и продолжаем работу
            print(f"Неожиданная ошибка: {e}. Работа будет продолжена через {SLEEP_INTERVAL} секунд.")
            time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
