#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сбор метрик производительности хостов РЕД Виртуализации через REST API
"""

import requests
import time
import csv
from datetime import datetime

# Отключаем предупреждения о небезопасном SSL
requests.packages.urllib3.disable_warnings()

# Параметры аутентификации
AUTH_URL = "https://engine.adris.local/ovirt-engine/sso/oauth/token"
USERNAME = "admin@internal"
PASSWORD = "P@ssw0rd"
SCOPE = "ovirt-app-api"

# Файл для записи метрик
CSV_FILE = "metrics.csv"

# Поля CSV
FIELD_NAMES = [
    "timestamp", "host_id", "host_name",
    "cpu_user", "cpu_system", "cpu_idle", "cpu_load",
    "memory_total", "memory_used", "memory_free",
    "ksm_cpu", "swap_total", "swap_used", "swap_free",
    "boot_time"
]

# ID хостов (из предыдущего запроса)
HOSTS = [
    {"id": "6b471a9f-ea73-4975-bc1b-ef86d2dd5559", "name": "redhost1.adris.local"},
    {"id": "e9aa1090-e2b8-46ac-9ab7-39e7aa5652e7", "name": "redhost2.adris.local"}
]

def get_token():
    """Получение OAuth токена"""
    data = {
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
        "scope": SCOPE
    }
    resp = requests.post(AUTH_URL, data=data, verify=False, 
                         headers={"Accept": "application/json"})
    if resp.status_code == 200:
        return resp.json()["access_token"]
    else:
        raise Exception(f"Ошибка получения токена: {resp.status_code} {resp.text}")

def get_host_stats(token, host_id):
    """Получение статистик хоста по его ID"""
    url = f"https://engine.adris.local/ovirt-engine/api/hosts/{host_id}/statistics"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    resp = requests.get(url, headers=headers, verify=False)
    if resp.status_code != 200:
        return None
    
    data = resp.json()
    stats = {}
    
    # Извлекаем значения по именам статистик
    for stat in data.get("statistic", []):
        name = stat.get("name", "")
        try:
            value = stat["values"]["value"][0]["datum"]
            stats[name] = value
        except (KeyError, IndexError, TypeError):
            stats[name] = None
    
    return stats

def main():
    print("Запуск сбора метрик...")
    print(f"Будут собираться метрики для хостов: {', '.join([h['name'] for h in HOSTS])}")
    
    # Инициализируем CSV с заголовком
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
    
    # Получаем начальный токен
    token = get_token()
    print(f"Токен получен. Будет обновляться каждые 5 минут.")
    
    last_token_refresh = time.time()
    count = 0
    
    while True:
        try:
            # Обновляем токен каждые 5 минут
            if time.time() - last_token_refresh > 300:
                token = get_token()
                last_token_refresh = time.time()
                print(f"[{datetime.now().isoformat()}] Токен обновлён")
            
            timestamp = datetime.now().isoformat()
            
            for host in HOSTS:
                stats = get_host_stats(token, host["id"])
                if stats:
                    row = {
                        "timestamp": timestamp,
                        "host_id": host["id"],
                        "host_name": host["name"],
                        "cpu_user": stats.get("cpu.current.user", 0),
                        "cpu_system": stats.get("cpu.current.system", 0),
                        "cpu_idle": stats.get("cpu.current.idle", 0),
                        "cpu_load": stats.get("cpu.load.avg.5m", 0),
                        "memory_total": stats.get("memory.total", 0),
                        "memory_used": stats.get("memory.used", 0),
                        "memory_free": stats.get("memory.free", 0),
                        "ksm_cpu": stats.get("ksm.cpu.current", 0),
                        "swap_total": stats.get("swap.total", 0),
                        "swap_used": stats.get("swap.used", 0),
                        "swap_free": stats.get("swap.free", 0),
                        "boot_time": stats.get("boot.time", 0)
                    }
                    
                    with open(CSV_FILE, mode='a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
                        writer.writerow(row)
                    
                    count += 1
                    cpu_total = row['cpu_user'] + row['cpu_system']
                    mem_percent = (row['memory_used'] / row['memory_total'] * 100) if row['memory_total'] > 0 else 0
                    print(f"[{timestamp}] {host['name']} | CPU: {cpu_total:.1f}% | MEM: {mem_percent:.1f}% | LOAD: {row['cpu_load']} | (всего: {count})")
                else:
                    print(f"[{timestamp}] {host['name']} | ОШИБКА получения статистик")
            
            time.sleep(10)  # Пауза 10 секунд
            
        except KeyboardInterrupt:
            print("\nСбор метрик остановлен пользователем")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
