#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# Параметры
MODEL_FILE = "failure_model.joblib"
METRICS_FILE = "metrics.csv"
ALERT_THRESHOLD = 0.7
HOST_NAME = "redhost1.adris.local"  # или бери из последней строки

def get_latest_features(csv_file, host_name, lookback_minutes=60):
    """Берёт последние метрики и вычисляет признаки"""
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df = df[df['host_name'] == host_name]
    df = df.set_index('timestamp')
    
    # Берём последние lookback_minutes минут
    cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
    df = df[df.index >= cutoff]
    
    if len(df) < 5:
        return None
    
    # Вычисляем признаки (как в train_model.py)
    df['cpu_total'] = df['cpu_user'] + df['cpu_system']
    df['mem_percent'] = df['memory_used'] / df['memory_total'] * 100
    df['swap_percent'] = df['swap_used'] / df['swap_total'] * 100 if df['swap_total'].iloc[-1] > 0 else 0
    df['cpu_sma_5'] = df['cpu_total'].rolling(5, min_periods=1).mean()
    df['cpu_delta'] = df['cpu_total'].diff().fillna(0)
    df['mem_delta'] = df['mem_percent'].diff().fillna(0)
    
    # Берём средние за период (или последние значения)
    features = {
        'cpu_total': df['cpu_total'].mean(),
        'cpu_load': df['cpu_load'].mean(),
        'mem_percent': df['mem_percent'].mean(),
        'swap_percent': df['swap_percent'].mean(),
        'cpu_sma_5': df['cpu_sma_5'].mean(),
        'cpu_delta': df['cpu_delta'].mean(),
        'mem_delta': df['mem_delta'].mean(),
    }
    return features

def send_alert(probability, host_name):
    """Отправка предупреждения (разные способы)"""
    message = f"[ALERT] {host_name}: вероятность отказа {probability:.1%} за 15 минут"
    
    # Вариант 1: в консоль
    print(f"!!! {message} !!!")
    
    # Вариант 2: в Telegram (через бота)
    # requests.post(f"https://api.telegram.org/bot<TOKEN>/sendMessage", 
    #               json={"chat_id": "<CHAT_ID>", "text": message})
    
    # Вариант 3: в Slack/Teams (webhook)
    # requests.post("<WEBHOOK_URL>", json={"text": message})
    
    # Вариант 4: в системный лог
    # import syslog; syslog.syslog(syslog.LOG_WARNING, message)
    
    # Вариант 5: запуск скрипта миграции (вызов API Engine)
    # migrate_vm_to_safe_host(host_name)

def main():
    model = joblib.load(MODEL_FILE)
    print("Система мониторинга запущена. Ctrl+C для остановки.")
    
    while True:
        try:
            features = get_latest_features(METRICS_FILE, HOST_NAME)
            if features is None:
                print(f"[{datetime.now()}] Недостаточно данных")
                time.sleep(60)
                continue
            
            # Признаки в нужном порядке (как в train_model.py)
            feature_list = [
                features['cpu_total'],
                features['mem_percent'],
                features['cpu_load'],
                features['swap_percent'],
                features['cpu_sma_5'],
                features['cpu_delta'],
                features['mem_delta'],
            ]
            
            proba = model.predict_proba([feature_list])[0][1]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Вероятность отказа: {proba:.3f}")
            
            if proba > ALERT_THRESHOLD:
                send_alert(proba, HOST_NAME)
                
        except Exception as e:
            print(f"Ошибка: {e}")
        
        time.sleep(30)  # Проверка каждые 30 секунд

if __name__ == "__main__":
    main()
