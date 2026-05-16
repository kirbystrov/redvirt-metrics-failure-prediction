#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ метрик для прогнозирования отказов узлов виртуализации
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from scipy.stats import shapiro, normaltest
warnings.filterwarnings('ignore')

# Настройка для графиков
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 8)

class MetricsAnalyzer:
    def __init__(self, csv_file='metrics.csv'):
        self.csv_file = csv_file
        self.df = None
        self.hosts = []

    def load_data(self):
        print("Загрузка данных...")
        self.df = pd.read_csv(self.csv_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.set_index('timestamp', inplace=True)
        self.hosts = self.df['host_name'].unique()
        print(f"Загружено {len(self.df)} записей для хостов: {', '.join(self.hosts)}")
        print(f"Период: с {self.df.index.min()} по {self.df.index.max()}")
        return self.df

    def calculate_derived_features(self, host_df):
        df = host_df.copy()
        df['cpu_total'] = df['cpu_user'] + df['cpu_system']
        df['mem_percent'] = df['memory_used'] / df['memory_total'] * 100
        df['swap_percent'] = df['swap_used'] / df['swap_total'] * 100 if df['swap_total'].iloc[0] > 0 else 0
        df['cpu_sma_5'] = df['cpu_total'].rolling(window=5, min_periods=1).mean()
        df['cpu_sma_15'] = df['cpu_total'].rolling(window=15, min_periods=1).mean()
        df['load_sma_5'] = df['cpu_load'].rolling(window=5, min_periods=1).mean()
        df['mem_sma_5'] = df['mem_percent'].rolling(window=5, min_periods=1).mean()
        df['cpu_delta'] = df['cpu_total'].diff().fillna(0)
        df['mem_delta'] = df['mem_percent'].diff().fillna(0)
        return df

    def plot_all_metrics(self, host_name):
        host_df = self.df[self.df['host_name'] == host_name].copy()
        host_df = self.calculate_derived_features(host_df)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        axes[0].plot(host_df.index, host_df['cpu_total'], label='CPU Usage (%)', linewidth=1)
        axes[0].plot(host_df.index, host_df['cpu_load'], label='Load Average', linewidth=1)
        axes[0].set_ylabel('CPU / Load')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'{host_name} - CPU Metrics')

        axes[1].plot(host_df.index, host_df['mem_percent'], label='Memory Used (%)', linewidth=1, color='orange')
        axes[1].set_ylabel('Memory (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Memory Usage')

        axes[2].plot(host_df.index, host_df['swap_percent'], label='Swap Used (%)', linewidth=1, color='red')
        axes[2].set_ylabel('Swap (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Swap Usage')

        axes[3].plot(host_df.index, host_df['cpu_total'], label='Raw', linewidth=0.5, alpha=0.5)
        axes[3].plot(host_df.index, host_df['cpu_sma_5'], label='5-min SMA', linewidth=1.5)
        axes[3].plot(host_df.index, host_df['cpu_sma_15'], label='15-min SMA', linewidth=1.5)
        axes[3].set_ylabel('CPU (%)')
        axes[3].set_xlabel('Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title('CPU - Moving Averages')

        plt.tight_layout()
        plt.savefig(f'{host_name}_metrics.png', dpi=150)
        plt.close()
        print(f"График сохранён как {host_name}_metrics.png")
        return host_df

    def detect_anomalies(self, host_name, calm_period_hours=13):
        host_df = self.df[self.df['host_name'] == host_name].copy()
        host_df = self.calculate_derived_features(host_df)

        calm_end = host_df.index.min() + timedelta(hours=calm_period_hours)
        calm_df = host_df[host_df.index < calm_end]

        if len(calm_df) < 10:
            calm_df = host_df
            print(f"  Используем все данные для определения порогов (мало данных)")

        # === Логарифмическое преобразование и тест на нормальность ===
        if len(calm_df) > 20:
            cpu_data = calm_df['cpu_total'].dropna()
            if len(cpu_data) > 3:
                cpu_log = np.log1p(cpu_data)
                stat_sh, p_shapiro = shapiro(cpu_log)
                print(f"  Shapiro-Wilk p-value для log(cpu_total+1) = {p_shapiro:.4f} (нормальность при p>0.05)")
                stat_nd, p_norm = normaltest(cpu_log)
                print(f"  Normaltest p-value = {p_norm:.4f}")

        # Вычисляем пороги (на логарифмической шкале для CPU, затем обратно)
        cpu_log_mean = np.mean(np.log1p(calm_df['cpu_total'].dropna()))
        cpu_log_std = np.std(np.log1p(calm_df['cpu_total'].dropna()))
        cpu_threshold = np.expm1(cpu_log_mean + 2 * cpu_log_std)
        mem_threshold = calm_df['mem_percent'].mean() + 2 * calm_df['mem_percent'].std()
        load_threshold = calm_df['cpu_load'].mean() + 2 * calm_df['cpu_load'].std()

        print(f"\n=== Пороги аномалий для {host_name} ===")
        print(f"  CPU порог: {cpu_threshold:.1f}%")
        print(f"  Memory порог: {mem_threshold:.1f}%")
        print(f"  Load Average порог: {load_threshold:.1f}")

        host_df['cpu_anomaly'] = host_df['cpu_total'] > cpu_threshold
        host_df['mem_anomaly'] = host_df['mem_percent'] > mem_threshold
        host_df['load_anomaly'] = host_df['cpu_load'] > load_threshold

        anomaly_count = host_df['cpu_anomaly'].astype(int) + host_df['mem_anomaly'].astype(int) + host_df['load_anomaly'].astype(int)
        host_df['anomaly'] = anomaly_count >= 2

        anomaly_pct = host_df['anomaly'].sum() / len(host_df) * 100
        print(f"\nОбнаружено аномалий: {host_df['anomaly'].sum()} из {len(host_df)} ({anomaly_pct:.1f}%)")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(host_df.index, host_df['cpu_total'], label='CPU %', linewidth=1)
        ax.fill_between(host_df.index, 0, host_df['cpu_total'],
                        where=host_df['anomaly'], color='red', alpha=0.3, label='Anomaly')
        ax.axhline(y=cpu_threshold, color='red', linestyle='--', label=f'CPU Threshold ({cpu_threshold:.0f}%)')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{host_name} - CPU Anomaly Detection')

        plt.tight_layout()
        plt.savefig(f'{host_name}_anomalies.png', dpi=150)
        plt.close()
        print(f"График аномалий сохранён как {host_name}_anomalies.png")
        return host_df

    def generate_statistics_report(self):
        print("\n" + "="*60)
        print("СТАТИСТИЧЕСКИЙ ОТЧЁТ ПО МЕТРИКАМ")
        print("="*60)
        for host_name in self.hosts:
            host_df = self.df[self.df['host_name'] == host_name].copy()
            host_df = self.calculate_derived_features(host_df)
            print(f"\n--- {host_name} ---")
            print(f"Всего записей: {len(host_df)}")
            print(f"Период: {host_df.index.min()} - {host_df.index.max()}")
            print("\nCPU:")
            print(f"  Средний: {host_df['cpu_total'].mean():.1f}%")
            print(f"  Максимальный: {host_df['cpu_total'].max():.1f}%")
            print(f"  95-й перцентиль: {host_df['cpu_total'].quantile(0.95):.1f}%")
            print("\nПамять:")
            print(f"  Средняя: {host_df['mem_percent'].mean():.1f}%")
            print(f"  Максимальная: {host_df['mem_percent'].max():.1f}%")
            print("\nLoad Average:")
            print(f"  Средний: {host_df['cpu_load'].mean():.2f}")
            print(f"  Максимальный: {host_df['cpu_load'].max():.2f}")

    def save_clean_data(self, output_file='clean_metrics.csv'):
        all_hosts_df = []
        for host_name in self.hosts:
            host_df = self.df[self.df['host_name'] == host_name].copy()
            host_df = self.calculate_derived_features(host_df)
            all_hosts_df.append(host_df)
        clean_df = pd.concat(all_hosts_df)
        clean_df.to_csv(output_file)
        print(f"\nОчищенные данные сохранены в {output_file}")
        return clean_df


if __name__ == "__main__":
    analyzer = MetricsAnalyzer('metrics.csv')
    try:
        analyzer.load_data()
    except FileNotFoundError:
        print("Файл metrics.csv не найден!")
        print("Сначала запустите сбор метрик: python3 collect_metrics.py")
        exit(1)

    analyzer.generate_statistics_report()

    for host in analyzer.hosts:
        print(f"\n{'='*60}")
        print(f"Анализ хоста: {host}")
        print("="*60)
        analyzer.plot_all_metrics(host)
        analyzer.detect_anomalies(host, calm_period_hours=13)

    analyzer.save_clean_data('clean_metrics.csv')

    print("\n" + "="*60)
    print("Анализ завершён!")
    print("Созданы файлы:")
    print("  - *_metrics.png - графики метрик")
    print("  - *_anomalies.png - графики аномалий")
    print("  - clean_metrics.csv - очищенные данные")
    print("="*60)
