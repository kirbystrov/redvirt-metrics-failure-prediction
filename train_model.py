#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение модели прогнозирования отказов на основе метрик
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn не установлен, тепловая карта будет отрисована в базовом стиле matplotlib.")


class FailurePredictor:
    def __init__(self, data_file='clean_metrics.csv', target_host=None):
        self.data_file = data_file
        self.target_host = target_host
        self.df = None
        self.model = None
        self.features = None

    def load_data(self):
        print("Загрузка данных...")
        self.df = pd.read_csv(self.data_file, index_col='timestamp', parse_dates=True)
        if self.target_host:
            print(f"Фильтрация по хосту: {self.target_host}")
            self.df = self.df[self.df['host_name'] == self.target_host]
            print(f"Загружено {len(self.df)} записей для хоста {self.target_host}")
        else:
            print(f"Загружено {len(self.df)} записей для всех хостов")
        return self.df

    def create_failure_labels(self, failure_times, window_minutes=15):
        print(f"\nСоздание меток с окном упреждения {window_minutes} минут...")
        self.df['failure_soon'] = 0
        for ft_str in failure_times:
            ft = pd.to_datetime(ft_str)
            window_start = ft - pd.Timedelta(minutes=window_minutes)
            mask = (self.df.index >= window_start) & (self.df.index < ft)
            self.df.loc[mask, 'failure_soon'] = 1
            print(f"  Отказ в {ft}: отмечено {mask.sum()} точек")
        print(f"\nБаланс классов:")
        print(self.df['failure_soon'].value_counts())
        return self.df

    def prepare_features(self):
        self.features = [
            'cpu_total', 'mem_percent', 'cpu_load', 'swap_percent',
            'cpu_sma_5', 'cpu_delta', 'mem_delta'
        ]
        missing = [f for f in self.features if f not in self.df.columns]
        if missing:
            print(f"Предупреждение: отсутствуют признаки {missing}")
            self.features = [f for f in self.features if f in self.df.columns]
        print(f"\nИспользуемые признаки: {', '.join(self.features)}")
        return self.features

    def train(self, test_size=0.3):
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("="*60)

        X = self.df[self.features].fillna(0)
        y = self.df['failure_soon']

        if len(y.unique()) < 2:
            print("Ошибка: в данных только один класс! Невозможно обучить модель.")
            return None

        # Обычное разбиение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Размер обучающей выборки: {len(X_train)}")
        print(f"Размер тестовой выборки: {len(X_test)}")

        # ===== 1. Стратифицированная K‑Fold кросс‑валидация =====
        print("\n=== Стратифицированная K‑Fold кросс‑валидация (n_splits=5) ===")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_tr_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            model_fold = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=10,
                class_weight='balanced', random_state=42
            )
            model_fold.fit(X_tr_fold, y_tr_fold)
            y_pred_fold = model_fold.predict_proba(X_val_fold)[:, 1]
            auc_fold = roc_auc_score(y_val_fold, y_pred_fold)
            cv_auc_scores.append(auc_fold)
            print(f"  Fold {fold+1}: ROC-AUC = {auc_fold:.3f}")
        print(f"Среднее: {np.mean(cv_auc_scores):.3f} ± {np.std(cv_auc_scores):.3f}")

        # ===== 2. Временная кросс-валидация (TimeSeriesSplit) =====
        print("\n=== Временная кросс-валидация TimeSeriesSplit (n_splits=5) ===")
        tscv = TimeSeriesSplit(n_splits=3)
        ts_auc_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr_ts = X.iloc[train_idx]
            X_val_ts = X.iloc[val_idx]
            y_tr_ts = y.iloc[train_idx]
            y_val_ts = y.iloc[val_idx]
            if len(y_val_ts.unique()) < 2:
                print(f"  Fold {fold+1}: пропущен (только один класс в тестовой выборке)")
                continue
            model_ts = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=10,
                class_weight='balanced', random_state=42
            )
            model_ts.fit(X_tr_ts, y_tr_ts)
            y_proba_ts = model_ts.predict_proba(X_val_ts)
            if y_proba_ts.shape[1] < 2:
                print(f"  Fold {fold+1}: пропущен (модель предсказывает только один класс)")
                continue
            y_pred_ts = y_proba_ts[:, 1]
            auc_ts = roc_auc_score(y_val_ts, y_pred_ts)
            ts_auc_scores.append(auc_ts)
            print(f"  Fold {fold+1}: ROC-AUC = {auc_ts:.3f}")
        if ts_auc_scores:
            print(f"Среднее TimeSeries CV: {np.mean(ts_auc_scores):.3f} ± {np.std(ts_auc_scores):.3f}")
        else:
            print("Недостаточно фолдов с обоими классами для временной кросс-валидации")

        # ===== 3. Обучение основной модели =====
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            min_samples_split=50,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Оценка на тестовой выборке
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        proba_class1 = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, 0]

        print("\nРезультаты на тестовой выборке:")
        print(classification_report(y_test, y_pred))
        print("Матрица ошибок:")
        print(confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, proba_class1)
        roc_auc_val = auc(fpr, tpr)
        print(f"\nROC-AUC (Random Forest): {roc_auc_val:.3f}")

        # ===== 4. Бутстрэп =====
        print("\n=== 95% доверительный интервал ROC-AUC (бутстрэп, 1000 итераций) ===")
        n_bootstrap = 1000
        auc_bootstrap = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_test), size=len(y_test), replace=True)
            auc_bs = roc_auc_score(y_test.iloc[idx], proba_class1[idx])
            auc_bootstrap.append(auc_bs)
        ci_low = np.percentile(auc_bootstrap, 2.5)
        ci_high = np.percentile(auc_bootstrap, 97.5)
        print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

        # ===== 5. Время инференса =====
        print("\n=== Время инференса модели (1000 предсказаний) ===")
        sample = X_test.iloc[:1]
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            self.model.predict_proba(sample)
            times.append(time.perf_counter() - start)
        print(f"Среднее время предсказания: {np.mean(times)*1000:.3f} мс")

        # Важность признаков
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nВажность признаков:")
        print(importance.to_string(index=False))

        # === Графики ===
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance for Failure Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_val:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=150)
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        if HAS_SEABORN:
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Failure'],
                        yticklabels=['Normal', 'Failure'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150)
            plt.close()
        else:
            plt.figure(figsize=(6,5))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Normal', 'Failure'])
            plt.yticks(tick_marks, ['Normal', 'Failure'])
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150)
            plt.close()

        # Распределение вероятностей
        plt.figure(figsize=(8,5))
        plt.hist(proba_class1[y_test == 0], bins=30, alpha=0.5, label='Normal (class 0)')
        plt.hist(proba_class1[y_test == 1], bins=30, alpha=0.5, label='Failure (class 1)')
        plt.xlabel('Predicted probability of failure')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution of Predicted Probabilities')
        plt.tight_layout()
        plt.savefig('prob_distribution.png', dpi=150)
        plt.close()

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, proba_class1)
        avg_precision = average_precision_score(y_test, proba_class1)
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pr_curve.png', dpi=150)
        plt.close()

        # Временной ряд предсказаний
        if isinstance(X_test.index, pd.DatetimeIndex):
            plt.figure(figsize=(12,5))
            plt.plot(X_test.index, y_test, 'o', label='Actual failure_soon', markersize=2)
            plt.plot(X_test.index, proba_class1, 'x', label='Predicted probability', markersize=2)
            plt.xlabel('Time')
            plt.ylabel('Probability / Class')
            plt.legend()
            plt.title('Predictions over time (test set)')
            plt.tight_layout()
            plt.savefig('predictions_timeseries.png', dpi=150)
            plt.close()

        # Сравнение с базовыми методами
        print("\n=== Сравнение с базовыми методами ===")
        for name, clf in [('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
                          ('SVM (линейное ядро)', SVC(kernel='linear', probability=True, random_state=42))]:
            clf.fit(X_train, y_train)
            y_proba_clf = clf.predict_proba(X_test)
            proba_clf = y_proba_clf[:, 1] if y_proba_clf.shape[1] == 2 else y_proba_clf[:, 0]
            auc_clf = roc_auc_score(y_test, proba_clf)
            print(f"{name}: ROC-AUC = {auc_clf:.3f}")

        # Устойчивость к шуму
        print("\n=== Устойчивость к шуму ===")
        noise_levels = [0, 0.05, 0.10]
        for noise in noise_levels:
            X_noisy = X.copy()
            for col in X.columns:
                std = X[col].std()
                if std > 0:
                    X_noisy[col] += np.random.normal(0, noise * std, size=len(X))
            Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42, stratify=y)
            model_noise = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model_noise.fit(Xn_train, yn_train)
            y_proba_noise = model_noise.predict_proba(Xn_test)
            proba_noise = y_proba_noise[:, 1] if y_proba_noise.shape[1] == 2 else y_proba_noise[:, 0]
            auc_noise = roc_auc_score(yn_test, proba_noise)
            print(f"Шум {noise*100:.0f}%: ROC-AUC = {auc_noise:.3f}")

        # Устойчивость к пропущенным значениям
        print("\n=== Устойчивость к пропущенным значениям ===")
        missing_frac = [0, 0.05, 0.10]
        for miss in missing_frac:
            X_miss = X.copy()
            mask = np.random.random(size=X_miss.shape) < miss
            X_miss[mask] = np.nan
            X_filled = X_miss.fillna(X_miss.mean())
            Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_filled, y, test_size=0.3, random_state=42, stratify=y)
            model_miss = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model_miss.fit(Xm_train, ym_train)
            y_proba_miss = model_miss.predict_proba(Xm_test)
            proba_miss = y_proba_miss[:, 1] if y_proba_miss.shape[1] == 2 else y_proba_miss[:, 0]
            auc_miss = roc_auc_score(ym_test, proba_miss)
            print(f"Пропуски {miss*100:.0f}%: ROC-AUC = {auc_miss:.3f}")

        return self.model

    def evaluate_prediction_horizon(self, failure_times, windows=[5, 10, 15, 20, 30]):
        print("\n=== Зависимость ROC-AUC от окна упреждения ===")
        X = self.df[self.features].fillna(0)
        auc_results = []
        for w in windows:
            y_temp = pd.Series(0, index=self.df.index)
            for ft_str in failure_times:
                ft = pd.to_datetime(ft_str)
                ws = ft - pd.Timedelta(minutes=w)
                mask = (self.df.index >= ws) & (self.df.index < ft)
                y_temp[mask] = 1
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_temp, test_size=0.3, random_state=42, stratify=y_temp)
            m = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
            m.fit(X_tr, y_tr)
            y_proba_w = m.predict_proba(X_te)
            proba_w = y_proba_w[:, 1] if y_proba_w.shape[1] == 2 else y_proba_w[:, 0]
            auc_w = roc_auc_score(y_te, proba_w)
            auc_results.append(auc_w)
            print(f"Окно {w} мин: ROC-AUC = {auc_w:.3f}")
        plt.figure(figsize=(8, 5))
        plt.plot(windows, auc_results, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('Окно упреждения, мин')
        plt.ylabel('ROC-AUC')
        plt.title('Зависимость точности от горизонта предсказания')
        plt.grid(True)
        plt.savefig('auc_vs_horizon.png', dpi=150)
        plt.close()
        print("График сохранён как auc_vs_horizon.png")

    def save_model(self, filename='failure_model.joblib'):
        try:
            joblib.dump(self.model, filename)
            print(f"\nМодель сохранена в {filename}")
        except ImportError:
            print("Установите joblib: pip install joblib")


if __name__ == "__main__":
    predictor = FailurePredictor('clean_metrics.csv', target_host='redhost1.adris.local')

    try:
        predictor.load_data()
    except FileNotFoundError:
        print("Файл clean_metrics.csv не найден!")
        print("Сначала запустите analyze_metrics.py для обработки данных")
        exit(1)

    failure_times = [
        '2026-05-13 20:41:15', '2026-05-13 21:21:07', '2026-05-13 21:48:34',
        '2026-05-13 20:55:04', '2026-05-13 21:32:51', '2026-05-13 22:29:15',
        '2026-05-13 22:42:19', '2026-05-13 23:51:26',
        '2026-05-13 21:09:44', '2026-05-13 22:02:09', '2026-05-13 22:16:01',
        '2026-05-13 23:12:39', '2026-05-13 23:39:52',
        '2026-05-13 22:56:44', '2026-05-13 23:26:11',
    ]

    predictor.create_failure_labels(failure_times, window_minutes=15)
    predictor.prepare_features()
    predictor.train()
    predictor.evaluate_prediction_horizon(failure_times, windows=[5, 10, 15, 20, 30])
    predictor.save_model()

    print("\n" + "="*60)
    print("Обучение завершено!")
    print("Созданы файлы:")
    print("  - feature_importance.png")
    print("  - roc_curve.png")
    print("  - confusion_matrix.png")
    print("  - prob_distribution.png")
    print("  - pr_curve.png")
    print("  - auc_vs_horizon.png")
    if predictor.df.index.inferred_type == 'datetime64':
        print("  - predictions_timeseries.png")
    print("  - failure_model.joblib")
    print("="*60)
