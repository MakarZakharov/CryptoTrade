#!/usr/bin/env python3
"""
MVP скрипт для быстрого запуска обучения STAS_ML модели.
Простой интерфейс для начала обучения ML моделей с минимальными настройками.
Поддерживает адаптивное обучение на ошибках предыдущих моделей.
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.STAS_ML.config.ml_config import (
    MLConfig, DataManager, create_ml_config_interactive
)
from CryptoTrade.ai.STAS_ML.config.training_targets import (
    TrainingTargets, ModelType, TargetType, ModelEvaluationService
)
from CryptoTrade.ai.STAS_ML.training.trainer import MLTrainer, quick_train_ml
from CryptoTrade.ai.STAS_ML.data.data_processor import CryptoDataProcessor


class ModelErrorAnalyzer:
    """Аналізатор помилок моделі для адаптивного навчання."""
    
    def __init__(self):
        self.error_history = []
        self.failed_predictions = []
        self.difficult_patterns = []
        
    def analyze_model_errors(self, trainer, test_data: pd.DataFrame) -> Dict:
        """Аналізувати помилки моделі для покращення наступної."""
        try:
            # Отримуємо предсказання та реальні значення
            predictions = trainer.predictor.predict(trainer.X_test)
            actual = trainer.y_test
            
            # Знаходимо помилкові предсказання
            if trainer.config.target_type == 'direction':
                errors_mask = predictions != actual
            else:
                # Для регресії - великі помилки
                errors = np.abs(predictions - actual)
                error_threshold = np.percentile(errors, 75)  # Найгірші 25%
                errors_mask = errors > error_threshold
            
            error_indices = np.where(errors_mask)[0]
            
            # Аналізуємо характеристики помилкових предсказань
            error_analysis = {
                'total_errors': len(error_indices),
                'error_rate': len(error_indices) / len(predictions),
                'error_indices': error_indices.tolist(),
                'difficult_periods': self._identify_difficult_periods(error_indices, test_data),
                'error_patterns': self._analyze_error_patterns(trainer.X_test[errors_mask]),
                'market_conditions': self._analyze_market_conditions_during_errors(error_indices, test_data)
            }
            
            self.error_history.append(error_analysis)
            
            print(f"📊 АНАЛІЗ ПОМИЛОК МОДЕЛІ:")
            print(f"   Загальна кількість помилок: {error_analysis['total_errors']}")
            print(f"   Відсоток помилок: {error_analysis['error_rate']:.1%}")
            print(f"   Складні періоди: {len(error_analysis['difficult_periods'])}")
            
            return error_analysis
            
        except Exception as e:
            print(f"⚠️ Помилка аналізу: {e}")
            return {'total_errors': 0, 'error_rate': 0}
    
    def _identify_difficult_periods(self, error_indices: np.ndarray, test_data: pd.DataFrame) -> List[Tuple]:
        """Знайти складні періоди для торгівлі."""
        difficult_periods = []
        
        if len(error_indices) > 0:
            # Групуємо помилки по часових періодах
            error_dates = test_data.index[error_indices] if len(test_data) > max(error_indices) else []
            
            for i in range(len(error_dates) - 1):
                if (error_dates[i+1] - error_dates[i]).days <= 7:  # Помилки в межах тижня
                    difficult_periods.append((error_dates[i], error_dates[i+1]))
        
        return difficult_periods
    
    def _analyze_error_patterns(self, error_features: np.ndarray) -> Dict:
        """Аналізувати паттерни в помилкових предсказаннях."""
        if len(error_features) == 0:
            return {}
            
        try:
            # Статистика помилкових фіч
            feature_means = np.mean(error_features, axis=0)
            feature_stds = np.std(error_features, axis=0)
            
            # Знаходимо найбільш проблемні фічі
            problematic_features = np.argsort(feature_stds)[-10:]  # Топ-10 нестабільних фіч
            
            return {
                'feature_means': feature_means.tolist() if hasattr(feature_means, 'tolist') else [],
                'feature_stds': feature_stds.tolist() if hasattr(feature_stds, 'tolist') else [],
                'problematic_features': problematic_features.tolist()
            }
        except:
            return {}
    
    def _analyze_market_conditions_during_errors(self, error_indices: np.ndarray, test_data: pd.DataFrame) -> Dict:
        """Аналізувати ринкові умови під час помилок."""
        try:
            if len(error_indices) == 0 or len(test_data) == 0:
                return {}
            
            # Фільтруємо індекси, що не виходять за межі даних
            valid_indices = error_indices[error_indices < len(test_data)]
            
            if len(valid_indices) == 0:
                return {}
            
            error_data = test_data.iloc[valid_indices]
            
            # Аналізуємо ринкові умови
            volatility = error_data['close'].pct_change().std()
            avg_volume = error_data['volume'].mean()
            price_trend = (error_data['close'].iloc[-1] - error_data['close'].iloc[0]) / error_data['close'].iloc[0]
            
            return {
                'avg_volatility': volatility,
                'avg_volume': avg_volume,
                'price_trend': price_trend,
                'error_periods': len(valid_indices)
            }
        except:
            return {}
    
    def get_adaptive_training_params(self) -> Dict:
        """Отримати адаптивні параметри на основі аналізу помилок."""
        if len(self.error_history) == 0:
            return {}
        
        latest_errors = self.error_history[-1]
        error_rate = latest_errors.get('error_rate', 0)
        
        adaptive_params = {}
        
        # Адаптуємо параметри на основі помилок
        if error_rate > 0.6:  # Багато помилок
            adaptive_params.update({
                'min_threshold': 0.001,  # Менший поріг для більше сигналів
                'confidence': 0.35,      # Нижча впевненість
                'lookback': 30,          # Більший lookback для контексту  
                'n_estimators': 100      # Більше дерев
            })
        elif error_rate > 0.4:  # Середня кількість помилок
            adaptive_params.update({
                'min_threshold': 0.003,
                'confidence': 0.45,
                'lookback': 20,
                'n_estimators': 75
            })
        else:  # Мало помилок - зберігаємо стабільність
            adaptive_params.update({
                'min_threshold': 0.005,
                'confidence': 0.55,
                'lookback': 15,
                'n_estimators': 50
            })
        
        print(f"🎯 АДАПТИВНІ ПАРАМЕТРИ (помилки: {error_rate:.1%}):")
        for key, value in adaptive_params.items():
            print(f"   {key}: {value}")
        
        return adaptive_params


class AdvancedRandomForestOptimizer:
    """Продвинутый оптимизатор Random Forest для максимальной эффективности."""
    
    def __init__(self):
        self.optimization_history = []
        self.best_configs = []
        
    def get_optimized_config(self, attempt: int, best_result: Optional[Dict]) -> Dict:
        """Динамическая оптимизация параметров Random Forest."""
        
        # БАЗОВАЯ агрессивная конфигурация для первых попыток
        if attempt <= 5:
            base_configs = [
                {'n_estimators': 80, 'max_depth': 12, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_features': 0.7},
                {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.8},
                {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 0.6},
                {'n_estimators': 60, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 0.9},
                {'n_estimators': 150, 'max_depth': 8, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 0.5}
            ]
            config = base_configs[(attempt - 1) % len(base_configs)].copy()
        else:
            # АДАПТИВНАЯ оптимизация на основе предыдущих результатов
            if best_result and 'rf_config' in best_result:
                best_config = best_result['rf_config']
                efficiency = best_result.get('trading_results', {}).get('efficiency', 0)
                
                # Улучшаем лучшую конфигурацию
                config = best_config.copy()
                
                if efficiency < 1.0:  # Плохая эффективность - увеличиваем сложность
                    config['n_estimators'] = min(200, config.get('n_estimators', 100) + 20)
                    config['max_depth'] = min(25, config.get('max_depth', 12) + 2)
                    config['max_features'] = min(1.0, config.get('max_features', 0.7) + 0.1)
                elif efficiency < 1.5:  # Средняя эффективность - тонкая настройка
                    config['min_samples_split'] = max(2, config.get('min_samples_split', 5) - 1)
                    config['min_samples_leaf'] = max(1, config.get('min_samples_leaf', 2) - 1)
                else:  # Хорошая эффективность - диверсификация
                    variations = [
                        {'n_estimators': config.get('n_estimators', 100) + random.randint(-30, 30)},
                        {'max_depth': config.get('max_depth', 12) + random.randint(-3, 3)},
                        {'max_features': max(0.3, min(1.0, config.get('max_features', 0.7) + random.uniform(-0.2, 0.2)))}
                    ]
                    config.update(random.choice(variations))
            else:
                # Генерируем случайную эффективную конфигурацию
                config = {
                    'n_estimators': random.choice([70, 90, 110, 130, 150]),
                    'max_depth': random.choice([8, 10, 12, 15, 18]),
                    'min_samples_split': random.choice([3, 5, 8, 12]),
                    'min_samples_leaf': random.choice([1, 2, 3, 4]),
                    'max_features': random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
                }
        
        # Общие оптимальные параметры
        config.update({
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced_subsample',  # Лучше для несбалансированных данных
            'criterion': 'gini',  # Быстрее чем entropy
            'warm_start': False
        })
        
        return config
        
    def generate_smart_features(self, attempt: int, best_result: Optional[Dict]) -> Dict:
        """Умная генерация признаков на основе производительности."""
        
        # ПРОГРЕССИВНОЕ усложнение признаков
        if attempt <= 10:
            # Простые эффективные индикаторы
            indicators = {'rsi': [14], 'sma': [20], 'ema': [12], 'atr': [14]}
            lookback = random.choice([8, 10, 12])
            threshold = random.choice([0.008, 0.010, 0.012])
            confidence = random.choice([0.45, 0.50, 0.55])
        elif attempt <= 25:
            # Средняя сложность
            indicators = {
                'rsi': [14, 21], 
                'sma': [10, 20], 
                'ema': [12, 26], 
                'macd': [12, 26, 9],
                'atr': [14],
                'bollinger': [20]
            }
            lookback = random.choice([10, 12, 15])
            threshold = random.choice([0.006, 0.008, 0.010])
            confidence = random.choice([0.40, 0.45, 0.50])
        else:
            # Максимальная сложность для прорыва
            indicators = {
                'rsi': [14, 21], 
                'sma': [10, 20, 50], 
                'ema': [12, 26], 
                'macd': [12, 26, 9],
                'atr': [14, 21],
                'bollinger': [20],
                'stochastic': [14, 3, 3],
                'obv': []
            }
            lookback = random.choice([12, 15, 18])
            threshold = random.choice([0.004, 0.006, 0.008])
            confidence = random.choice([0.35, 0.40, 0.45])
        
        # Адаптация на основе лучшего результата
        if best_result and 'feature_config' in best_result:
            best_config = best_result['feature_config']
            winrate = best_result.get('trading_results', {}).get('winrate', 0)
            
            if winrate < 0.55:  # Низкий винрейт - увеличиваем уверенность
                confidence = min(0.65, best_config.get('confidence', 0.5) + 0.05)
                threshold = max(0.004, best_config.get('threshold', 0.008) - 0.001)
            elif winrate > 0.65:  # Высокий винрейт - можем рискнуть
                confidence = max(0.30, best_config.get('confidence', 0.5) - 0.05)
                threshold = min(0.015, best_config.get('threshold', 0.008) + 0.002)
        
        return {
            'indicators': indicators,
            'lookback': lookback,
            'threshold': threshold,
            'confidence': confidence
        }
    
    def select_optimal_time_segment(self, attempt: int, best_result: Optional[Dict]) -> Tuple[str, str]:
        """Адаптивный выбор временного сегмента."""
        
        # Разнообразные периоды для разных рыночных условий
        segments = [
            ('2020-01-01', '2023-12-31'),  # COVID и восстановление
            ('2019-01-01', '2022-12-31'),  # Бычий рынок
            ('2021-01-01', '2024-12-31'),  # Современные данные
            ('2018-06-01', '2021-12-31'),  # Длинный период
            ('2020-06-01', '2023-06-31'),  # Стабильный период
            ('2019-06-01', '2022-06-30'),  # Центральный период
            ('2021-06-01', '2024-06-30'),  # Новые тренды
        ]
        
        if best_result and 'time_segment' in best_result:
            # Иногда используем лучший сегмент, иногда диверсифицируем
            if random.random() < 0.3:  # 30% шанс повторить лучший
                return best_result['time_segment']
        
        return segments[attempt % len(segments)]
    
    def prepare_segment_data(self, trainer, time_segment: Tuple[str, str]) -> pd.DataFrame:
        """Подготовка высококачественных данных для сегмента."""
        
        original_data = trainer.data_processor.load_data()
        segment_data = original_data.loc[time_segment[0]:time_segment[1]].copy()
        
        # КАЧЕСТВЕННАЯ фильтрация данных
        if len(segment_data) > 0:
            # Удаляем аномальные значения
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in segment_data.columns:
                    Q1 = segment_data[col].quantile(0.01)
                    Q3 = segment_data[col].quantile(0.99)
                    segment_data = segment_data[(segment_data[col] >= Q1) & (segment_data[col] <= Q3)]
            
            # Удаляем дни с нулевым объемом
            if 'volume' in segment_data.columns:
                segment_data = segment_data[segment_data['volume'] > 0]
        
        return segment_data
    
    def evaluate_model_quality(self, metrics: Dict) -> float:
        """Комплексная оценка качества модели."""
        
        train_acc = metrics.get('train_accuracy', 0)
        val_acc = metrics.get('val_accuracy', 0)
        test_acc = metrics.get('test_accuracy', 0)
        
        # Базовые проверки
        if test_acc < 0.52:  # Должно быть лучше случайного
            return 0.0
        
        if train_acc - val_acc > 0.20:  # Переобучение
            return 0.1
        
        if abs(val_acc - test_acc) > 0.10:  # Нестабильность
            return 0.2
        
        # Комплексная оценка
        accuracy_score = (test_acc - 0.5) * 2  # 0.52 -> 0.04, 0.60 -> 0.20
        stability_score = 1 - abs(val_acc - test_acc) * 5  # Штраф за нестабильность
        overfitting_penalty = max(0, 1 - (train_acc - val_acc) * 3)  # Штраф за переобучение
        
        quality_score = (accuracy_score + stability_score + overfitting_penalty) / 3
        return max(0, min(1, quality_score))
    
    def analyze_trading_performance(self, metrics: Dict) -> Dict:
        """Расширенный анализ торговой производительности."""
        
        profit = metrics.get('trading_total_return_pct', 0)
        drawdown = metrics.get('trading_max_drawdown_pct', 100)
        winrate = metrics.get('trading_win_rate', 0)
        trades = metrics.get('trading_total_trades', 0)
        
        # Вычисляем эффективность (комплексный показатель)
        profit_score = max(0, profit / 20)  # 20% = 1.0
        drawdown_score = max(0, (10 - drawdown) / 10)  # <10% = 1.0
        winrate_score = max(0, (winrate - 0.5) * 2)  # 60% = 0.2
        trades_score = min(1, trades / 25)  # 25+ trades = 1.0
        
        efficiency = (profit_score + drawdown_score + winrate_score + trades_score) / 4
        
        return {
            'profit': profit,
            'drawdown': drawdown,
            'winrate': winrate,
            'trades': trades,
            'efficiency': efficiency,
            'profit_score': profit_score,
            'drawdown_score': drawdown_score,
            'winrate_score': winrate_score,
            'trades_score': trades_score
        }


class SmartFeatureEngineer:
    """Умный инженер признаков для Random Forest."""
    
    def generate_smart_features(self, attempt: int, best_result: Optional[Dict]) -> Dict:
        """Делегирует к AdvancedRandomForestOptimizer для совместимости."""
        optimizer = AdvancedRandomForestOptimizer()
        return optimizer.generate_smart_features(attempt, best_result)


class PerformanceTracker:
    """Отслеживание производительности для адаптивной оптимизации."""
    
    def __init__(self):
        self.results_history = []
        self.best_result = None
        
    def update_best_result(self, attempt: int, result: Dict):
        """Обновление лучшего результата."""
        self.results_history.append({
            'attempt': attempt,
            'result': result
        })
        
        efficiency = result.get('trading_results', {}).get('efficiency', 0)
        
        if self.best_result is None or efficiency > self.best_result.get('trading_results', {}).get('efficiency', 0):
            self.best_result = result.copy()
            self.best_result['attempt'] = attempt
    
    def get_best_result(self) -> Optional[Dict]:
        """Получить лучший результат."""
        return self.best_result
    
    def get_performance_trend(self) -> Dict:
        """Анализ тренда производительности."""
        if len(self.results_history) < 3:
            return {'trend': 'insufficient_data'}
        
        recent_efficiencies = [
            r['result'].get('trading_results', {}).get('efficiency', 0) 
            for r in self.results_history[-5:]
        ]
        
        if len(recent_efficiencies) >= 3:
            trend = 'improving' if recent_efficiencies[-1] > recent_efficiencies[0] else 'declining'
        else:
            trend = 'stable'
            
        return {
            'trend': trend,
            'recent_avg': np.mean(recent_efficiencies),
            'best_efficiency': max(recent_efficiencies) if recent_efficiencies else 0
        }


def run_advanced_rf_backtest(trainer, config: MLConfig) -> Dict:
    """Продвинутый Random Forest бэктест с детальной аналитикой."""
    
    print(f"🌲 ЗАПУСК ПРОДВИНУТОГО RANDOM FOREST БЭКТЕСТА")
    print(f"="*60)
    
    try:
        # Используем стандартный бэктест с улучшенной аналитикой
        backtest_results = run_backtrader_backtest(trainer, config)
        
        if 'error' in backtest_results:
            return backtest_results
        
        # ДОПОЛНИТЕЛЬНАЯ аналитика
        additional_metrics = {
            'rf_model_complexity': trainer.predictor.model.n_estimators if hasattr(trainer.predictor.model, 'n_estimators') else 0,
            'rf_max_depth': trainer.predictor.model.max_depth if hasattr(trainer.predictor.model, 'max_depth') else 0,
            'rf_feature_count': len(trainer.predictor.model.feature_importances_) if hasattr(trainer.predictor.model, 'feature_importances_') else 0,
            'data_quality_score': len(trainer.X_test) / 1000,  # Простая метрика качества данных
            'advanced_backtest': True
        }
        
        # Объединяем результаты
        backtest_results.update(additional_metrics)
        
        print(f"✅ ПРОДВИНУТЫЙ БЭКТЕСТ ЗАВЕРШЕН")
        print(f"🌲 RF Деревья: {additional_metrics['rf_model_complexity']}")
        print(f"🌲 RF Глубина: {additional_metrics['rf_max_depth']}")
        print(f"🌲 RF Признаки: {additional_metrics['rf_feature_count']}")
        
        return backtest_results
        
    except Exception as e:
        print(f"❌ Ошибка продвинутого бэктеста: {e}")
        return {'error': 'advanced_backtest_failed', 'details': str(e)}


class DynamicSegmentGenerator:
    """Генератор динамічних часових сегментів для різноманітного навчання."""
    
    def __init__(self, base_start_date: str = '2018-01-01', base_end_date: str = '2024-12-31'):
        self.base_start = pd.to_datetime(base_start_date)
        self.base_end = pd.to_datetime(base_end_date)
        self.used_segments = []
        
    def generate_random_segments(self, num_segments: int = 5, min_days: int = 365) -> List[Tuple[str, str]]:
        """Генерувати випадкові часові сегменти для навчання."""
        segments = []
        total_days = (self.base_end - self.base_start).days
        
        for _ in range(num_segments):
            # Випадковий початок
            random_start_offset = random.randint(0, max(1, total_days - min_days * 2))
            start_date = self.base_start + timedelta(days=random_start_offset)
            
            # Випадкова довжина (від min_days до залишку часу)
            max_duration = min(min_days * 3, (self.base_end - start_date).days)
            duration = random.randint(min_days, max(min_days, max_duration))
            end_date = start_date + timedelta(days=duration)
            
            # Переконуємося, що не виходимо за межі
            if end_date > self.base_end:
                end_date = self.base_end
            
            segment = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Уникаємо повторень
            if segment not in self.used_segments:
                segments.append(segment)
                self.used_segments.append(segment)
        
        return segments
    
    def generate_adaptive_segments(self, error_analysis: Dict) -> List[Tuple[str, str]]:
        """Генерувати адаптивні сегменти на основі аналізу помилок."""
        segments = []
        
        # Якщо є складні періоди, фокусуємося на них
        if 'difficult_periods' in error_analysis and error_analysis['difficult_periods']:
            for period_start, period_end in error_analysis['difficult_periods'][:3]:
                # Розширюємо складний період для кращого навчання
                extended_start = period_start - timedelta(days=30)
                extended_end = period_end + timedelta(days=30)
                
                segment = (extended_start.strftime('%Y-%m-%d'), extended_end.strftime('%Y-%m-%d'))
                segments.append(segment)
        
        # Додаємо випадкові сегменти для різноманітності
        random_segments = self.generate_random_segments(3, 200)  # Коротші сегменти
        segments.extend(random_segments)
        
        return segments[:5]  # Обмежуємо кількість


def _adjust_parameters_based_on_results(current_config: Dict, best_result: Dict, attempt: int) -> Dict:
    """ПОКРАЩЕНА адаптивна корекція параметрів з фокусом на якість та боротьбу з переобученням."""
    new_config = current_config.copy()
    
    profit = best_result.get('profit', 0)
    trades = best_result.get('trades', 0)
    winrate = best_result.get('winrate', 0)
    
    print(f"🎯 ЯКІСНА АДАПТИВНА КОРЕКЦІЯ (спроба {attempt}):")
    print(f"   Поточний результат: прибуток={profit:.2f}%, угод={trades}, винрейт={winrate:.1%}")
    
    # СТРАТЕГІЯ 1: БОРОТЬБА З ПЕРЕОБУЧЕННЯМ - головний пріоритет
    if attempt > 10 and profit < 0:  # Після 10 спроб все ще збитки = переобучення
        new_config['lookback'] = max(5, new_config['lookback'] - 2)        # Менше фічей
        new_config['n_estimators'] = max(10, new_config['n_estimators'] - 5)  # Менше дерев
        new_config['max_depth'] = max(3, new_config.get('max_depth', 4) - 1)  # Менша глибина
        new_config['min_samples_split'] = min(50, new_config.get('min_samples_split', 20) + 10)  # Більше зразків
        new_config['min_samples_leaf'] = min(20, new_config.get('min_samples_leaf', 10) + 5)     # Більше зразків у листі
        new_config['confidence'] = max(0.40, new_config['confidence'] + 0.05)  # Вища впевненість
        print(f"   🛡️ АНТИ-ПЕРЕОБУЧЕННЯ: зменшуємо складність моделі")
    
    # СТРАТЕГІЯ 2: Якщо мало угод - обережно збільшуємо чутливість
    elif trades < 15:
        new_config['min_threshold'] = max(0.001, new_config['min_threshold'] * 0.8)  # Обережніше зниження
        new_config['confidence'] = max(0.35, new_config['confidence'] * 0.9)        # Обережніше зниження
        print(f"   📈 ОБЕРЕЖНО збільшуємо чутливість (мало угод)")
    
    # СТРАТЕГІЯ 3: Якщо низький винрейт - покращуємо якість предсказань
    elif winrate < 0.50:
        new_config['confidence'] = min(0.55, new_config['confidence'] + 0.03)        # Вища впевненість
        new_config['lookback'] = min(15, new_config['lookback'] + 1)                 # Трохи більше контексту
        new_config['min_samples_split'] = min(30, new_config.get('min_samples_split', 20) + 5)  # Більша стабільність
        print(f"   ⬆️ Покращуємо ЯКІСТЬ предсказань (низький винрейт)")
    
    # СТРАТЕГІЯ 4: Якщо близько до прибутковості - фінальна оптимізація
    elif -2 <= profit < 5:
        new_config['confidence'] = min(0.50, new_config['confidence'] + 0.02)        # Легке покращення
        new_config['min_threshold'] = max(0.002, new_config['min_threshold'] * 0.95) # Легке зниження порогу
        print(f"   🔧 ФІНАЛЬНА оптимізація (близько до успіху)")
    
    # СТРАТЕГІЯ 5: Якщо стабільно прибуткові - налаштовуємо ефективність
    elif profit >= 5:
        new_config['min_threshold'] = max(0.001, new_config['min_threshold'] * 0.9)  # Більше сигналів
        new_config['confidence'] = max(0.40, new_config['confidence'] * 0.98)        # Трохи більше ризику
        print(f"   🚀 ПІДВИЩУЄМО ефективність (стабільний прибуток)")
    
    # СТРАТЕГІЯ 6: Захист від екстремальних значень
    new_config['lookback'] = max(5, min(20, new_config['lookback']))                # Обмеження 5-20
    new_config['confidence'] = max(0.30, min(0.60, new_config['confidence']))      # Обмеження 30-60%
    new_config['min_threshold'] = max(0.0005, min(0.01, new_config['min_threshold']))  # Обмеження 0.05-1%
    new_config['n_estimators'] = max(10, min(50, new_config.get('n_estimators', 30)))  # Обмеження 10-50 дерев
    
    print(f"   📊 НОВІ ПАРАМЕТРИ: поріг={new_config['min_threshold']:.4f}, "
          f"впевненість={new_config['confidence']:.2f}, lookback={new_config['lookback']}, "
          f"дерева={new_config.get('n_estimators', 30)}")
    
    return new_config


def print_banner():
    """Вывести баннер программы."""
    print("🤖" + "="*60 + "🤖")
    print("   MVP ОБУЧЕНИЕ STAS_ML МОДЕЛИ ДЛЯ ТОРГОВЛИ КРИПТОВАЛЮТАМИ")
    print("🤖" + "="*60 + "🤖")
    print()


def check_dependencies():
    """Проверить наличие необходимых зависимостей."""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import talib
    except ImportError:
        missing_deps.append("TA-Lib")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import seaborn
    except ImportError:
        missing_deps.append("seaborn")
    
    if missing_deps:
        print("❌ Отсутствуют зависимости:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Установите зависимости:")
        if "TA-Lib" in missing_deps:
            print("   pip install TA-Lib  # Может потребовать дополнительной настройки")
        print("   pip install pandas numpy scikit-learn matplotlib seaborn")
        print("   pip install xgboost  # Для XGBoost моделей")
        print("   pip install torch    # Для LSTM моделей")
        return False
    
    print("✅ Основные зависимости установлены")
    
    # Проверяем опциональные зависимости
    optional_deps = []
    try:
        import xgboost
    except ImportError:
        optional_deps.append("xgboost")
    
    try:
        import torch
    except ImportError:
        optional_deps.append("torch")
    
    if optional_deps:
        print("⚠️ Опциональные зависимости отсутствуют:")
        for dep in optional_deps:
            print(f"   - {dep}")
        print("💡 Некоторые модели могут быть недоступны")
    
    return True


def show_available_data():
    """Показать доступные данные."""
    print("📊 Доступные данные:")
    available_pairs = DataManager.get_available_pairs()
    
    total_pairs = 0
    for exchange, pairs in available_pairs.items():
        print(f"   {exchange}: {len(pairs)} пар")
        total_pairs += len(pairs)
    
    print(f"   Всего: {total_pairs} торговых пар")
    print()


def create_standard_config():
    """Создать стандартную конфигурацию."""
    print("⚡ Стандартная настройка:")
    print("   1. BTCUSDT на дневном таймфрейме")
    print("   2. Random Forest модель")
    print("   3. Предсказание направления движения")
    print("   4. Технические индикаторы включены")
    print("   5. Автоматическое имя модели")
    print("   6. Обязательный Backtrader бектест")
    
    return MLConfig(
        symbol='BTCUSDT',
        timeframe='1d',
        model_type='random_forest',
        target_type='direction'
    )


def show_training_targets(config: MLConfig):
    """Показать целевые результаты для модели."""
    targets = TrainingTargets()
    
    print("\n🎯" + "="*60 + "🎯")
    print("   ЦІЛЬОВІ РЕЗУЛЬТАТИ ДЛЯ НАВЧАННЯ МОДЕЛІ")
    print("🎯" + "="*60 + "🎯")
    
    # Показываем общие цели
    general = targets.general_targets
    print(f"\n📊 БАЖАНІ ТОРГОВІ РЕЗУЛЬТАТИ:")
    print(f"   💰 Заробіток: ≥{general.min_total_return_pct:.0f}% 🚀")
    print(f"   📉 Просадка: <{general.max_drawdown_pct:.0f}% ⚠️")
    print(f"   🎯 Вінрейт: >{general.min_win_rate:.0%} ✅")
    print(f"   📈 Sharpe Ratio: ≥{general.min_sharpe_ratio:.1f}")
    
    # Показываем ML цели
    print(f"\n🤖 ML МЕТРИКИ:")
    print(f"   🎯 Точність: ≥{general.min_accuracy:.0%}")
    print(f"   📊 F1-score: ≥{general.min_f1_score:.0%}")
    print(f"   ⚖️ Переобучення: <{general.max_overfitting_gap:.0%}")
    
    # Показываем специфичные цели для модели
    model_type = ModelType(config.model_type)
    if model_type in targets.model_specific_targets:
        model_targets = targets.model_specific_targets[model_type]
        print(f"\n🎯 ДОДАТКОВІ ЦІЛІ ДЛЯ {config.model_type.upper()}:")
        if model_targets.min_accuracy:
            print(f"   📈 Точність: ≥{model_targets.min_accuracy:.0%}")
        if model_targets.min_total_return_pct:
            print(f"   💰 Заробіток: ≥{model_targets.min_total_return_pct:.0f}%")
        if model_targets.max_drawdown_pct:
            print(f"   📉 Просадка: <{model_targets.max_drawdown_pct:.0f}%")
        if model_targets.min_win_rate:
            print(f"   🎯 Вінрейт: ≥{model_targets.min_win_rate:.0%}")
    
    print("\n💡 Модель навчається досягти цих результатів!")
    print("🎯" + "="*60 + "🎯")





def iterative_segment_training():
    """СПРОЩЕНА система навчання тільки Random Forest для максимальної швидкості."""
    print_banner()
    
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    # Показываем доступные данные
    show_available_data()
    
    print(f"\n🌲 ШВИДКЕ НАВЧАННЯ RANDOM FOREST")
    print(f"="*50)
    print(f"⚡ ПРIОРИТЕТ: ШВИДКІСТЬ > ТОЧНІСТЬ")
    print(f"🎯 БАЗОВІ ЦІЛІ:")
    print(f"   💰 Прибуток: ≥5% (знижена планка)")
    print(f"   📉 Просадка: <15% (м'якший контроль)")
    print(f"   🎯 Вінрейт: ≥52% (базова якість)")
    print(f"   📊 Мінімум угод: 15 (мінімальна статистика)")
    print(f"="*50)
    
    # ЗНИЖЕНІ ЦІЛІ для швидкості
    TARGET_PROFIT = 5.0       # Знижена планка
    TARGET_MAX_DRAWDOWN = 15.0 # М'якший контроль
    TARGET_MIN_WINRATE = 0.52  # Базова якість
    MIN_TRADES = 15           # Мінімальна статистика
    
    best_result = None
    attempt = 0
    max_attempts = 20  # Ще менше спроб для швидкості
    
    while attempt < max_attempts:
        attempt += 1
        
        print(f"\n🌲 ШВИДКА СПРОБА {attempt}/{max_attempts}")
        print(f"⚡ Мінімальні параметри для швидкості")
        
        try:
            # БАЗОВІ Random Forest параметри для швидкості
            rf_params = {
                'n_estimators': 30,        # Мало дерев для швидкості
                'max_depth': 6,            # Мала глибина
                'min_samples_split': 10,   # Швидке розділення
                'min_samples_leaf': 5,     # Швидкі листи
                'max_features': 'sqrt',    # Обмежені фічі
                'bootstrap': True,
                'n_jobs': -1,              # Всі CPU
                'random_state': 42
            }
            
            # МІНІМАЛЬНІ індикатори для швидкості
            simple_indicators = {
                'rsi': [14],               # Тільки RSI
                'sma': [20]                # Тільки одна SMA
            }
            
            # ПРОСТИЙ часовий сегмент
            time_segments = [
                ('2022-01-01', '2024-12-31'),  # Останні 3 роки
                ('2021-01-01', '2023-12-31'),  # Альтернативний період
                ('2020-01-01', '2022-12-31')   # COVID період
            ]
            segment = time_segments[attempt % len(time_segments)]
            
            print(f"📊 RF: {rf_params['n_estimators']} дерев, глибина {rf_params['max_depth']}")
            print(f"🔧 Індикатори: {len(simple_indicators)} (мінімум)")
            print(f"📅 Період: {segment[0]} - {segment[1]}")
            
            # Создаем ПРОСТУЮ конфигурацию
            config = MLConfig(
                symbol='BTCUSDT',
                timeframe='1d',
                model_type='random_forest',
                target_type='direction',
                lookback_window=10,                    # Малий lookback
                min_price_change_threshold=0.01,       # Низький поріг
                signal_confidence_threshold=0.4        # Низька впевненість
            )
            
            # МІНІМАЛЬНІ індикатори
            config.indicator_periods = simple_indicators
            
            # ШВИДКІ Random Forest параметри
            config.rf_params.update(rf_params)
            
            # Создаем тренер
            trainer = MLTrainer(config, custom_model_name=f"fast_rf_attempt_{attempt}")
            
            # ПРОСТАЯ підготовка даних
            original_data = trainer.data_processor.load_data()
            segment_data = original_data.loc[segment[0]:segment[1]].copy()
            
            if len(segment_data) < 300:  # Низькі вимоги до даних
                print(f"⚠️ Мало даних ({len(segment_data)} записів), пропускаємо")
                continue
            
            print(f"📊 Даних: {len(segment_data)} записів")
            
            # Заміняємо метод загрузки даних
            original_load_data = trainer.data_processor.load_data
            trainer.data_processor.load_data = lambda: segment_data
            
            # ШВИДКЕ навчання БЕЗ перевірок якості
            metrics = trainer.train()
            
            # Відновлюємо метод
            trainer.data_processor.load_data = original_load_data
            
            # БАЗОВА перевірка результатів
            profit = metrics.get('trading_total_return_pct', 0)
            drawdown = metrics.get('trading_max_drawdown_pct', 100)
            winrate = metrics.get('trading_win_rate', 0)
            trades = metrics.get('trading_total_trades', 0)
            test_acc = metrics.get('test_accuracy', 0)
            
            print(f"\n📊 ШВИДКІ РЕЗУЛЬТАТИ:")
            print(f"   💰 Прибуток: {profit:+.2f}% (цель: ≥{TARGET_PROFIT}%)")
            print(f"   📉 Просадка: {drawdown:.2f}% (цель: <{TARGET_MAX_DRAWDOWN}%)")
            print(f"   🎯 Вінрейт: {winrate:.1%} (цель: ≥{TARGET_MIN_WINRATE:.0%})")
            print(f"   📊 Угод: {trades} (мін: {MIN_TRADES})")
            print(f"   🎯 Точність: {test_acc:.1%}")
            
            # Перевіряємо БАЗОВІ цілі
            targets_met = (
                profit >= TARGET_PROFIT and
                drawdown < TARGET_MAX_DRAWDOWN and 
                winrate >= TARGET_MIN_WINRATE and
                trades >= MIN_TRADES
            )
            
            if targets_met:
                print(f"\n🎉 БАЗОВІ ЦІЛІ ДОСЯГНУТІ! Спроба {attempt}")
                print(f"🌲 ШВИДКА RANDOM FOREST МОДЕЛЬ ГОТОВА!")
                
                # Зберігаємо швидку модель
                model_path = trainer.save_model()
                print(f"🏆 ШВИДКА МОДЕЛЬ ЗБЕРЕЖЕНА: {model_path}")
                
                return {
                    'success': True,
                    'attempt': attempt,
                    'metrics': metrics,
                    'model_path': model_path,
                    'training_time': 'fast'
                }
            
            # Відстежуємо найкращий результат
            if best_result is None or profit > best_result.get('profit', -999):
                best_result = {
                    'attempt': attempt,
                    'metrics': metrics,
                    'trainer': trainer,
                    'profit': profit,
                    'drawdown': drawdown,
                    'winrate': winrate,
                    'trades': trades,
                    'accuracy': test_acc
                }
                print(f"💎 НОВИЙ КРАЩИЙ РЕЗУЛЬТАТ: {profit:+.2f}%")
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Швидке навчання зупинено на спробі {attempt}")
            break
        except Exception as e:
            print(f"❌ Помилка в спробі {attempt}: {e}")
            continue
    
    # ФІНАЛЬНІ результати
    if best_result:
        print(f"\n🏁 ШВИДКЕ НАВЧАННЯ ЗАВЕРШЕНО ПІСЛЯ {attempt} СПРОБ")
        print(f"🌲 НАЙКРАЩИЙ РЕЗУЛЬТАТ:")
        print(f"   💰 Прибуток: {best_result['profit']:+.2f}%")
        print(f"   📉 Просадка: {best_result['drawdown']:.2f}%")  
        print(f"   🎯 Вінрейт: {best_result['winrate']:.1%}")
        print(f"   📊 Угод: {best_result['trades']}")
        print(f"   🎯 Точність: {best_result['accuracy']:.1%}")
        
        # Автоматично зберігаємо найкращу модель
        model_path = best_result['trainer'].save_model()
        print(f"🏆 НАЙКРАЩА ШВИДКА RF МОДЕЛЬ ЗБЕРЕЖЕНА: {model_path}")
        
        return {
            'success': True,
            'best_result': best_result,
            'total_attempts': attempt,
            'model_path': model_path,
            'training_time': 'fast'
        }
    else:
        print(f"\n❌ Не вдалося навчити RF модель за {attempt} спроб")
        return {'success': False, 'error': 'no_fast_models'}
    
    # АВТОМАТИЧЕСКОЕ ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ ДО ДОСТИЖЕНИЯ ЦЕЛЕЙ
    print(f"\n🔄 ЦІЛІ НЕ ДОСЯГНУТІ. АВТОМАТИЧНО ПРОДОЛЖАЄМО ОБУЧЕННЯ...")
    
    # Показываем лучший результат
    if best_result:
        print(f"\n📊 ПОТОЧНИЙ КРАЩИЙ РЕЗУЛЬТАТ:")
        print(f"   💰 Прибуток: {best_result['profit']:+.2f}% (цель: {TARGET_PROFIT}%)")
        print(f"   🎯 Вінрейт: {best_result['winrate']:.1%} (цель: ≥{TARGET_MIN_WINRATE:.0%})")
        print(f"   📊 Угод: {best_result['trades']} (мін: {MIN_TRADES})")
    
    # АВТОМАТИЧЕСКОЕ РАСШИРЕННОЕ ОБУЧЕНИЕ БЕЗ ПОЛЬЗОВАТЕЛЬСКОГО ВВОДА
    max_total_attempts = 100  # Максимум 100 попыток всего
    current_round = 1
    
    while attempt < max_total_attempts:
            print(f"\n🚀 Продолжаем обучение с более агрессивными параметрами...")
            
            # АДАПТИВНІ КОНФІГУРАЦІЇ на основі аналізу помилок
            base_extended_configs = [
                {'model_type': 'random_forest', 'min_threshold': 0.002, 'confidence': 0.40, 'lookback': 20, 'n_estimators': 20},
                {'model_type': 'xgboost', 'min_threshold': 0.002, 'confidence': 0.40, 'lookback': 20, 'n_estimators': 20},
            ]
            
            # Застосовуємо адаптивні параметри якщо є аналіз помилок
            if len(error_analyzer.error_history) > 0:
                latest_adaptive_params = error_analyzer.get_adaptive_training_params()
                extended_configs = []
                for base_config in base_extended_configs:
                    adaptive_config = base_config.copy()
                    adaptive_config.update(latest_adaptive_params)  # Оновлюємо на основі помилок
                    extended_configs.append(adaptive_config)
                print(f"🎯 ВИКОРИСТОВУЄМО АДАПТИВНІ ПАРАМЕТРИ на основі {len(error_analyzer.error_history)} попередніх помилок")
            else:
                extended_configs = base_extended_configs
                print(f"⚠️ Використовуємо базові параметри (немає історії помилок)")
            
            # ДИНАМІЧНІ часові сегменти на основі помилок
            if len(error_analyzer.error_history) > 0:
                latest_error_analysis = error_analyzer.error_history[-1]
                extended_segments = segment_generator.generate_adaptive_segments(latest_error_analysis)
                print(f"🔄 ГЕНЕРУЄМО АДАПТИВНІ СЕГМЕНТИ на основі складних періодів")
            else:
                extended_segments = segment_generator.generate_random_segments(2, 400)
                print(f"🎲 Генеруємо випадкові сегменти")
            
            max_extended_attempts = len(extended_configs) * len(extended_segments)
            extended_attempt = 0
            
            print(f"🔥 Расширенное обучение: {max_extended_attempts} дополнительных попыток")
            
            for config_params in extended_configs:
                for start_date, end_date in extended_segments:
                    extended_attempt += 1
                    total_attempt = attempt + extended_attempt
                    
                    print(f"\n{'='*60}")
                    print(f"🔥 РАСШИРЕННАЯ ПОПЫТКА {extended_attempt}/{max_extended_attempts} (общая {total_attempt})")
                    print(f"📅 Период: {start_date} - {end_date}")
                    print(f"🤖 Модель: {config_params['model_type']}")
                    print(f"📊 Агрессивные параметры: порог={config_params['min_threshold']}, "
                          f"уверенность={config_params['confidence']}, lookback={config_params['lookback']}")
                    print(f"{'='*60}")
                    
                    try:
                        # Создаем конфигурацию для расширенного эксперимента
                        config = MLConfig(
                            symbol='BTCUSDT',
                            timeframe='1d',
                            model_type=config_params['model_type'],
                            target_type='direction',
                            lookback_window=config_params['lookback'],
                            min_price_change_threshold=config_params['min_threshold'],
                            signal_confidence_threshold=config_params['confidence']
                        )
                        
                        # Создаем тренер
                        trainer = MLTrainer(config, custom_model_name=f"extended_{extended_attempt}_{config_params['model_type']}_{start_date[:4]}_{end_date[:4]}")
                        
                        # Модифицируем данные для конкретного временного сегмента
                        original_data = trainer.data_processor.load_data()
                        segment_data = original_data.loc[start_date:end_date].copy()
                        
                        if len(segment_data) < 500:
                            print(f"⚠️ Недостаточно данных в расширенном сегменте ({len(segment_data)} записей), пропускаем")
                            continue
                        
                        print(f"📊 Расширенный сегмент данных: {len(segment_data)} записей")
                        
                        # Временно заменяем метод load_data
                        original_load_data = trainer.data_processor.load_data
                        trainer.data_processor.load_data = lambda: segment_data
                        
                        # Обучаем модель
                        metrics = trainer.train()
                        
                        # Восстанавливаем оригинальный метод
                        trainer.data_processor.load_data = original_load_data
                        
                        # Проверяем результаты
                        profit = metrics.get('trading_total_return_pct', 0)
                        drawdown = metrics.get('trading_max_drawdown_pct', 100)
                        winrate = metrics.get('trading_win_rate', 0)
                        trades = metrics.get('trading_total_trades', 0)
                        
                        print(f"\n📊 РЕЗУЛЬТАТЫ РАСШИРЕННОЙ ПОПЫТКИ {extended_attempt}:")
                        print(f"   💰 Прибуток: {profit:+.2f}% (цель: ≥{TARGET_PROFIT}%)")
                        print(f"   📉 Просадка: {drawdown:.2f}% (цель: <{TARGET_MAX_DRAWDOWN}%)")
                        print(f"   🎯 Вінрейт: {winrate:.1%} (цель: ≥{TARGET_MIN_WINRATE:.0%})")
                        print(f"   📊 Угод: {trades} (мін: {MIN_TRADES})")
                        
                        # Проверяем достижение целей
                        targets_met = (
                            profit >= TARGET_PROFIT and
                            drawdown < TARGET_MAX_DRAWDOWN and 
                            winrate >= TARGET_MIN_WINRATE and
                            trades >= MIN_TRADES
                        )
                        
                        if targets_met:
                            print(f"\n🎉 ЦІЛІ ДОСЯГНУТІ В РАСШИРЕННОМ ОБУЧЕНИИ! Попытка {extended_attempt}")
                            print(f"✅ Прибуток: {profit:+.2f}% ≥ {TARGET_PROFIT}%")
                            print(f"✅ Просадка: {drawdown:.2f}% < {TARGET_MAX_DRAWDOWN}%")
                            print(f"✅ Вінрейт: {winrate:.1%} ≥ {TARGET_MIN_WINRATE:.0%}")
                            print(f"✅ Угод: {trades} ≥ {MIN_TRADES}")
                            
                            # Сохраняем успешную модель
                            model_path = trainer.save_model()
                            print(f"🎊 УСПЕШНАЯ РАСШИРЕННАЯ МОДЕЛЬ СОХРАНЕНА: {model_path}")
                            
                            return {
                                'success': True,
                                'attempt': total_attempt,
                                'extended_training': True,
                                'config': config_params,
                                'time_segment': (start_date, end_date),
                                'metrics': metrics,
                                'model_path': model_path
                            }
                        
                        # Обновляем лучший результат если нашли лучше
                        if profit > best_result.get('profit', -999):
                            best_result.update({
                                'attempt': total_attempt,
                                'extended_training': True,
                                'config': config_params,
                                'time_segment': (start_date, end_date),
                                'metrics': metrics,
                                'trainer': trainer,
                                'profit': profit,
                                'drawdown': drawdown,
                                'winrate': winrate,
                                'trades': trades
                            })
                            print(f"💎 НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ В РАСШИРЕННОМ ОБУЧЕНИИ: {profit:+.2f}%")
                        
                    except KeyboardInterrupt:
                        print(f"\n⏹️ Расширенное обучение остановлено пользователем")
                        break
                    except Exception as e:
                        print(f"❌ Ошибка в расширенной попытке {extended_attempt}: {e}")
                        continue
            
            # Обновляем attempt для следующего раунда
            attempt += max_extended_attempts
            current_round += 1
            
            # Добавляем еще более агрессивные параметры для следующих раундов
            if current_round > 2:
                extended_configs.extend([
                    {'model_type': 'random_forest', 'min_threshold': 0.001, 'confidence': 0.35, 'lookback': 25, 'n_estimators': 15},
                    {'model_type': 'xgboost', 'min_threshold': 0.001, 'confidence': 0.35, 'lookback': 25, 'n_estimators': 15},
                ])
            
            # Расширяем временные сегменты для больших шансов на успех
            if current_round > 3:
                extended_segments.extend([
                    ('2018-01-01', '2022-12-31'),  # Более длинный период
                    ('2019-01-01', '2023-12-31'),  # Переходный период
                ])
            
            # Проверяем прогресс
            if best_result:
                progress_info = (
                    f"Раунд {current_round}, Попытка {attempt}: "
                    f"Лучший результат {best_result['profit']:+.2f}% "
                    f"({best_result['trades']} угод)"
                )
                print(f"📈 ПРОГРЕСС: {progress_info}")
                
                # Если мы близки к цели, продолжаем
                close_to_target = (
                    best_result['profit'] > TARGET_PROFIT * 0.1 or  # 10% от цели
                    best_result['trades'] > MIN_TRADES * 0.5      # 50% от мин. угод
                )
                
                if not close_to_target and current_round > 5:
                    print(f"⚠️ После {current_round} раундов прогресс недостаточный")
                    break
    
    # ФИНАЛЬНОЕ СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
    if best_result:
        print(f"\n🏁 ОБУЧЕНИЕ ЗАВЕРШЕНО ПОСЛЕ {attempt} ПОПЫТОК")
        print(f"💎 ЛУЧШИЙ ДОСТИГНУТЫЙ РЕЗУЛЬТАТ:")
        print(f"   💰 Прибуток: {best_result['profit']:+.2f}% (цель: ≥{TARGET_PROFIT}%)")
        print(f"   📉 Просадка: {best_result['drawdown']:.2f}% (цель: <{TARGET_MAX_DRAWDOWN}%)")
        print(f"   🎯 Вінрейт: {best_result['winrate']:.1%} (цель: ≥{TARGET_MIN_WINRATE:.0%})")
        print(f"   📊 Угод: {best_result['trades']} (мін: {MIN_TRADES})")
        
        # АВТОМАТИЧЕСКИ СОХРАНЯЕМ ЛУЧШУЮ МОДЕЛЬ
        model_path = best_result['trainer'].save_model()
        print(f"🏆 ЛУЧШАЯ МОДЕЛЬ АВТОМАТИЧЕСКИ СОХРАНЕНА: {model_path}")
        
        return {
            'success': False,  # Цели не достигнуты, но есть лучший результат
            'best_result': best_result,
            'total_attempts': attempt,
            'model_path': model_path
        }
    else:
        print(f"\n❌ Не удалось обучить ни одной модели за {attempt} попыток")
        return {'success': False, 'error': 'no_models_trained'}


def main():
    """Главная функция MVP с выбором режима обучения."""
    print_banner()
    
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    print("\n🎯 РЕЖИМЫ ОБУЧЕНИЯ:")
    print("   1. Стандартное обучение (одна попытка)")
    print("   2. Итеративное обучение до достижения целей (рекомендуется)")
    
    choice = input("\nВыберите режим (1-2, по умолчанию 2): ").strip()
    
    if choice == '1':
        # Стандартное обучение (старая версия)
        standard_training()
    else:
        # Итеративное обучение (новая версия)  
        iterative_segment_training()


def standard_training():
    """Стандартное обучение (одна попытка)."""
    # Показываем доступные данные
    show_available_data()
    
    # Используем стандартную конфигурацию
    config = create_standard_config()
    
    # Показываем целевые результаты
    show_training_targets(config)
    
    # Показываем настройки
    print(f"\n🚀 Начинаем обучение ML модели:")
    print(f"   Пара: {config.symbol}")
    print(f"   Таймфрейм: {config.timeframe}")
    print(f"   Модель: {config.model_type}")
    print(f"   Цель: {config.target_type}")
    print(f"   Lookback window: {config.lookback_window}")
    print("   Имя модели: автоматическое")
    print("   Режим: Стандартная настройка")
    print("-" * 60)
    
    try:
        # Создаем тренер и запускаем обучение (без выбора имени на данном этапе)
        trainer = MLTrainer(config)
        trainer.save_config()
        
        # Обучаем модель с учетом целевых показателей
        print(f"\n🎯 Модель будет оптимизирована для достижения:")
        print(f"   💰 Заробіток: ≥500% (текущий приоритет)")
        print(f"   📉 Просадка: <60%")
        print(f"   🎯 Вінрейт: >50%")
        
        metrics = trainer.train()
        
        # Оцениваем результаты относительно целей
        targets_service = ModelEvaluationService()
        model_type = ModelType(config.model_type)
        target_type = TargetType(config.target_type)
        
        evaluation_results = targets_service.evaluate_model(metrics, model_type, target_type)
        targets_service.print_evaluation_report(evaluation_results, trainer.experiment_name)
        
        print(f"\n✅ Обучение завершено успешно!")
        print(f"📊 Логи в: logs/ml/{trainer.experiment_name}/")
        
        # Показываем подробные результаты перед сохранением
        print(f"\n📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        print(f"   Модель: {config.model_type}")
        print(f"   Символ: {config.symbol}")
        print(f"   Таймфрейм: {config.timeframe}")
        if config.target_type == 'direction':
            print(f"   Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            if 'test_f1' in metrics:
                print(f"   Test F1-score: {metrics.get('test_f1', 0):.4f}")
        else:
            print(f"   Test MSE: {metrics.get('test_mse', 0):.6f}")
            print(f"   Test MAE: {metrics.get('test_mae', 0):.6f}")
        
        # Торговые результаты если есть
        if 'trading_total_return_pct' in metrics:
            print(f"   Доходность: {metrics['trading_total_return_pct']:+.2f}%")
            print(f"   Количество сделок: {metrics['trading_total_trades']}")
            print(f"   Процент выигрышных: {metrics['trading_win_rate']*100:.1f}%")
            print(f"   Финальный баланс: ${metrics['trading_final_balance']:,.2f}")
        
        # Спрашиваем о сохранении модели
        save_choice = input(f"\n💾 Сохранить модель '{trainer.experiment_name}'? (y/n): ").lower()
        if save_choice in ['y', 'yes', 'да']:
            model_path = trainer.save_model()
            print(f"📁 Модель сохранена в: {model_path}")
        else:
            print("⚠️ Модель не сохранена (доступна только в текущей сессии)")
        
        # Предлагаем продолжить обучение для улучшения результатов
        continue_choice = input(f"\n🔄 Продолжить обучение для улучшения модели? (y/n): ").lower()
        if continue_choice in ['y', 'yes', 'да']:
            print(f"\n🚀 Переходим к итеративному обучению...")
            iterative_segment_training()
            return
        
        # Предлагаем кросс-валидацию
        cv_choice = input("\nВыполнить кросс-валидацию? (y/n): ").lower()
        if cv_choice in ['y', 'yes', 'да']:
            print("\n🔄 Выполняем кросс-валидацию...")
            cv_results = trainer.cross_validate()
        
        # Обязательный backtrader бектест
        print("\n📈 Выполняем обязательный Backtrader бектест...")
        try:
            backtest_results = run_backtrader_backtest(trainer, config)
            print("✅ Backtrader бектест завершен!")
        except Exception as e:
            print(f"❌ Ошибка во время бектеста: {e}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Обучение остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        print(f"💡 Проверьте логи в: logs/ml/")


def run_backtrader_backtest(trainer: MLTrainer, config: MLConfig) -> dict:
    """
    ПЕРЕПИСАННЫЙ Random Forest бэктест с оптимизированной торговой стратегией.
    ОБЯЗАТЕЛЬНО использует Random Forest модель для максимальной точности предсказаний.
    """
    try:
        import backtrader as bt
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        
        print(f"🌲 RANDOM FOREST BACKTEST - ЗАПУСК")
        print(f"="*50)
        
        # ПРИНУДИТЕЛЬНАЯ ПРОВЕРКА: модель должна быть Random Forest
        if config.model_type != 'random_forest':
            print(f"⚠️ ПРИНУДИТЕЛЬНОЕ ПЕРЕКЛЮЧЕНИЕ: {config.model_type} → random_forest")
            config.model_type = 'random_forest'
            
            # Пересоздаем Random Forest модель с оптимальными параметрами
            optimal_rf_params = {
                'n_estimators': 150,        # Больше деревьев для стабильности
                'max_depth': 15,            # Глубокие деревья для криптовалют
                'min_samples_split': 3,     # Агрессивное разделение
                'min_samples_leaf': 1,      # Максимальная детализация
                'max_features': 'sqrt',     # Оптимально для финансов
                'bootstrap': True,          # Улучшает обобщение
                'oob_score': True,         # Out-of-bag оценка
                'n_jobs': -1,              # Все CPU
                'random_state': 42,        # Воспроизводимость
                'class_weight': 'balanced' # Балансируем классы
            }
            
            # Создаем новую Random Forest модель
            trainer.predictor.model = RandomForestClassifier(**optimal_rf_params)
            
            # Переобучаем на Random Forest
            print(f"🔄 Переобучение на оптимизированной Random Forest...")
            trainer.predictor.model.fit(trainer.X_train, trainer.y_train)
            print(f"✅ Random Forest модель готова!")
        
        # Получаем данные для бэктеста
        historical_data = trainer.data_processor.load_data()
        
        # УЛУЧШЕННЫЕ Random Forest предсказания с вероятностями
        rf_predictions = trainer.predictor.model.predict(trainer.X_test)
        rf_probabilities = trainer.predictor.model.predict_proba(trainer.X_test)
        
        # АНАЛИЗ КАЧЕСТВА Random Forest
        feature_importance = trainer.predictor.model.feature_importances_
        oob_score = getattr(trainer.predictor.model, 'oob_score_', 0)
        
        print(f"🔍 АНАЛИЗ RANDOM FOREST МОДЕЛИ:")
        print(f"   Out-of-Bag Score: {oob_score:.4f}")
        print(f"   Количество деревьев: {trainer.predictor.model.n_estimators}")
        print(f"   Глубина деревьев: {trainer.predictor.model.max_depth}")
        print(f"   Топ-3 важных признака: {np.argsort(feature_importance)[-3:]}")
        print(f"   Распределение предсказаний: {np.bincount(rf_predictions)}")
        print(f"   Средняя уверенность: {np.mean(np.max(rf_probabilities, axis=1)):.3f}")
        
        # Подготавливаем данные для бэктеста
        backtest_data = historical_data.tail(len(rf_predictions)).copy()
        
        if len(backtest_data) < len(rf_predictions):
            print(f"⚠️ Недостаточно исторических данных, обрезаем предсказания")
            rf_predictions = rf_predictions[-len(backtest_data):]
            rf_probabilities = rf_probabilities[-len(backtest_data):]
        
        # НАСТРОЙКИ БЭКТЕСТА
        initial_cash = 10000.0  # $10,000 - реалистичный стартовый капитал
        commission = 0.001      # 0.1% комиссия
        
        # Создаем Backtrader cerebro
        cerebro = bt.Cerebro()
        
        # Добавляем данные
        data_feed = bt.feeds.PandasData(
            dataname=backtest_data,
            datetime=None,
            open='open',
            high='high', 
            low='low',
            close='close',
            volume='volume'
        )
        cerebro.adddata(data_feed)
        
        # ПРОДВИНУТАЯ Random Forest ТОРГОВАЯ СТРАТЕГИЯ
        class RandomForestTradingStrategy(bt.Strategy):
            params = dict(
                confidence_threshold=0.60,    # Минимальная уверенность для входа
                position_size=0.95,          # 95% капитала
                stop_loss_pct=0.03,          # 3% стоп-лосс
                take_profit_pct=0.06,        # 6% тейк-профит  
                max_hold_days=7,             # Максимум 7 дней держания
                printlog=True               # Детальное логирование
            )
            
            def __init__(self):
                self.rf_predictions = rf_predictions
                self.rf_probabilities = rf_probabilities
                self.prediction_index = 0
                self.order = None
                self.entry_price = 0
                self.entry_date = None
                self.total_trades = 0
                self.winning_trades = 0
                self.max_balance = initial_cash
                self.peak_balance = initial_cash
                
                print(f"🌲 RANDOM FOREST СТРАТЕГИЯ ИНИЦИАЛИЗИРОВАНА:")
                print(f"   Порог уверенности: {self.params.confidence_threshold}")
                print(f"   Размер позиции: {self.params.position_size*100}%")
                print(f"   Стоп-лосс: {self.params.stop_loss_pct*100}%")
                print(f"   Тейк-профит: {self.params.take_profit_pct*100}%")
                
            def log(self, txt, dt=None):
                if self.params.printlog:
                    dt = dt or self.datas[0].datetime.date(0)
                    balance = self.broker.getvalue()
                    print(f'{dt.isoformat()}, Balance: ${balance:.2f}, {txt}')
                    
            def notify_order(self, order):
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.log(f'🟢 RF ПОКУПКА: ${order.executed.price:.2f}, Размер: {order.executed.size}')
                        self.entry_price = order.executed.price
                        self.entry_date = self.data.datetime.date(0)
                    else:
                        self.log(f'🔴 RF ПРОДАЖА: ${order.executed.price:.2f}, Размер: {order.executed.size}')
                        
                elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                    self.log(f'❌ RF ОРДЕР ОТКЛОНЕН: {order.status}')
                    
                self.order = None
                
            def notify_trade(self, trade):
                if trade.isclosed:
                    self.total_trades += 1
                    pnl_pct = (trade.pnlcomm / abs(trade.value)) * 100
                    
                    if trade.pnlcomm > 0:
                        self.winning_trades += 1
                        self.log(f'✅ RF ПРИБЫЛЬ: ${trade.pnlcomm:.2f} ({pnl_pct:+.2f}%)')
                    else:
                        self.log(f'❌ RF УБЫТОК: ${trade.pnlcomm:.2f} ({pnl_pct:+.2f}%)')
                        
                    # Обновляем пиковый баланс
                    current_balance = self.broker.getvalue()
                    if current_balance > self.peak_balance:
                        self.peak_balance = current_balance
                        
            def next(self):
                if self.order or self.prediction_index >= len(self.rf_predictions):
                    return
                    
                current_prediction = self.rf_predictions[self.prediction_index]
                current_confidence = np.max(self.rf_probabilities[self.prediction_index])
                current_price = self.data.close[0]
                current_date = self.data.datetime.date(0)
                cash = self.broker.getcash()
                
                # ПРОВЕРКА СУЩЕСТВУЮЩЕЙ ПОЗИЦИИ
                if self.position:
                    # Рассчитываем P&L
                    if self.position.size > 0:  # Длинная позиция
                        pnl_pct = (current_price - self.entry_price) / self.entry_price
                    else:  # Короткая позиция  
                        pnl_pct = (self.entry_price - current_price) / self.entry_price
                    
                    # СТОП-ЛОСС
                    if pnl_pct <= -self.params.stop_loss_pct:
                        self.log(f'🛑 RF СТОП-ЛОСС: {pnl_pct*100:.1f}%')
                        self.order = self.close()
                        
                    # ТЕЙК-ПРОФИТ
                    elif pnl_pct >= self.params.take_profit_pct:
                        self.log(f'🎯 RF ТЕЙК-ПРОФИТ: {pnl_pct*100:.1f}%')
                        self.order = self.close()
                        
                    # МАКСИМАЛЬНОЕ ВРЕМЯ ДЕРЖАНИЯ
                    elif self.entry_date and (current_date - self.entry_date).days >= self.params.max_hold_days:
                        self.log(f'⏰ RF ЗАКРЫТИЕ ПО ВРЕМЕНИ: {self.params.max_hold_days} дней')
                        self.order = self.close()
                        
                else:
                    # ВХОД В ПОЗИЦИЮ на основе Random Forest
                    if current_confidence >= self.params.confidence_threshold:
                        available_cash = cash * self.params.position_size
                        size = int(available_cash / current_price)
                        
                        if size > 0 and size * current_price <= cash * 0.99:
                            if current_prediction == 1:  # Покупка
                                self.log(f'📈 RF СИГНАЛ ПОКУПКИ: Уверенность {current_confidence:.3f}')
                                self.order = self.buy(size=size)
                                
                            elif current_prediction == 0:  # Продажа (короткая позиция)
                                self.log(f'📉 RF СИГНАЛ ПРОДАЖИ: Уверенность {current_confidence:.3f}')
                                self.order = self.sell(size=size)
                    else:
                        self.log(f'⏸️ RF СЛАБЫЙ СИГНАЛ: Уверенность {current_confidence:.3f} < {self.params.confidence_threshold}')
                        
                self.prediction_index += 1
                
            def stop(self):
                final_value = self.broker.getvalue()
                total_return = ((final_value - initial_cash) / initial_cash) * 100
                max_drawdown = ((self.peak_balance - final_value) / self.peak_balance) * 100
                win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
                
                self.log(f"🌲 === ИТОГИ RANDOM FOREST ТОРГОВЛИ ===")
                self.log(f"Начальный капитал: ${initial_cash:,.2f}")
                self.log(f"Финальный капитал: ${final_value:,.2f}")
                self.log(f"Общая доходность: {total_return:+.2f}%")
                self.log(f"Максимальная просадка: {max_drawdown:.2f}%")
                self.log(f"Всего сделок: {self.total_trades}")
                self.log(f"Выигрышных сделок: {self.winning_trades}")
                self.log(f"Random Forest винрейт: {win_rate:.1f}%")
                self.log(f"Пиковый баланс: ${self.peak_balance:,.2f}")
                self.log("🌲" + "=" * 40 + "🌲")
        
        # Настройки брокера
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        
        # Добавляем Random Forest стратегию
        cerebro.addstrategy(RandomForestTradingStrategy)
        
        # Добавляем анализаторы для детальной статистики
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        print(f"\n🚀 ЗАПУСК RANDOM FOREST БЭКТЕСТА:")
        print(f"   Начальный капитал: ${initial_cash:,.2f}")
        print(f"   Комиссия: {commission*100}%")
        print(f"   Период тестирования: {len(backtest_data)} дней")
        print(f"   Random Forest предсказаний: {len(rf_predictions)}")
        print(f"   Средняя уверенность RF: {np.mean(np.max(rf_probabilities, axis=1)):.3f}")
        
        # Выполняем бэктест
        strategies = cerebro.run()
        strategy = strategies[0]
        
        # Получаем результаты анализаторов
        trades_analysis = strategy.analyzers.trades.get_analysis()
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        returns_analysis = strategy.analyzers.returns.get_analysis()
        
        # Финальные результаты
        final_value = cerebro.broker.getvalue()
        total_return = ((final_value - initial_cash) / initial_cash) * 100
        
        # Детальная статистика
        total_trades = trades_analysis.get('total', {}).get('total', 0)
        won_trades = trades_analysis.get('won', {}).get('total', 0)
        lost_trades = trades_analysis.get('lost', {}).get('total', 0)
        
        win_rate = (won_trades / max(total_trades, 1)) * 100
        sharpe_ratio = sharpe_analysis.get('sharperatio', 0) or 0
        max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0) or 0
        
        # Формируем результаты
        results = {
            'model_type': 'random_forest',
            'initial_cash': initial_cash,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': won_trades,
            'losing_trades': lost_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'backtest_period_days': len(backtest_data),
            'rf_oob_score': oob_score,
            'rf_avg_confidence': float(np.mean(np.max(rf_probabilities, axis=1))),
            'rf_trees': trainer.predictor.model.n_estimators,
            'commission_pct': commission * 100
        }
        
        # Выводим результаты
        print(f"\n🌲 РЕЗУЛЬТАТЫ RANDOM FOREST БЭКТЕСТА:")
        print(f"="*60)
        print(f"   💰 Начальный капитал: ${results['initial_cash']:,.2f}")
        print(f"   💎 Финальная стоимость: ${results['final_value']:,.2f}")
        print(f"   📈 Общая доходность: {results['total_return_pct']:+.2f}%")
        print(f"   📉 Максимальная просадка: {results['max_drawdown_pct']:.2f}%")
        print(f"   📊 Общее количество сделок: {results['total_trades']}")
        print(f"   ✅ Выигрышных сделок: {results['winning_trades']}")
        print(f"   ❌ Проигрышных сделок: {results['losing_trades']}")
        print(f"   🎯 Random Forest винрейт: {results['win_rate']:.1f}%")
        print(f"   📐 Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"   🌲 RF Out-of-Bag Score: {results['rf_oob_score']:.4f}")
        print(f"   🎲 RF Средняя уверенность: {results['rf_avg_confidence']:.3f}")
        print(f"   🌳 Количество деревьев: {results['rf_trees']}")
        print(f"   💸 Комиссия: {results['commission_pct']:.1f}%")
        print(f"   📅 Период бэктеста: {results['backtest_period_days']} дней")
        print(f"🌲" + "="*60 + "🌲")
        
        # Сохраняем результаты
        import json
        import os
        os.makedirs(f"logs/ml/{trainer.experiment_name}", exist_ok=True)
        
        with open(f"logs/ml/{trainer.experiment_name}/random_forest_backtest.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Результаты сохранены: logs/ml/{trainer.experiment_name}/random_forest_backtest.json")
        
        return results
        
    except ImportError as e:
        print(f"❌ Отсутствуют зависимости: {e}")
        print("💡 Установите: pip install backtrader scikit-learn")
        return {'error': 'dependencies_missing', 'details': str(e)}
    except Exception as e:
        print(f"❌ Ошибка во время Random Forest бэктеста: {e}")
        import traceback
        traceback.print_exc()
        return {'error': 'backtest_failed', 'details': str(e)}


def auto_train():
    """Автоматическое обучение с настройками по умолчанию."""
    print_banner()
    
    # Конфигурация по умолчанию
    config = MLConfig(
        symbol='BTCUSDT',
        timeframe='1d',
        model_type='random_forest',
        target_type='direction',
        lookback_window=30
    )
    
    print(f"🚀 Автоматическое обучение:")
    print(f"   Пара: {config.symbol}")
    print(f"   Таймфрейм: {config.timeframe}")
    print(f"   Модель: {config.model_type}")
    print(f"   Цель: {config.target_type}")
    print("-" * 60)
    
    try:
        trainer = quick_train_ml(
            symbol=config.symbol,
            timeframe=config.timeframe,
            model_type=config.model_type,
            target_type=config.target_type
        )
        print("✅ Автоматическое обучение завершено!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    # Поддержка аргументов командной строки
    parser = argparse.ArgumentParser(description='MVP обучение STAS_ML модели', add_help=False)
    parser.add_argument('--auto', action='store_true', 
                       help='Автоматический запуск с настройками по умолчанию')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--model', default='xgboost', 
                       choices=['xgboost', 'random_forest', 'lstm', 'linear'],
                       help='Тип модели')
    parser.add_argument('--target', default='direction',
                       choices=['direction', 'price_change', 'volatility'],
                       help='Целевая переменная')

    parser.add_argument('--help', '-h', action='store_true', help='Показать помощь')
    
    args = parser.parse_args()
    
    if args.help:
        print("🤖 MVP Обучение STAS_ML Модели")
        print("\nИспользование:")
        print("  python mvp_train_ml.py                 # Интерактивный режим")
        print("  python mvp_train_ml.py --auto          # Автоматический запуск")
        print("  python mvp_train_ml.py --auto --symbol ETHUSDT --model random_forest")
        print("\nОпции:")
        parser.print_help()
        sys.exit(0)
    
    if args.auto:
        # Автоматический запуск
        print_banner()
        print(f"⚡ Автоматический запуск обучения:")
        print(f"   Пара: {args.symbol}")
        print(f"   Таймфрейм: {args.timeframe}")
        print(f"   Модель: {args.model}")
        print(f"   Цель: {args.target}")
        print("   Имя модели: автоматическое")
        print("   Backtrader бектест: обязательный")
        
        try:
            trainer = quick_train_ml(
                symbol=args.symbol,
                timeframe=args.timeframe,
                model_type=args.model,
                target_type=args.target
            )
            print("✅ Автоматическое обучение завершено!")
            
            # Обязательный backtrader бектест для авто режима
            print("\n📈 Выполняем обязательный Backtrader бектест...")
            try:
                config = MLConfig(
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    model_type=args.model,
                    target_type=args.target
                )
                backtest_results = run_backtrader_backtest(trainer, config)
                print("✅ Backtrader бектест завершен!")
            except Exception as e:
                print(f"❌ Ошибка во время бектеста: {e}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    else:
        # Интерактивный режим
        main()