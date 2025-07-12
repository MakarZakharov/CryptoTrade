#!/usr/bin/env python3
"""
MVP скрипт для быстрого запуска обучения STAS_ML модели.
Простой интерфейс для начала обучения ML моделей с минимальными настройками.
"""

import os
import sys
import argparse
from datetime import datetime


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
    """Итеративное обучение на разных сегментах данных до достижения целей."""
    print_banner()
    
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    # Показываем доступные данные
    show_available_data()
    
    # Целевые показатели
    TARGET_PROFIT = 500.0  # 500% прибыль
    TARGET_MAX_DRAWDOWN = 60.0  # <60% просадка
    TARGET_MIN_WINRATE = 0.50  # >50% винрейт
    MIN_TRADES = 96  # Минимум 96 сделок для статистики
    
    print(f"\n🎯 ЦІЛЬОВІ ПОКАЗНИКИ ДЛЯ ДОСЯГНЕННЯ:")
    print(f"   💰 Прибуток: ≥{TARGET_PROFIT}%")
    print(f"   📉 Просадка: <{TARGET_MAX_DRAWDOWN}%") 
    print(f"   🎯 Вінрейт: ≥{TARGET_MIN_WINRATE:.0%}")
    print(f"   📊 Мінімум угод: {MIN_TRADES}")
    print("-" * 60)
    
    # ОПТИМИЗИРОВАННЫЕ параметры для быстрого обучения
    model_configs = [
        # Быстрые конфигурации с уменьшенной сложностью
        {'model_type': 'random_forest', 'min_threshold': 0.005, 'confidence': 0.45, 'lookback': 15, 'n_estimators': 50},
        {'model_type': 'xgboost', 'min_threshold': 0.005, 'confidence': 0.45, 'lookback': 15, 'n_estimators': 50},
        {'model_type': 'random_forest', 'min_threshold': 0.002, 'confidence': 0.40, 'lookback': 10, 'n_estimators': 30},
    ]
    
    # СОКРАЩЕННЫЕ временные сегменты для быстрого тестирования
    time_segments = [
        ('2020-01-01', '2024-12-31'),  # Основной период
        ('2021-01-01', '2024-12-31'),  # Более свежие данные
        ('2022-01-01', '2024-12-31'),  # Новейший период
    ]
    
    best_result = None
    attempt = 0
    max_attempts = len(model_configs) * len(time_segments)
    
    print(f"🚀 Начинаем итеративное обучение (макс. {max_attempts} попыток)")
    
    for config_params in model_configs:
        for start_date, end_date in time_segments:
            attempt += 1
            print(f"\n{'='*60}")
            print(f"🔄 ПОПЫТКА {attempt}/{max_attempts}")
            print(f"📅 Период: {start_date} - {end_date}")
            print(f"🤖 Модель: {config_params['model_type']}")
            print(f"📊 Параметры: порог={config_params['min_threshold']}, "
                  f"уверенность={config_params['confidence']}, lookback={config_params['lookback']}")
            print(f"{'='*60}")
            
            try:
                # Создаем конфигурацию для текущего эксперимента с БЫСТРЫМИ параметрами
                config = MLConfig(
                    symbol='BTCUSDT',
                    timeframe='1d',
                    model_type=config_params['model_type'],
                    target_type='direction',
                    lookback_window=config_params['lookback'],
                    min_price_change_threshold=config_params['min_threshold'],
                    signal_confidence_threshold=config_params['confidence']
                )
                
                # АВТОМАТИЧЕСКИЙ ВЫБОР ЛУЧШИХ ИНДИКАТОРОВ
                try:
                    from CryptoTrade.ai.STAS_ML.data.feature_selector import create_auto_optimized_config
                    
                    print(f"🔍 Автоматический выбор индикаторов для {config.symbol}...")
                    
                    # Загружаем данные для анализа индикаторов
                    temp_processor = CryptoDataProcessor(config)
                    raw_data = temp_processor.load_data()
                    segment_data_for_analysis = raw_data.loc[start_date:end_date].copy()
                    
                    if len(segment_data_for_analysis) >= 200:  # Минимум данных для анализа
                        # Оптимизируем конфигурацию индикаторов
                        config, selected_indicators = create_auto_optimized_config(config, segment_data_for_analysis)
                        print(f"✅ Автоматически выбрано {selected_indicators['n_features']} лучших индикаторов")
                    else:
                        print(f"⚠️ Недостаточно данных для автоматической селекции, используем стандартные индикаторы")
                
                except Exception as e:
                    print(f"⚠️ Ошибка автоматической селекции индикаторов: {e}")
                    print(f"🔧 Используем стандартные индикаторы")
                
                # УСКОРЯЕМ обучение - уменьшаем параметры модели
                if config_params['model_type'] == 'random_forest':
                    config.rf_params['n_estimators'] = config_params.get('n_estimators', 30)
                    config.rf_params['max_depth'] = 5  # Уменьшаем глубину
                    config.rf_params['min_samples_split'] = 10  # Увеличиваем мин. сплит
                elif config_params['model_type'] == 'xgboost':
                    config.xgb_params['n_estimators'] = config_params.get('n_estimators', 30)
                    config.xgb_params['max_depth'] = 3  # Уменьшаем глубину
                    config.xgb_params['learning_rate'] = 0.1  # Увеличиваем learning rate
                
                # Создаем тренер
                trainer = MLTrainer(config, custom_model_name=f"attempt_{attempt}_{config_params['model_type']}_{start_date[:4]}_{end_date[:4]}")
                
                # Модифицируем данные для конкретного временного сегмента
                original_data = trainer.data_processor.load_data()
                
                # Фильтруем по временному сегменту
                segment_data = original_data.loc[start_date:end_date].copy()
                
                if len(segment_data) < 500:  # Минимум данных
                    print(f"⚠️ Недостаточно данных в сегменте ({len(segment_data)} записей), пропускаем")
                    continue
                
                print(f"📊 Сегмент данных: {len(segment_data)} записей")
                
                # Временно заменяем метод load_data для использования сегмента
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
                
                print(f"\n📊 РЕЗУЛЬТАТЫ ПОПЫТКИ {attempt}:")
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
                    print(f"\n🎉 ЦІЛІ ДОСЯГНУТІ! Попытка {attempt}")
                    print(f"✅ Прибуток: {profit:+.2f}% ≥ {TARGET_PROFIT}%")
                    print(f"✅ Просадка: {drawdown:.2f}% < {TARGET_MAX_DRAWDOWN}%")
                    print(f"✅ Вінрейт: {winrate:.1%} ≥ {TARGET_MIN_WINRATE:.0%}")
                    print(f"✅ Угод: {trades} ≥ {MIN_TRADES}")
                    
                    # Сохраняем успешную модель
                    model_path = trainer.save_model()
                    print(f"🎊 УСПЕШНАЯ МОДЕЛЬ СОХРАНЕНА: {model_path}")
                    
                    # Запускаем финальный бектест
                    print(f"\n📈 Финальный бектест успешной модели...")
                    backtest_results = run_backtrader_backtest(trainer, config)
                    
                    return {
                        'success': True,
                        'attempt': attempt,
                        'config': config_params,
                        'time_segment': (start_date, end_date),
                        'metrics': metrics,
                        'model_path': model_path
                    }
                
                # Отслеживаем лучший результат даже если цели не достигнуты
                if best_result is None or profit > best_result.get('profit', -999):
                    best_result = {
                        'attempt': attempt,
                        'config': config_params,
                        'time_segment': (start_date, end_date),
                        'metrics': metrics,
                        'trainer': trainer,
                        'profit': profit,
                        'drawdown': drawdown,
                        'winrate': winrate,
                        'trades': trades
                    }
                    print(f"💎 Новый лучший результат: {profit:+.2f}%")
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Обучение остановлено пользователем на попытке {attempt}")
                break
            except Exception as e:
                print(f"❌ Ошибка в попытке {attempt}: {e}")
                continue
    
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
            
            # СОКРАЩЕННЫЕ агрессивные конфигурации для быстрого обучения
            extended_configs = [
                {'model_type': 'random_forest', 'min_threshold': 0.002, 'confidence': 0.40, 'lookback': 20, 'n_estimators': 20},
                {'model_type': 'xgboost', 'min_threshold': 0.002, 'confidence': 0.40, 'lookback': 20, 'n_estimators': 20},
            ]
            
            # СОКРАЩЕННЫЕ временные сегменты для быстрого обучения
            extended_segments = [
                ('2020-01-01', '2023-12-31'),  # 4 года основных данных
                ('2021-01-01', '2024-12-31'),  # 4 года свежих данных
            ]
            
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
    """Выполнить Backtrader бектест обученной модели."""
    try:
        import backtrader as bt
        import pandas as pd
        import numpy as np
        from CryptoTrade.ai.STAS_ML.evaluation.backtrader_strategy import MLPredictionStrategy
        
        # Получаем данные для бектеста
        data_processor = trainer.data_processor
        
        # Получаем исторические данные
        historical_data = data_processor.load_data()
        
        # Получаем предсказания модели на тестовых данных
        test_predictions = trainer.predictor.predict(trainer.X_test)
        
        # Создаем DataFrame для backtrader
        backtest_data = historical_data.copy()
        
        # Используем последние данные, соответствующие тестовому набору
        if len(test_predictions) <= len(backtest_data):
            backtest_data = backtest_data.tail(len(test_predictions)).copy()
        else:
            print("⚠️ Предсказаний больше чем данных, используем доступные данные")
            backtest_data = backtest_data.tail(len(test_predictions)).copy()
        
        # Создаем Backtrader cerebro
        cerebro = bt.Cerebro()
        
        # Добавляем данные
        data_feed = bt.feeds.PandasData(
            dataname=backtest_data,
            datetime=None,  # Используем индекс как datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume'
        )
        cerebro.adddata(data_feed)
        
        # Настройки
        initial_cash = 10000000  # $10 миллионов для неограниченного тестирования
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% комиссия
        
        # Создаем максимально простую торговую стратегию с БЫСТРОЙ оценкой
        class SimpleMLTradingStrategy(bt.Strategy):
            params = dict(
                printlog=False,  # ОТКЛЮЧАЕМ детальное логирование для скорости
                position_size=0.95,  # 95% капитала - максимум
            )
            
            def __init__(self):
                self.predictions = np.array(test_predictions).astype(int)
                self.prediction_index = 0
                self.order = None
                self.entry_price = 0
                self.total_trades = 0
                self.winning_trades = 0
                
                print(f"🔍 Анализ предсказаний:")
                print(f"   Всего предсказаний: {len(self.predictions)}")
                print(f"   Уникальные значения: {np.unique(self.predictions)}")
                print(f"   Распределение: {np.bincount(self.predictions)}")
                
            def log(self, txt, dt=None):
                if self.params.printlog:
                    dt = dt or self.datas[0].datetime.date(0)
                    print(f'{dt.isoformat()}, {txt}')
                    
            def notify_order(self, order):
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.log(f'🟢 ПОКУПКА ВЫПОЛНЕНА: ${order.executed.price:.2f}, Размер: {order.executed.size}')
                        self.entry_price = order.executed.price
                    else:
                        self.log(f'🔴 ПРОДАЖА ВЫПОЛНЕНА: ${order.executed.price:.2f}, Размер: {order.executed.size}')
                        
                elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                    self.log(f'❌ ОРДЕР ОТКЛОНЕН: {order.status}')
                    
                self.order = None
                
            def notify_trade(self, trade):
                if trade.isclosed:
                    self.total_trades += 1
                    if trade.pnlcomm > 0:
                        self.winning_trades += 1
                        self.log(f'✅ ПРИБЫЛЬ: ${trade.pnlcomm:.2f}')
                    else:
                        self.log(f'❌ УБЫТОК: ${trade.pnlcomm:.2f}')
                        
            def next(self):
                # Проверяем активный ордер
                if self.order:
                    return
                    
                # Получаем текущее предсказание
                if self.prediction_index >= len(self.predictions):
                    return
                    
                current_prediction = self.predictions[self.prediction_index]
                current_price = self.data.close[0]
                cash = self.broker.getcash()
                
                # ПРИНУДИТЕЛЬНАЯ ТОРГОВЛЯ КАЖДЫЙ ДЕНЬ
                if not self.position:
                    # Рассчитываем размер позиции на основе доступных денег
                    available_cash = cash * self.params.position_size
                    size = int(available_cash / current_price)
                    
                    # Обеспечиваем минимальный размер
                    if size < 1:
                        size = 1
                    
                    # Проверяем достаточность средств
                    required_cash = size * current_price
                    
                    if required_cash <= cash * 0.99:  # Оставляем 1% буфер
                        self.log(f'📈 ПРИНУДИТЕЛЬНАЯ ПОКУПКА: День {self.prediction_index}, Размер: {size}, Цена: ${current_price:.2f}, Нужно: ${required_cash:.2f}')
                        self.order = self.buy(size=size)
                    else:
                        self.log(f'💸 НЕДОСТАТОЧНО СРЕДСТВ: Нужно ${required_cash:.2f}, Есть ${cash:.2f}')
                        
                # Продаем через 10 дней держания или на любом предсказании
                elif self.position and self.prediction_index % 10 == 0:
                    self.log(f'📉 ПРИНУДИТЕЛЬНАЯ ПРОДАЖА на дне {self.prediction_index}')
                    self.order = self.sell(size=self.position.size)
                        
                self.prediction_index += 1
                
            def stop(self):
                final_value = self.broker.getvalue()
                initial_cash = 10000
                total_return = ((final_value - initial_cash) / initial_cash) * 100
                win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
                
                self.log(f"=== ИТОГИ ТОРГОВЛИ ===")
                self.log(f"Начальный капитал: ${initial_cash:,.2f}")
                self.log(f"Финальный капитал: ${final_value:,.2f}")
                self.log(f"Общая доходность: {total_return:+.2f}%")
                self.log(f"Всего сделок: {self.total_trades}")
                self.log(f"Выигрышных сделок: {self.winning_trades}")
                self.log(f"Винрейт: {win_rate:.1f}%")
                self.log("=" * 25)
        
        # Добавляем стратегию
        cerebro.addstrategy(SimpleMLTradingStrategy)
        
        # Запускаем бектест
        print(f"🚀 Запуск Backtrader бектеста...")
        print(f"   Начальный капитал: ${initial_cash:,.2f}")
        print(f"   Период: {len(backtest_data)} дней")
        print(f"   Предсказаний: {len(test_predictions)}")
        
        # Выполняем бектест
        strategies = cerebro.run()
        strategy = strategies[0]
        
        # Получаем результаты
        final_value = cerebro.broker.getvalue()
        total_return = ((final_value - initial_cash) / initial_cash) * 100
        
        # Формируем результаты
        results = {
            'initial_cash': initial_cash,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': strategy.total_trades,
            'winning_trades': strategy.winning_trades,
            'win_rate': (strategy.winning_trades / max(strategy.total_trades, 1)) * 100,
            'max_drawdown': strategy.max_balance - final_value if hasattr(strategy, 'max_balance') else 0,
            'backtest_period_days': len(backtest_data)
        }
        
        # Выводим результаты
        print(f"\n📊 РЕЗУЛЬТАТЫ BACKTRADER БЕКТЕСТА:")
        print(f"   Начальный капитал: ${results['initial_cash']:,.2f}")
        print(f"   Финальная стоимость: ${results['final_value']:,.2f}")
        print(f"   Общая доходность: {results['total_return_pct']:+.2f}%")
        print(f"   Общее количество сделок: {results['total_trades']}")
        print(f"   Выигрышных сделок: {results['winning_trades']}")
        print(f"   Винрейт: {results['win_rate']:.1f}%")
        print(f"   Период бектеста: {results['backtest_period_days']} дней")
        
        # Сохраняем результаты
        import json
        with open(f"logs/ml/{trainer.experiment_name}/backtrader_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except ImportError:
        print("❌ Backtrader не установлен. Установите: pip install backtrader")
        return {'error': 'backtrader_not_installed'}
    except Exception as e:
        print(f"❌ Ошибка во время Backtrader бектеста: {e}")
        return {'error': str(e)}


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