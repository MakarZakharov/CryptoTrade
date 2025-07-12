#!/usr/bin/env python3
"""
Демонстрация улучшенной системы визуализации для автоматического селектора признаков.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Добавляем путь к модулям
sys.path.append(os.path.dirname(__file__))

from data.feature_selector import AutomaticFeatureSelector, create_auto_optimized_config


class MockConfig:
    """Мок-конфигурация для демонстрации."""
    def __init__(self, symbol='BTCUSDT'):
        self.symbol = symbol
        self.target_type = 'direction'  # или 'regression'
        self.indicator_periods = {}


def create_sample_crypto_data(n_samples=2000):
    """Создание реалистичных криптовалютных данных для демонстрации."""
    print("🔄 Создание демонстрационных данных...")
    
    # Параметры для реалистичных данных
    np.random.seed(42)
    
    # Временной индекс
    start_date = datetime.now() - timedelta(days=n_samples)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='1H')
    
    # Базовая цена и волатильность
    base_price = 45000  # Примерная цена BTC
    volatility = 0.02   # 2% часовая волатильность
    
    # Генерация цен с трендом и случайной компонентой
    price_changes = np.random.normal(0.0001, volatility, n_samples)  # Небольшой восходящий тренд
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Создание OHLC данных
    data = []
    for i in range(n_samples):
        # Open цена
        if i == 0:
            open_price = base_price
        else:
            open_price = data[i-1]['close']
        
        # Close цена
        close_price = prices[i]
        
        # High и Low с учетом волатильности
        intraday_range = abs(close_price - open_price) + np.random.exponential(close_price * 0.005)
        high_price = max(open_price, close_price) + np.random.uniform(0, intraday_range * 0.3)
        low_price = min(open_price, close_price) - np.random.uniform(0, intraday_range * 0.3)
        
        # Volume с некоторой корреляцией с волатильностью
        volume_base = 1000000  # Базовый объем
        volume_multiplier = 1 + abs(price_changes[i]) * 50  # Больше объема при больших движениях
        volume = int(volume_base * volume_multiplier * np.random.uniform(0.5, 2.0))
        
        data.append({
            'open': max(0.01, open_price),
            'high': max(0.01, high_price),
            'low': max(0.01, low_price),
            'close': max(0.01, close_price),
            'volume': volume
        })
    
    # Создание DataFrame
    df = pd.DataFrame(data, index=dates)
    
    print(f"✅ Создано {len(df)} записей демо-данных")
    print(f"📅 Период: {df.index[0]} - {df.index[-1]}")
    print(f"💰 Цена: от ${df['close'].min():.2f} до ${df['close'].max():.2f}")
    print(f"📈 Общий рост: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1)*100:.1f}%")
    
    return df


def demonstrate_enhanced_visualization():
    """Демонстрация улучшенной системы визуализации."""
    print("🚀 ДЕМОНСТРАЦИЯ УЛУЧШЕННОЙ СИСТЕМЫ ВИЗУАЛИЗАЦИИ")
    print("=" * 80)
    
    try:
        # 1. Создание данных
        sample_data = create_sample_crypto_data(1500)
        
        # 2. Создание конфигурации
        config = MockConfig('BTC/USDT')
        
        # 3. Создание селектора
        selector = AutomaticFeatureSelector(config)
        
        print("\n🔍 Запуск автоматического выбора признаков...")
        
        # 4. Выбор лучших индикаторов
        selected_indicators = selector.select_best_indicators(sample_data)
        
        if not selected_indicators:
            print("❌ Не удалось выбрать индикаторы")
            return
        
        print(f"\n✅ Выбрано {selected_indicators.get('n_features', 0)} индикаторов")
        
        # 5. Генерация всех индикаторов для полного анализа
        print("\n🔧 Генерация всех технических индикаторов...")
        full_data = selector._generate_all_indicators(sample_data)
        
        print(f"📊 Сгенерировано {len(full_data.columns)} всего признаков")
        
        # 6. Создание профессиональных визуализаций
        print("\n🎨 Создание профессиональных визуализаций...")
        
        # График важности признаков
        print("   📊 График важности признаков...")
        importance_fig = selector.visualize_feature_importance(selected_indicators)
        if importance_fig:
            print("   ✅ График важности создан")
        
        # Корреляционная матрица
        print("   🔥 Корреляционная матрица...")
        if 'selected_features' in selected_indicators:
            corr_fig = selector.visualize_correlation_matrix(
                full_data, 
                selected_indicators['selected_features']
            )
            if corr_fig:
                print("   ✅ Корреляционная матрица создана")
        
        # Распределение по категориям
        print("   🥧 График категорий...")
        cat_fig = selector.visualize_feature_categories(selected_indicators)
        if cat_fig:
            print("   ✅ График категорий создан")
        
        # 7. Создание комплексного отчета
        print("\n📑 Создание комплексного отчета...")
        reports = selector.create_comprehensive_report(
            selected_indicators, 
            full_data,
            "feature_selection_demo_report"
        )
        
        # 8. Отображение интерактивных графиков
        print("\n🌐 Запуск интерактивных графиков в браузере...")
        try:
            selector.show_interactive_plots(selected_indicators, full_data)
        except Exception as e:
            print(f"⚠️ Не удалось открыть браузер: {e}")
            print("💡 Графики сохранены в HTML файлы в директории feature_selection_demo_report/")
        
        # 9. Вывод итоговой статистики
        print("\n" + "=" * 80)
        print("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("=" * 80)
        
        print("📊 СТАТИСТИКА УЛУЧШЕНИЙ:")
        print(f"  ✅ Создано {len(reports)} интерактивных графиков")
        print(f"  ✅ Проанализировано {len(full_data.columns)} признаков")
        print(f"  ✅ Выбрано {selected_indicators.get('n_features', 0)} лучших признаков")
        print(f"  ✅ Метод селекции: {selected_indicators.get('selection_method', 'unknown')}")
        
        if 'feature_importance' in selected_indicators:
            top_feature = max(selected_indicators['feature_importance'].items(), key=lambda x: x[1])
            print(f"  🏆 Лучший признак: {top_feature[0]} (важность: {top_feature[1]:.4f})")
        
        print("\n🎯 НОВЫЕ ВОЗМОЖНОСТИ ВИЗУАЛИЗАЦИИ:")
        print("  ✅ Интерактивные графики важности признаков")
        print("  ✅ Корреляционные матрицы с hover-эффектами") 
        print("  ✅ Круговые диаграммы распределения по категориям")
        print("  ✅ Комплексный dashboard с множественными панелями")
        print("  ✅ Автоматическое сохранение в HTML формате")
        print("  ✅ Профессиональное оформление в стиле Plotly")
        
        print("\n💡 Файлы отчетов сохранены в:")
        print("   📁 feature_selection_demo_report/")
        print("     📄 feature_importance.html")
        print("     📄 correlation_matrix.html") 
        print("     📄 feature_categories.html")
        print("     📄 dashboard.html")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Тест базовой функциональности без визуализации."""
    print("\n🧪 ТЕСТ БАЗОВОЙ ФУНКЦИОНАЛЬНОСТИ")
    print("-" * 50)
    
    try:
        # Создание простых данных
        data = create_sample_crypto_data(500)
        config = MockConfig()
        
        # Тест создания селектора
        selector = AutomaticFeatureSelector(config)
        print("✅ Селектор создан успешно")
        
        # Тест генерации индикаторов
        enhanced_data = selector._generate_all_indicators(data)
        print(f"✅ Сгенерировано {len(enhanced_data.columns)} индикаторов")
        
        # Тест выбора признаков
        results = selector.select_best_indicators(data)
        print(f"✅ Выбрано {results.get('n_features', 0)} признаков")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте: {e}")
        return False


def main():
    """Главная функция демонстрации."""
    print("🎨 ДЕМОНСТРАЦИЯ УЛУЧШЕННЫХ ВОЗМОЖНОСТЕЙ ВИЗУАЛИЗАЦИИ")
    print("💫 Feature Selector с профессиональными интерактивными графиками")
    print("=" * 80)
    
    # Сначала базовый тест
    if not test_basic_functionality():
        print("❌ Базовая функциональность не работает")
        return 1
    
    # Затем полная демонстрация
    if not demonstrate_enhanced_visualization():
        print("❌ Демонстрация визуализации не удалась")
        return 1
    
    print("\n🚀 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("🎉 Система визуalizации улучшена и готова к использованию!")
    
    return 0


if __name__ == "__main__":
    exit(main())