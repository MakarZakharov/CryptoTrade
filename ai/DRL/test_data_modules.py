#!/usr/bin/env python3
"""Тестовый скрипт для проверки модулей обработки данных."""

import sys
import os
sys.path.append('.')

def test_data_modules():
    """Тестирование всех модулей обработки данных."""
    
    print("🧪 Тестирование модулей обработки данных DRL системы")
    print("=" * 60)
    
    try:
        # Тест импортов
        print("1. Тестирование импортов...")
        from CryptoTrade.ai.DRL import CSVDataLoader, TechnicalIndicators, DataPreprocessor
        from CryptoTrade.ai.DRL.config import TradingConfig
        from CryptoTrade.ai.DRL.utils import DRLLogger
        print("✅ Все импорты успешны")
        
        # Тест конфигурации
        print("\n2. Создание конфигурации...")
        config = TradingConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1d",
            lookback_window=20,
            include_technical_indicators=True
        )
        logger = DRLLogger("data_test")
        print("✅ Конфигурация создана")
        
        # Тест загрузчика данных
        print("\n3. Тестирование загрузчика данных...")
        loader = CSVDataLoader(logger=logger)
        
        # Получаем информацию о доступных данных
        symbols = loader.get_available_symbols("binance")
        print(f"✅ Найдено {len(symbols)} символов: {symbols[:5]}...")
        
        # Получаем информацию о данных без полной загрузки
        if "BTCUSDT" in symbols:
            data_info = loader.get_data_info("BTCUSDT", "binance", "1d")
            print(f"✅ Информация о BTCUSDT: {data_info.get('approx_rows', 'N/A')} строк")
            
            # Загружаем небольшой образец данных
            df = loader.load_data("BTCUSDT", "binance", "1d", 
                                start_date="2023-01-01", end_date="2023-02-01")
            print(f"✅ Загружено {len(df)} строк данных BTCUSDT")
            print(f"   Колонки: {df.columns.tolist()}")
            print(f"   Период: {df.index[0]} - {df.index[-1]}")
        else:
            print("⚠️ BTCUSDT не найден, пропускаем загрузку данных")
            return
        
        # Тест технических индикаторов
        print("\n4. Тестирование технических индикаторов...")
        indicators = TechnicalIndicators(logger=logger)
        
        # Добавляем индикаторы к данным
        df_with_indicators = indicators.add_all_indicators(df, {
            'sma': [20],
            'ema': [12, 26],
            'rsi': [14],
            'macd': [12, 26, 9],
            'bollinger': [20],
            'atr': [14]
        })
        
        print(f"✅ Добавлено {len(df_with_indicators.columns) - len(df.columns)} индикаторов")
        print(f"   Новые колонки: {[col for col in df_with_indicators.columns if col not in df.columns]}")
        
        # Тест оценки важности фичей
        importance = indicators.get_feature_importance_score(df_with_indicators)
        if importance:
            top_features = list(importance.items())[:5]
            print(f"✅ Топ-5 важных фичей: {[(feat, f'{score:.3f}') for feat, score in top_features]}")
        
        # Тест предобработчика
        print("\n5. Тестирование предобработчика...")
        preprocessor = DataPreprocessor(config, logger=logger)
        
        # Создание дополнительных фичей
        df_with_features = preprocessor.create_features(df_with_indicators)
        print(f"✅ Создано {len(df_with_features.columns) - len(df_with_indicators.columns)} дополнительных фичей")
        
        # Полная подготовка данных
        df_prepared = preprocessor.prepare_for_drl(df_with_features.copy())
        print(f"✅ Данные подготовлены: {df_prepared.shape}")
        
        # Разделение данных
        train_df, val_df, test_df = preprocessor.split_data_for_training(df_prepared)
        print(f"✅ Данные разделены: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Создание последовательностей
        sequences, targets = preprocessor.create_sequences(train_df, sequence_length=10)
        print(f"✅ Создано {len(sequences)} последовательностей размером {sequences.shape}")
        
        # Сводка по фичам
        summary = preprocessor.get_feature_summary(df_prepared)
        print(f"\n📊 Сводка по данным:")
        print(f"   Всего фичей: {summary['total_features']}")
        print(f"   Всего образцов: {summary['total_samples']}")
        print(f"   Качество данных: {summary['data_quality']['completeness']:.1f}%")
        print(f"   Типы фичей: {summary['feature_types']}")
        
        # Отчет о качестве данных
        quality_report = loader.get_data_quality_report(df)
        print(f"\n📈 Отчет о качестве:")
        print(f"   Пропущенные значения: {quality_report['missing_values']}")
        print(f"   Дубликаты: {quality_report['duplicates']}")
        print(f"   Период данных: {quality_report['date_range']['days']} дней")
        if 'price_anomalies' in quality_report:
            print(f"   Ценовые аномалии: {quality_report['price_anomalies']}")
        
        print(f"\n🎉 Все тесты пройдены успешно!")
        print(f"✅ Модуль обработки данных готов к использованию")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_modules()
    if not success:
        sys.exit(1)
    
    print(f"\n🚀 Готово к переходу к Этапу 3: Торговые среды")
    print(f"   Скажите 'давай с 3' для продолжения")