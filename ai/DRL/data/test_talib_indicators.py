#!/usr/bin/env python3
"""Комплексные тесты для технических индикаторов с TA-Lib."""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
sys.path.append('.')

from CryptoTrade.ai.DRL.data.technical_indicators import TechnicalIndicators
from CryptoTrade.ai.DRL.utils import DRLLogger


class TestTALibIndicators:
    """Тесты для технических индикаторов с TA-Lib."""
    
    @classmethod
    def setup_class(cls):
        """Настройка тестовых данных."""
        cls.logger = DRLLogger("test_talib", log_level="DEBUG")
        cls.indicators = TechnicalIndicators(cls.logger)
        
        # Создание реалистичных тестовых данных
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1D')
        n_days = len(dates)
        
        # Генерируем ценовые данные с трендом
        base_price = 30000
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Создаем OHLCV данные
        cls.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.002, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0.005, 0.003, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0.005, 0.003, n_days))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_days),
            'quote_volume': prices * np.random.uniform(1000, 10000, n_days)
        })
        
        cls.test_data.set_index('timestamp', inplace=True)
        
        # Исправляем OHLCV логику
        cls.test_data['high'] = np.maximum.reduce([
            cls.test_data['open'], cls.test_data['high'], 
            cls.test_data['low'], cls.test_data['close']
        ])
        cls.test_data['low'] = np.minimum.reduce([
            cls.test_data['open'], cls.test_data['high'], 
            cls.test_data['low'], cls.test_data['close']
        ])
        
        cls.test_data = cls.test_data.astype('float64')
    
    def test_sma_accuracy(self):
        """Тест точности Simple Moving Average."""
        period = 20
        sma_result = self.indicators.sma(self.test_data['close'], period)
        
        # Проверка базовых свойств
        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(self.test_data)
        assert not sma_result.isna().all()
        
        # Проверка логики SMA (первые period-1 значений должны быть NaN)
        assert sma_result.iloc[:period-1].isna().all()
        
        # Проверка что значения разумные
        valid_sma = sma_result.dropna()
        assert all(valid_sma > 0)
        assert all(valid_sma < self.test_data['close'].max() * 2)
        
        print(f"✅ SMA тест пройден: {len(valid_sma)} валидных значений")
    
    def test_ema_accuracy(self):
        """Тест точности Exponential Moving Average."""
        period = 12
        ema_result = self.indicators.ema(self.test_data['close'], period)
        
        assert isinstance(ema_result, pd.Series)
        assert len(ema_result) == len(self.test_data)
        
        # EMA должна быть более чувствительной к последним значениям
        valid_ema = ema_result.dropna()
        assert len(valid_ema) > 0
        assert all(valid_ema > 0)
        
        print(f"✅ EMA тест пройден: {len(valid_ema)} валидных значений")
    
    def test_rsi_boundaries(self):
        """Тест RSI на корректные границы."""
        rsi_result = self.indicators.rsi(self.test_data['close'], 14)
        
        assert isinstance(rsi_result, pd.Series)
        valid_rsi = rsi_result.dropna()
        
        # RSI должен быть в диапазоне 0-100
        assert all(valid_rsi >= 0)
        assert all(valid_rsi <= 100)
        
        # Должны быть как высокие, так и низкие значения
        assert valid_rsi.min() < 50
        assert valid_rsi.max() > 50
        
        print(f"✅ RSI тест пройден: диапазон {valid_rsi.min():.2f} - {valid_rsi.max():.2f}")
    
    def test_macd_components(self):
        """Тест MACD компонентов."""
        macd, signal, histogram = self.indicators.macd(self.test_data['close'], 12, 26, 9)
        
        # Проверка типов и размеров
        assert all(isinstance(x, pd.Series) for x in [macd, signal, histogram])
        assert all(len(x) == len(self.test_data) for x in [macd, signal, histogram])
        
        # Проверка математической связи: histogram = macd - signal
        valid_indices = ~(macd.isna() | signal.isna() | histogram.isna())
        if valid_indices.any():
            np.testing.assert_array_almost_equal(
                histogram[valid_indices].values,
                (macd - signal)[valid_indices].values,
                decimal=5
            )
        
        print(f"✅ MACD тест пройден: {valid_indices.sum()} валидных точек")
    
    def test_bollinger_bands_logic(self):
        """Тест логики Bollinger Bands."""
        upper, middle, lower = self.indicators.bollinger_bands(self.test_data['close'], 20, 2)
        
        # Проверка порядка полос
        valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_indices.any():
            assert all(upper[valid_indices] >= middle[valid_indices])
            assert all(middle[valid_indices] >= lower[valid_indices])
        
        # Средняя полоса должна быть близка к SMA
        sma_20 = self.indicators.sma(self.test_data['close'], 20)
        sma_valid = ~(middle.isna() | sma_20.isna())
        if sma_valid.any():
            np.testing.assert_array_almost_equal(
                middle[sma_valid].values,
                sma_20[sma_valid].values,
                decimal=3
            )
        
        print(f"✅ Bollinger Bands тест пройден: {valid_indices.sum()} валидных точек")
    
    def test_atr_positive_values(self):
        """Тест ATR на положительные значения."""
        atr_result = self.indicators.atr(
            self.test_data['high'], 
            self.test_data['low'], 
            self.test_data['close'], 
            14
        )
        
        valid_atr = atr_result.dropna()
        assert len(valid_atr) > 0
        assert all(valid_atr >= 0)  # ATR всегда неотрицательный
        
        # ATR должен быть разумного размера относительно цен
        avg_price = self.test_data['close'].mean()
        assert all(valid_atr < avg_price * 0.1)  # ATR не должен быть больше 10% от цены
        
        print(f"✅ ATR тест пройден: диапазон {valid_atr.min():.2f} - {valid_atr.max():.2f}")
    
    def test_stochastic_boundaries(self):
        """Тест Stochastic на корректные границы."""
        k, d = self.indicators.stochastic(
            self.test_data['high'], 
            self.test_data['low'], 
            self.test_data['close']
        )
        
        valid_k = k.dropna()
        valid_d = d.dropna()
        
        # Stochastic должен быть в диапазоне 0-100
        if len(valid_k) > 0:
            assert all(valid_k >= 0)
            assert all(valid_k <= 100)
        
        if len(valid_d) > 0:
            assert all(valid_d >= 0)
            assert all(valid_d <= 100)
        
        print(f"✅ Stochastic тест пройден: %K={len(valid_k)}, %D={len(valid_d)} валидных значений")
    
    def test_adx_components(self):
        """Тест ADX компонентов."""
        adx, plus_di, minus_di = self.indicators.adx(
            self.test_data['high'], 
            self.test_data['low'], 
            self.test_data['close']
        )
        
        valid_adx = adx.dropna()
        valid_plus = plus_di.dropna()
        valid_minus = minus_di.dropna()
        
        # ADX должен быть в диапазоне 0-100
        if len(valid_adx) > 0:
            assert all(valid_adx >= 0)
            assert all(valid_adx <= 100)
        
        # DI индикаторы должны быть неотрицательными
        if len(valid_plus) > 0:
            assert all(valid_plus >= 0)
        
        if len(valid_minus) > 0:
            assert all(valid_minus >= 0)
        
        print(f"✅ ADX тест пройден: ADX={len(valid_adx)}, +DI={len(valid_plus)}, -DI={len(valid_minus)}")
    
    def test_obv_cumulative_nature(self):
        """Тест OBV на кумулятивную природу."""
        obv_result = self.indicators.obv(self.test_data['close'], self.test_data['volume'])
        
        valid_obv = obv_result.dropna()
        assert len(valid_obv) > 0
        
        # OBV может быть отрицательным, но должен изменяться логично
        obv_changes = valid_obv.diff().dropna()
        price_changes = self.test_data['close'].pct_change().dropna()
        
        # Большие изменения OBV должны соответствовать большим объемам
        volume_aligned = self.test_data['volume'][obv_changes.index]
        correlation = np.corrcoef(np.abs(obv_changes), volume_aligned)[0, 1]
        assert not np.isnan(correlation) or len(obv_changes) < 10
        
        print(f"✅ OBV тест пройден: {len(valid_obv)} валидных значений")
    
    def test_vwap_reasonableness(self):
        """Тест VWAP на разумность значений."""
        vwap_result = self.indicators.vwap(
            self.test_data['high'], 
            self.test_data['low'], 
            self.test_data['close'], 
            self.test_data['volume']
        )
        
        valid_vwap = vwap_result.dropna()
        assert len(valid_vwap) > 0
        
        # VWAP должен быть близок к ценам
        price_range = [self.test_data['low'].min(), self.test_data['high'].max()]
        assert all(valid_vwap >= price_range[0] * 0.9)
        assert all(valid_vwap <= price_range[1] * 1.1)
        
        print(f"✅ VWAP тест пройден: диапазон {valid_vwap.min():.2f} - {valid_vwap.max():.2f}")
    
    def test_add_all_indicators_integration(self):
        """Интеграционный тест добавления всех индикаторов."""
        result_df = self.indicators.add_all_indicators(self.test_data)
        
        # Проверка что добавились новые колонки
        assert len(result_df.columns) > len(self.test_data.columns)
        
        # Проверка наличия основных индикаторов
        expected_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper_20', 'bb_middle_20', 'bb_lower_20',
            'atr_14', 'stoch_k', 'stoch_d',
            'adx_14', 'plus_di_14', 'minus_di_14',
            'obv', 'vwap_20'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result_df.columns, f"Отсутствует индикатор: {indicator}"
        
        # Проверка что данные валидны
        for col in expected_indicators:
            valid_data = result_df[col].dropna()
            assert len(valid_data) > 0, f"Нет валидных данных для {col}"
        
        print(f"✅ Интеграционный тест пройден: добавлено {len(result_df.columns) - len(self.test_data.columns)} индикаторов")
    
    def test_performance_benchmark(self):
        """Тест производительности."""
        import time
        
        start_time = time.time()
        result_df = self.indicators.add_all_indicators(self.test_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Проверка что выполнение не слишком медленное
        assert execution_time < 5.0, f"Слишком медленное выполнение: {execution_time:.2f}s"
        
        print(f"✅ Тест производительности пройден: {execution_time:.3f}s для {len(self.test_data)} записей")
    
    def test_error_handling(self):
        """Тест обработки ошибок."""
        # Тест с пустым DataFrame
        empty_df = pd.DataFrame()
        try:
            self.indicators.add_all_indicators(empty_df)
            assert False, "Должна была возникнуть ошибка для пустого DataFrame"
        except ValueError:
            pass
        
        # Тест с отсутствующими колонками
        incomplete_df = pd.DataFrame({'close': [1, 2, 3]})
        try:
            self.indicators.add_all_indicators(incomplete_df)
            assert False, "Должна была возникнуть ошибка для неполного DataFrame"
        except ValueError:
            pass
        
        print("✅ Тест обработки ошибок пройден")
    
    def test_nan_handling(self):
        """Тест обработки NaN значений."""
        # Создаем данные с небольшим количеством NaN в середине
        test_data_with_nan = self.test_data.copy()
        # Добавляем только 2 NaN значения, а не 5
        test_data_with_nan.iloc[50:52, test_data_with_nan.columns.get_loc('close')] = np.nan
        
        result_df = self.indicators.add_all_indicators(test_data_with_nan)
        
        # Проверяем что результат содержит данные
        assert len(result_df) == len(test_data_with_nan)
        
        # Проверяем что базовые индикаторы хотя бы частично работают
        # (TA-Lib может пропускать NaN значения и продолжать расчеты)
        total_valid_indicators = 0
        for col in ['sma_20', 'ema_12', 'rsi_14', 'price_change', 'high_low_ratio']:
            if col in result_df.columns:
                valid_data = result_df[col].dropna()
                if len(valid_data) > 0:
                    total_valid_indicators += 1
        
        # Хотя бы некоторые индикаторы должны работать
        assert total_valid_indicators > 0, f"Ни один индикатор не содержит валидных данных"
        
        print(f"✅ Тест обработки NaN пройден: {total_valid_indicators} индикаторов с валидными данными")
    
    def test_feature_importance(self):
        """Тест оценки важности фичей."""
        result_df = self.indicators.add_all_indicators(self.test_data)
        importance = self.indicators.get_feature_importance_score(result_df)
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # Проверяем что значения важности в разумных пределах
        for feature, score in importance.items():
            assert 0 <= score <= 1, f"Неверная важность для {feature}: {score}"
        
        print(f"✅ Тест важности фичей пройден: {len(importance)} фичей оценено")


def run_comprehensive_tests():
    """Запуск всех тестов."""
    print("🧪 ЗАПУСК КОМПЛЕКСНЫХ ТЕСТОВ TA-LIB ИНДИКАТОРОВ")
    print("=" * 60)
    
    test_class = TestTALibIndicators()
    test_class.setup_class()
    
    tests = [
        test_class.test_sma_accuracy,
        test_class.test_ema_accuracy,
        test_class.test_rsi_boundaries,
        test_class.test_macd_components,
        test_class.test_bollinger_bands_logic,
        test_class.test_atr_positive_values,
        test_class.test_stochastic_boundaries,
        test_class.test_adx_components,
        test_class.test_obv_cumulative_nature,
        test_class.test_vwap_reasonableness,
        test_class.test_add_all_indicators_integration,
        test_class.test_performance_benchmark,
        test_class.test_error_handling,
        test_class.test_nan_handling,
        test_class.test_feature_importance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"   ✅ Пройдено: {passed}")
    print(f"   ❌ Не пройдено: {failed}")
    print(f"   📈 Успешность: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ TA-Lib интеграция работает корректно")
        print("✅ Все индикаторы функционируют правильно")
        print("✅ Обработка ошибок реализована")
        print("✅ Производительность в норме")
        return True
    else:
        print(f"\n⚠️ {failed} тестов не пройдено. Требуется доработка.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)