"""Валидатор данных для DRL системы."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils import DRLLogger


class DataValidator:
    """Валидатор качества финансовых данных."""
    
    def __init__(self, logger: Optional[DRLLogger] = None):
        """Инициализация валидатора."""
        self.logger = logger or DRLLogger("data_validator")
    
    def validate_ohlcv(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Валидация OHLCV данных.
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            Словарь с результатами валидации
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Проверка обязательных колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results['is_valid'] = False
            results['errors'].append(f"Отсутствуют колонки: {missing_columns}")
            return results
        
        # Проверка логики цен OHLC
        invalid_ohlc = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ]
        
        if len(invalid_ohlc) > 0:
            results['warnings'].append(f"Найдено {len(invalid_ohlc)} записей с некорректными OHLC данными")
        
        # Проверка на отрицательные цены и объемы
        negative_prices = df[(df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)]
        if len(negative_prices) > 0:
            results['is_valid'] = False
            results['errors'].append(f"Найдено {len(negative_prices)} записей с отрицательными/нулевыми ценами")
        
        negative_volume = df[df['volume'] < 0]
        if len(negative_volume) > 0:
            results['warnings'].append(f"Найдено {len(negative_volume)} записей с отрицательным объемом")
        
        # Проверка на экстремальные ценовые движения
        price_changes = df['close'].pct_change().dropna()
        extreme_moves = price_changes[np.abs(price_changes) > 0.5]  # >50% за период
        if len(extreme_moves) > 0:
            results['warnings'].append(f"Найдено {len(extreme_moves)} экстремальных ценовых движений (>50%)")
        
        # Статистики
        results['stats'] = {
            'total_records': len(df),
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'volume_stats': {
                'mean': float(df['volume'].mean()),
                'median': float(df['volume'].median()),
                'zero_volume_count': int((df['volume'] == 0).sum())
            },
            'missing_data': df.isnull().sum().to_dict(),
            'date_range': {
                'start': str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else 'N/A',
                'end': str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else 'N/A'
            }
        }
        
        return results
    
    def validate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, any]:
        """Валидация технических индикаторов."""
        results = {
            'is_valid': True,
            'warnings': [],
            'indicator_stats': {}
        }
        
        # Проверка RSI в диапазоне 0-100
        rsi_columns = [col for col in df.columns if 'rsi' in col.lower()]
        for col in rsi_columns:
            invalid_rsi = df[(df[col] < 0) | (df[col] > 100)].dropna()
            if len(invalid_rsi) > 0:
                results['warnings'].append(f"RSI {col} имеет значения вне диапазона 0-100: {len(invalid_rsi)} записей")
        
        # Проверка Bollinger Bands
        bb_upper_cols = [col for col in df.columns if 'bb_upper' in col.lower()]
        bb_lower_cols = [col for col in df.columns if 'bb_lower' in col.lower()]
        
        for upper_col, lower_col in zip(bb_upper_cols, bb_lower_cols):
            if upper_col.replace('upper', 'lower') == lower_col:
                invalid_bb = df[df[upper_col] <= df[lower_col]].dropna()
                if len(invalid_bb) > 0:
                    results['warnings'].append(f"Bollinger Bands {upper_col}/{lower_col}: верхняя полоса ниже нижней в {len(invalid_bb)} записях")
        
        # Статистики по индикаторам
        technical_columns = [col for col in df.columns 
                           if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'sma', 'ema', 'atr', 'bb_'])]
        
        for col in technical_columns:
            if df[col].dtype in ['float32', 'float64']:
                results['indicator_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'null_count': int(df[col].isnull().sum())
                }
        
        return results
    
    def generate_report(self, validation_results: Dict) -> str:
        """Генерация отчета о валидации."""
        report = []
        report.append("=== ОТЧЕТ О ВАЛИДАЦИИ ДАННЫХ ===\n")
        
        if validation_results['is_valid']:
            report.append("✅ Данные прошли базовую валидацию")
        else:
            report.append("❌ Данные не прошли валидацию")
            for error in validation_results['errors']:
                report.append(f"  ОШИБКА: {error}")
        
        if validation_results['warnings']:
            report.append("\n⚠️ Предупреждения:")
            for warning in validation_results['warnings']:
                report.append(f"  {warning}")
        
        if 'stats' in validation_results:
            stats = validation_results['stats']
            report.append(f"\n📊 Статистики:")
            report.append(f"  Записей: {stats['total_records']}")
            report.append(f"  Диапазон цен: {stats['price_range']['min']:.2f} - {stats['price_range']['max']:.2f}")
            report.append(f"  Средний объем: {stats['volume_stats']['mean']:.2f}")
            report.append(f"  Период данных: {stats['date_range']['start']} - {stats['date_range']['end']}")
        
        return "\n".join(report)
    
    def quick_validate(self, df: pd.DataFrame) -> bool:
        """Быстрая валидация для основных проверок."""
        if df.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Проверка на отрицательные цены
        if (df[required_columns] <= 0).any().any():
            return False
        
        # Проверка логики OHLC
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).any()
        
        if invalid_ohlc:
            return False
        
        return True