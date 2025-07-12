"""
Визуализация данных на которых обучается DRL система торговли криптовалютой.
Показывает весь пайплайн от сырых данных до обработанных признаков.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

# Добавляем путь к модулям DRL
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_processing.data_collector import CryptoDataCollector, DataConfig
from data_processing.feature_engineering import FeatureEngineer, DataNormalizer
from environment.trading_env import TradingEnvironment, TradingConfig

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.facecolor'] = 'white'

class DRLDataVisualizer:
    """Класс для визуализации данных DRL системы."""
    
    def __init__(self):
        self.setup_matplotlib()
        
    def setup_matplotlib(self):
        """Настройка matplotlib для корректного отображения."""
        plt.rcParams['axes.unicode_minus'] = False
        
    def visualize_raw_data(self, data: pd.DataFrame, symbol: str):
        """Визуализация сырых OHLCV данных."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Сырые данные DRL: {symbol}', fontsize=16, fontweight='bold')
        
        # График цены (свечной график упрощенный)
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['close'], label='Close Price', color='#1f77b4', linewidth=1)
        ax1.fill_between(data.index, data['low'], data['high'], alpha=0.2, color='gray', label='High-Low Range')
        ax1.set_title('Цена закрытия и диапазон High-Low')
        ax1.set_ylabel('Цена (USDT)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # График объема
        ax2 = axes[0, 1]
        ax2.plot(data.index, data['volume'], color='orange', alpha=0.7, linewidth=1)
        ax2.set_title('Объем торгов')
        ax2.set_ylabel('Объем')
        ax2.tick_params(axis='x', rotation=45)
        
        # Распределение дневных изменений
        ax3 = axes[1, 0]
        daily_returns = data['close'].pct_change().dropna()
        ax3.hist(daily_returns * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title('Распределение изменений цены (%)')
        ax3.set_xlabel('Изменение цены (%)')
        ax3.set_ylabel('Частота')
        ax3.axvline(daily_returns.mean() * 100, color='red', linestyle='--', 
                   label=f'Среднее: {daily_returns.mean()*100:.2f}%')
        ax3.legend()
        
        # Статистика данных
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""СТАТИСТИКА СЫРЫХ ДАННЫХ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Период: {data.index[0].strftime('%Y-%m-%d %H:%M')} - {data.index[-1].strftime('%Y-%m-%d %H:%M')}
Записей: {len(data):,}
Цена мин/макс: ${data['close'].min():.2f} / ${data['close'].max():.2f}
Средняя цена: ${data['close'].mean():.2f}
Волатильность: {daily_returns.std()*100:.2f}%
Общий рост: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1)*100:.1f}%
Объем среднесуточный: {data['volume'].mean():,.0f}
        """
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        return fig

    def visualize_technical_indicators(self, data_with_indicators: pd.DataFrame, symbol: str):
        """Визуализация технических индикаторов DRL."""
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle(f'Технические индикаторы DRL: {symbol}', fontsize=16, fontweight='bold')
        
        # График 1: Цена + Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(data_with_indicators.index, data_with_indicators['close'], 
                label='Close', color='black', linewidth=1.5)
        
        # SMA и EMA индикаторы
        sma_cols = [col for col in data_with_indicators.columns if col.startswith('sma_')]
        ema_cols = [col for col in data_with_indicators.columns if col.startswith('ema_')]
        
        colors_sma = ['red', 'blue', 'green']
        colors_ema = ['orange', 'purple']
        
        for i, col in enumerate(sma_cols[:3]):
            if col in data_with_indicators.columns:
                period = col.split('_')[1]
                ax1.plot(data_with_indicators.index, data_with_indicators[col], 
                        label=f'SMA {period}', color=colors_sma[i], alpha=0.8, linewidth=1)
        
        for i, col in enumerate(ema_cols[:2]):
            if col in data_with_indicators.columns:
                period = col.split('_')[1]
                ax1.plot(data_with_indicators.index, data_with_indicators[col], 
                        label=f'EMA {period}', color=colors_ema[i], alpha=0.8, 
                        linewidth=1, linestyle='--')
        
        ax1.set_title('Цена и скользящие средние')
        ax1.set_ylabel('Цена (USDT)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # График 2: RSI
        ax2 = axes[0, 1]
        rsi_cols = [col for col in data_with_indicators.columns if col.startswith('rsi_')]
        for col in rsi_cols[:2]:
            if col in data_with_indicators.columns:
                period = col.split('_')[1]
                ax2.plot(data_with_indicators.index, data_with_indicators[col], 
                        label=f'RSI {period}', linewidth=1.5)
        
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Перекупленность (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Перепроданность (30)')
        ax2.set_title('RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # График 3: MACD
        ax3 = axes[1, 0]
        if 'macd' in data_with_indicators.columns:
            ax3.plot(data_with_indicators.index, data_with_indicators['macd'], 
                    label='MACD', color='blue', linewidth=1.5)
        if 'macd_signal' in data_with_indicators.columns:
            ax3.plot(data_with_indicators.index, data_with_indicators['macd_signal'], 
                    label='Signal', color='red', linewidth=1.5)
        if 'macd_histogram' in data_with_indicators.columns:
            ax3.bar(data_with_indicators.index, data_with_indicators['macd_histogram'], 
                   label='Histogram', alpha=0.6, color='gray', width=0.8)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # График 4: Bollinger Bands
        ax4 = axes[1, 1]
        ax4.plot(data_with_indicators.index, data_with_indicators['close'], 
                label='Close', color='black', linewidth=1.5)
        
        if 'bb_upper' in data_with_indicators.columns:
            ax4.plot(data_with_indicators.index, data_with_indicators['bb_upper'], 
                    color='red', alpha=0.7, label='BB Upper')
            ax4.plot(data_with_indicators.index, data_with_indicators['bb_middle'], 
                    color='blue', alpha=0.7, label='BB Middle')
            ax4.plot(data_with_indicators.index, data_with_indicators['bb_lower'], 
                    color='green', alpha=0.7, label='BB Lower')
            
            # Заливка между лентами
            ax4.fill_between(data_with_indicators.index, 
                           data_with_indicators['bb_upper'], 
                           data_with_indicators['bb_lower'], 
                           alpha=0.1, color='blue')
        
        ax4.set_title('Bollinger Bands')
        ax4.set_ylabel('Цена (USDT)')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        # График 5: ATR (волатильность)
        ax5 = axes[2, 0]
        if 'atr' in data_with_indicators.columns:
            ax5.plot(data_with_indicators.index, data_with_indicators['atr'], 
                    label='ATR', color='purple', linewidth=1.5)
        
        ax5.set_title('ATR (Average True Range)')
        ax5.set_ylabel('ATR')
        ax5.legend()
        ax5.tick_params(axis='x', rotation=45)
        
        # График 6: Объем и соотношения
        ax6 = axes[2, 1]
        
        # Нормализованный объем
        if 'volume' in data_with_indicators.columns:
            volume_norm = data_with_indicators['volume'] / data_with_indicators['volume'].max()
            ax6.bar(data_with_indicators.index[::10], volume_norm.iloc[::10], 
                   alpha=0.6, color='orange', label='Volume (norm)', width=10)
        
        # Volume ratio если есть
        if 'volume_ratio' in data_with_indicators.columns:
            ax6_twin = ax6.twinx()
            ax6_twin.plot(data_with_indicators.index, data_with_indicators['volume_ratio'], 
                         color='red', linewidth=1.5, label='Volume Ratio')
            ax6_twin.set_ylabel('Volume Ratio')
            ax6_twin.legend(loc='upper right')
        
        ax6.set_title('Объем и Volume Ratio')
        ax6.set_ylabel('Объем (нормализованный)')
        ax6.legend(loc='upper left')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

    def visualize_feature_engineering(self, original_data: pd.DataFrame, enhanced_data: pd.DataFrame):
        """Визуализация процесса генерации признаков."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Генерация признаков для DRL', fontsize=16, fontweight='bold')
        
        # График 1: Количество признаков
        ax1 = axes[0, 0]
        categories = ['Сырые данные', 'С техническими\nиндикаторами', 'С ценовыми\nпризнаками', 'Все признаки']
        feature_counts = [len(original_data.columns), 
                         len([col for col in enhanced_data.columns if col.startswith(('sma_', 'ema_', 'rsi_', 'macd', 'bb_', 'atr'))]),
                         len([col for col in enhanced_data.columns if 'price_change' in col or 'return' in col or 'volatility' in col]),
                         len(enhanced_data.columns)]
        
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        bars = ax1.bar(categories, feature_counts, color=colors, edgecolor='black')
        ax1.set_title('Количество признаков на каждом этапе')
        ax1.set_ylabel('Количество признаков')
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, feature_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # График 2: Корреляционная матрица ключевых признаков
        ax2 = axes[0, 1]
        key_features = ['close', 'volume', 'sma_21', 'rsi_14', 'macd', 'atr', 'bb_width']
        available_features = [f for f in key_features if f in enhanced_data.columns]
        
        if len(available_features) > 1:
            corr_matrix = enhanced_data[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            ax2.set_title('Корреляция ключевых признаков')
        else:
            ax2.text(0.5, 0.5, 'Недостаточно признаков\nдля корреляционной матрицы', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Корреляция признаков')
        
        # График 3: Временные признаки
        ax3 = axes[1, 0]
        time_cols = [col for col in enhanced_data.columns if col.endswith(('_sin', '_cos')) or col in ['hour', 'day_of_week']]
        
        if time_cols:
            # Показываем пример временных признаков
            sample_data = enhanced_data[time_cols[:4]].iloc[-100:] if len(enhanced_data) > 100 else enhanced_data[time_cols[:4]]
            for col in sample_data.columns[:4]:
                ax3.plot(sample_data.index, sample_data[col], label=col, alpha=0.8)
            ax3.set_title('Примеры временных признаков')
            ax3.set_ylabel('Значение')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Временные признаки\nне найдены', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Временные признаки')
        
        # График 4: Статистика признаков
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Подсчет различных типов признаков
        tech_indicators = len([col for col in enhanced_data.columns if col.startswith(('sma_', 'ema_', 'rsi_', 'macd', 'bb_', 'atr'))])
        price_features = len([col for col in enhanced_data.columns if 'price_change' in col or 'return' in col])
        time_features = len([col for col in enhanced_data.columns if col.endswith(('_sin', '_cos', 'hour', 'day_of_week', 'month'))])
        vol_features = len([col for col in enhanced_data.columns if 'volatility' in col or 'vol' in col])
        
        stats_text = f"""СТАТИСТИКА ПРИЗНАКОВ DRL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Всего признаков: {len(enhanced_data.columns)}
Исходные OHLCV: {len(original_data.columns)}
Технические индикаторы: {tech_indicators}
Ценовые признаки: {price_features}
Временные признаки: {time_features}
Признаки волатильности: {vol_features}

Период данных: {enhanced_data.index[0].strftime('%Y-%m-%d')} - {enhanced_data.index[-1].strftime('%Y-%m-%d')}
Записей после обработки: {len(enhanced_data):,}
Потеря данных из-за NaN: {len(original_data) - len(enhanced_data)} записей
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        return fig

    def visualize_trading_environment_data(self, data: pd.DataFrame, env_config: TradingConfig):
        """Визуализация данных для торговой среды."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Данные для торговой среды DRL', fontsize=16, fontweight='bold')
        
        # График 1: Lookback window пример
        ax1 = axes[0, 0]
        lookback = env_config.lookback_window
        sample_end = min(lookback + 50, len(data))
        sample_data = data.iloc[:sample_end]
        
        ax1.plot(range(len(sample_data)), sample_data['close'], color='blue', linewidth=1.5)
        
        # Показываем окно lookback
        ax1.axvspan(0, lookback, alpha=0.3, color='green', label=f'Lookback Window ({lookback})')
        ax1.axvspan(lookback, len(sample_data), alpha=0.3, color='orange', label='Prediction Zone')
        
        ax1.set_title(f'Пример Lookback Window (размер: {lookback})')
        ax1.set_xlabel('Временной шаг')
        ax1.set_ylabel('Цена закрытия')
        ax1.legend()
        
        # График 2: Параметры торговой среды
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        params_text = f"""ПАРАМЕТРЫ ТОРГОВОЙ СРЕДЫ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Начальный баланс: ${env_config.initial_balance:,.2f}
Комиссия за транзакцию: {env_config.transaction_fee*100:.2f}%
Проскальзывание: {env_config.slippage*100:.3f}%
Максимальный размер позиции: {env_config.max_position_size*100:.0f}%
Минимальная сумма сделки: ${env_config.min_trade_amount:.2f}
Размер окна lookback: {env_config.lookback_window}

РАЗМЕРЫ ДАННЫХ:
Всего записей: {len(data):,}
Признаков на временной шаг: {len(data.columns)}
Размер наблюдения: ({env_config.lookback_window}, {len(data.columns) + 4})
        """
        
        ax2.text(0.05, 0.95, params_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # График 3: Распределение признаков
        ax3 = axes[1, 0]
        feature_sample = data.select_dtypes(include=[np.number]).iloc[:1000]  # Берем выборку для скорости
        
        # Показываем распределение нескольких ключевых признаков
        key_features = ['close', 'volume']
        if 'rsi_14' in feature_sample.columns:
            key_features.append('rsi_14')
        if 'macd' in feature_sample.columns:
            key_features.append('macd')
        
        for i, feature in enumerate(key_features[:3]):
            if feature in feature_sample.columns:
                ax3.hist(feature_sample[feature].dropna(), bins=30, alpha=0.6, 
                        label=feature, density=True)
        
        ax3.set_title('Распределение ключевых признаков')
        ax3.set_xlabel('Значение')
        ax3.set_ylabel('Плотность')
        ax3.legend()
        
        # График 4: Временная структура данных
        ax4 = axes[1, 1]
        
        # Показываем как выглядят данные по времени
        time_sample = data.tail(100)  # Последние 100 записей
        
        # Нормализуем данные для отображения
        close_norm = (time_sample['close'] - time_sample['close'].min()) / (time_sample['close'].max() - time_sample['close'].min())
        volume_norm = (time_sample['volume'] - time_sample['volume'].min()) / (time_sample['volume'].max() - time_sample['volume'].min())
        
        ax4.plot(range(len(time_sample)), close_norm, label='Close (norm)', color='blue')
        ax4.plot(range(len(time_sample)), volume_norm, label='Volume (norm)', color='orange', alpha=0.7)
        
        if 'rsi_14' in time_sample.columns:
            rsi_norm = time_sample['rsi_14'] / 100  # RSI уже от 0 до 100
            ax4.plot(range(len(time_sample)), rsi_norm, label='RSI/100', color='green', alpha=0.7)
        
        ax4.set_title('Временная структура данных (нормализованные)')
        ax4.set_xlabel('Временной шаг')
        ax4.set_ylabel('Нормализованное значение')
        ax4.legend()
        
        plt.tight_layout()
        return fig


def main():
    """Основная функция для запуска визуализации DRL данных."""
    print("🚀 Запуск визуalizации данных DRL системы...")
    
    # Создаем визуализатор
    visualizer = DRLDataVisualizer()
    
    try:
        print("🔄 Сбор данных...")
        
        # Конфигурация для сбора данных с 2018 года
        data_config = DataConfig(
            symbol='BTC/USDT',
            timeframe='1d',  # Используем дневной таймфрейм для долгосрочных данных
            start_date='2018-01-01',
            exchange='binance'
        )
        
        # Собираем данные
        collector = CryptoDataCollector(data_config)
        raw_data = collector.collect_ohlcv_data()
        
        if raw_data.empty or len(raw_data) < 100:
            print("⚠️ Данных недостаточно для анализа, создаем реалистичные демо-данные Bitcoin с 2018 года...")
            # Создаем реалистичные данные Bitcoin с 2018 года
            dates = pd.date_range('2018-01-01', '2025-07-12', freq='1D')
            np.random.seed(42)
            
            # Реалистичная модель цены Bitcoin с трендом
            n_days = len(dates)
            
            # Начальная цена Bitcoin в 2018 году (~$3,200)
            initial_price = 3200
            
            # Создаем реалистичные ценовые движения
            # Bitcoin рос с пиками в 2021 (~$67k) и 2024 (~$100k+)
            price_trend = np.zeros(n_days)
            
            # Добавляем долгосрочный тренд роста
            for i in range(n_days):
                years_passed = i / 365.25
                # Экспоненциальный рост с циклами
                trend_multiplier = 1 + years_passed * 0.8  # Базовый рост
                
                # Добавляем циклы (bull/bear markets)
                cycle_component = 1 + 0.5 * np.sin(years_passed * 2 * np.pi / 4)  # 4-летний цикл
                
                price_trend[i] = initial_price * trend_multiplier * cycle_component
            
            # Добавляем волатильность
            volatility = 0.05  # 5% дневная волатильность Bitcoin
            price_changes = np.random.normal(0, volatility, n_days)
            
            # Применяем изменения к тренду
            prices = np.zeros(n_days)
            prices[0] = initial_price
            
            for i in range(1, n_days):
                # Комбинируем тренд и случайные изменения
                trend_price = price_trend[i]
                daily_change = price_changes[i]
                
                # Применяем изменение к предыдущей цене с притяжением к тренду
                prices[i] = prices[i-1] * (1 + daily_change) * 0.9 + trend_price * 0.1
            
            # Создаем OHLCV данные
            data = []
            
            for i in range(n_days):
                close_price = prices[i]
                
                # Open цена (предыдущее закрытие + небольшой гэп)
                if i == 0:
                    open_price = close_price
                else:
                    gap = np.random.normal(0, 0.01) * close_price
                    open_price = prices[i-1] + gap
                
                # High и Low на основе внутридневной волатильности
                intraday_range = abs(close_price * np.random.uniform(0.02, 0.08))
                high_price = max(open_price, close_price) + np.random.uniform(0, intraday_range * 0.5)
                low_price = min(open_price, close_price) - np.random.uniform(0, intraday_range * 0.5)
                
                # Volume коррелирует с волатильностью и ценой
                base_volume = 1000000000  # Базовый объем в USDT
                price_factor = close_price / initial_price  # Больше объема при высокой цене
                volatility_factor = 1 + abs(close_price - open_price) / open_price * 10
                volume = int(base_volume * price_factor * volatility_factor * np.random.uniform(0.5, 2.0))
                
                data.append({
                    'open': max(0.01, open_price),
                    'high': max(0.01, high_price),
                    'low': max(0.01, low_price),
                    'close': max(0.01, close_price),
                    'volume': volume
                })
            
            raw_data = pd.DataFrame(data, index=dates)
            
            print(f"📊 Создано {len(raw_data)} дней реалистичных данных Bitcoin")
            print(f"💰 Цена: от ${raw_data['close'].min():.2f} до ${raw_data['close'].max():.2f}")
            print(f"📈 Общий рост: {((raw_data['close'].iloc[-1] / raw_data['close'].iloc[0]) - 1)*100:.1f}%")
        
        print(f"✅ Получено {len(raw_data)} записей данных")
        
        print("🔧 Генерация признаков...")
        
        # Генерируем признаки
        feature_engineer = FeatureEngineer()
        enhanced_data = feature_engineer.add_all_features(raw_data)
        
        print(f"✅ Сгенерировано {len(enhanced_data.columns)} признаков")
        
        print("📊 Создание визуализаций...")
        
        # 1. Визуализация сырых данных
        print("   📈 График сырых данных...")
        fig1 = visualizer.visualize_raw_data(raw_data, data_config.symbol)
        
        # 2. Визуализация технических индикаторов
        print("   🔧 График технических индикаторов...")
        fig2 = visualizer.visualize_technical_indicators(enhanced_data, data_config.symbol)
        
        # 3. Визуализация процесса генерации признаков
        print("   ⚙️ График генерации признаков...")
        fig3 = visualizer.visualize_feature_engineering(raw_data, enhanced_data)
        
        # 4. Визуализация данных для торговой среды
        print("   🎮 График торговой среды...")
        env_config = TradingConfig(lookback_window=50)
        fig4 = visualizer.visualize_trading_environment_data(enhanced_data, env_config)
        
        # Показываем все графики
        print("✅ Визуализация готова! Показываю графики...")
        plt.show()
        
        # Сохраняем графики
        output_dir = Path('CryptoTrade/ai/DRL/logs/data_visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig1.savefig(output_dir / f'drl_raw_data_{timestamp}.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / f'drl_technical_indicators_{timestamp}.png', dpi=300, bbox_inches='tight')
        fig3.savefig(output_dir / f'drl_feature_engineering_{timestamp}.png', dpi=300, bbox_inches='tight')
        fig4.savefig(output_dir / f'drl_trading_environment_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        print(f"💾 Графики сохранены в: {output_dir}")
        
        # Выводим итоговую статистику
        print(f"\n📋 РЕЗЮМЕ ДАННЫХ DRL СИСТЕМЫ:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"💰 Торговая пара: {data_config.symbol}")
        print(f"⏰ Таймфрейм: {data_config.timeframe}")
        print(f"📅 Период данных: {raw_data.index[0].strftime('%Y-%m-%d %H:%M')} - {raw_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Сырых записей: {len(raw_data):,}")
        print(f"🔧 Обработанных записей: {len(enhanced_data):,}")
        print(f"📈 Всего признаков: {len(enhanced_data.columns)}")
        print(f"🎯 Размер окна lookback: {env_config.lookback_window}")
        print(f"📐 Размер наблюдения для DRL: ({env_config.lookback_window}, {len(enhanced_data.columns) + 4})")
        print(f"💾 Графики сохранены: {output_dir}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()