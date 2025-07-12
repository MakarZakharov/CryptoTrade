"""
Визуализация данных на которых обучается STAS_ML модель.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from CryptoTrade.ai.STAS_ML.config.ml_config import MLConfig
from CryptoTrade.ai.STAS_ML.data.data_processor import CryptoDataProcessor

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def visualize_raw_data(data: pd.DataFrame, symbol: str):
    """Визуализация исходных OHLCV данных."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'📊 Исходные данные для обучения: {symbol}', fontsize=16, fontweight='bold')
    
    # График цены
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.8)
    ax1.fill_between(data.index, data['low'], data['high'], alpha=0.3, color='gray', label='High-Low Range')
    ax1.set_title('💰 Цена закрытия и диапазон High-Low')
    ax1.set_ylabel('Цена (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График объема
    ax2 = axes[0, 1]
    ax2.plot(data.index, data['volume'], color='orange', alpha=0.7)
    ax2.set_title('📈 Объем торгов')
    ax2.set_ylabel('Объем')
    ax2.grid(True, alpha=0.3)
    
    # Гистограмма дневных изменений
    ax3 = axes[1, 0]
    daily_returns = data['close'].pct_change().dropna()
    ax3.hist(daily_returns * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('📊 Распределение дневных изменений цены')
    ax3.set_xlabel('Изменение цены (%)')
    ax3.set_ylabel('Частота')
    ax3.axvline(daily_returns.mean() * 100, color='red', linestyle='--', label=f'Среднее: {daily_returns.mean()*100:.2f}%')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Статистика
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
📈 СТАТИСТИКА ДАННЫХ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Период: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}
📊 Записей: {len(data):,}
💰 Цена мин/макс: ${data['close'].min():.2f} / ${data['close'].max():.2f}
📈 Средняя цена: ${data['close'].mean():.2f}
📊 Волатильность: {daily_returns.std()*100:.2f}%
📈 Общий рост: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1)*100:.1f}%
📊 Объем среднесуточный: {data['volume'].mean():,.0f}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

def visualize_technical_indicators(data_with_indicators: pd.DataFrame, symbol: str):
    """Визуализация технических индикаторов."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f'🔧 Технические индикаторы для обучения: {symbol}', fontsize=16, fontweight='bold')
    
    # График 1: Цена + Moving Averages
    ax1 = axes[0, 0]
    ax1.plot(data_with_indicators.index, data_with_indicators['close'], label='Close', color='black', linewidth=1)
    
    # SMA индикаторы
    sma_cols = [col for col in data_with_indicators.columns if col.startswith('sma_')]
    for col in sma_cols:
        if col in data_with_indicators.columns:
            period = col.split('_')[1]
            ax1.plot(data_with_indicators.index, data_with_indicators[col], 
                    label=f'SMA {period}', alpha=0.8, linewidth=1)
    
    # EMA индикаторы
    ema_cols = [col for col in data_with_indicators.columns if col.startswith('ema_')]
    for col in ema_cols:
        if col in data_with_indicators.columns:
            period = col.split('_')[1]
            ax1.plot(data_with_indicators.index, data_with_indicators[col], 
                    label=f'EMA {period}', alpha=0.8, linewidth=1, linestyle='--')
    
    ax1.set_title('📈 Цена и скользящие средние')
    ax1.set_ylabel('Цена (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: RSI
    ax2 = axes[0, 1]
    rsi_cols = [col for col in data_with_indicators.columns if col.startswith('rsi_')]
    for col in rsi_cols:
        if col in data_with_indicators.columns:
            period = col.split('_')[1]
            ax2.plot(data_with_indicators.index, data_with_indicators[col], 
                    label=f'RSI {period}', linewidth=1.5)
    
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Перекупленность (70)')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Перепроданность (30)')
    ax2.set_title('⚡ RSI (Relative Strength Index)')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
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
               label='Histogram', alpha=0.6, color='gray', width=1)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('🌊 MACD')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # График 4: Bollinger Bands
    ax4 = axes[1, 1]
    ax4.plot(data_with_indicators.index, data_with_indicators['close'], 
            label='Close', color='black', linewidth=1.5)
    
    bb_cols = [col for col in data_with_indicators.columns if col.startswith('bb_')]
    bb_periods = set([col.split('_')[2] for col in bb_cols if len(col.split('_')) > 2])
    
    for period in bb_periods:
        upper_col = f'bb_upper_{period}'
        middle_col = f'bb_middle_{period}'
        lower_col = f'bb_lower_{period}'
        
        if all(col in data_with_indicators.columns for col in [upper_col, middle_col, lower_col]):
            ax4.plot(data_with_indicators.index, data_with_indicators[upper_col], 
                    color='red', alpha=0.7, label=f'BB Upper ({period})')
            ax4.plot(data_with_indicators.index, data_with_indicators[middle_col], 
                    color='blue', alpha=0.7, label=f'BB Middle ({period})')
            ax4.plot(data_with_indicators.index, data_with_indicators[lower_col], 
                    color='green', alpha=0.7, label=f'BB Lower ({period})')
            
            # Заливка между лентами
            ax4.fill_between(data_with_indicators.index, 
                           data_with_indicators[upper_col], 
                           data_with_indicators[lower_col], 
                           alpha=0.1, color='blue')
            break  # Показываем только первый период
    
    ax4.set_title('🎯 Bollinger Bands')
    ax4.set_ylabel('Цена (USDT)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # График 5: ATR (волатильность)
    ax5 = axes[2, 0]
    atr_cols = [col for col in data_with_indicators.columns if col.startswith('atr_')]
    for col in atr_cols:
        if col in data_with_indicators.columns:
            period = col.split('_')[1]
            ax5.plot(data_with_indicators.index, data_with_indicators[col], 
                    label=f'ATR {period}', linewidth=1.5)
    
    ax5.set_title('💥 ATR (Average True Range)')
    ax5.set_ylabel('ATR')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # График 6: Объем + OBV
    ax6 = axes[2, 1]
    
    # Нормализованный объем
    volume_norm = data_with_indicators['volume'] / data_with_indicators['volume'].max()
    ax6.bar(data_with_indicators.index, volume_norm, alpha=0.6, color='orange', 
           label='Volume (norm)', width=1)
    
    # OBV если есть
    if 'obv' in data_with_indicators.columns:
        obv_norm = data_with_indicators['obv'] / data_with_indicators['obv'].abs().max()
        ax6_twin = ax6.twinx()
        ax6_twin.plot(data_with_indicators.index, obv_norm, 
                     color='purple', linewidth=1.5, label='OBV (norm)')
        ax6_twin.set_ylabel('OBV (нормализованный)')
        ax6_twin.legend(loc='upper right')
    
    ax6.set_title('📊 Объем и OBV')
    ax6.set_ylabel('Объем (нормализованный)')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_target_distribution(target_data: np.ndarray, config: MLConfig):
    """Визуализация распределения целевой переменной."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'🎯 Целевая переменная для обучения: {config.target_type.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # График 1: Распределение целевой переменной
    ax1 = axes[0]
    
    if config.target_type == 'direction':
        # Для классификации
        unique, counts = np.unique(target_data, return_counts=True)
        colors = ['red' if x == 0 else 'green' for x in unique]
        labels = ['📉 Падение (0)' if x == 0 else '📈 Рост (1)' for x in unique]
        
        bars = ax1.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('📊 Распределение направлений движения')
        ax1.set_ylabel('Количество образцов')
        
        # Добавляем проценты
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = (count / total) * 100
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Статистика
        balance = min(counts) / max(counts) * 100
        ax1.text(0.5, 0.95, f'Баланс классов: {balance:.1f}%', 
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=11, fontweight='bold')
        
    else:
        # Для регрессии
        ax1.hist(target_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('📊 Распределение значений цели')
        ax1.set_xlabel('Значение')
        ax1.set_ylabel('Частота')
        
        # Статистика
        stats_text = f'Среднее: {np.mean(target_data):.4f}\nСтд. откл.: {np.std(target_data):.4f}'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax1.grid(True, alpha=0.3)
    
    # График 2: Временное распределение
    ax2 = axes[1]
    
    # Разбиваем на временные сегменты
    segment_size = len(target_data) // 10
    segments = []
    segment_labels = []
    
    for i in range(10):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < 9 else len(target_data)
        segment = target_data[start_idx:end_idx]
        
        if config.target_type == 'direction':
            positive_ratio = np.sum(segment == 1) / len(segment) * 100
            segments.append(positive_ratio)
        else:
            segments.append(np.mean(segment))
        
        segment_labels.append(f'Сегмент {i+1}')
    
    colors = plt.cm.RdYlGn([x/100 for x in segments]) if config.target_type == 'direction' else 'blue'
    bars = ax2.bar(range(10), segments, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f'S{i+1}' for i in range(10)])
    
    if config.target_type == 'direction':
        ax2.set_title('⏰ Доля положительных сигналов по времени')
        ax2.set_ylabel('% положительных сигналов')
        ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% баланс')
        ax2.legend()
    else:
        ax2.set_title('⏰ Среднее значение цели по времени')
        ax2.set_ylabel('Среднее значение')
    
    ax2.set_xlabel('Временные сегменты (от старых к новым)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_feature_importance_preview(data_with_indicators: pd.DataFrame, config: MLConfig):
    """Предварительный анализ важности признаков."""
    from scipy.stats import pearsonr
    
    # Создаем простую целевую переменную для анализа корреляции
    if config.target_type == 'direction':
        target = (data_with_indicators['close'].pct_change().shift(-1) > 0).astype(int)
    else:
        target = data_with_indicators['close'].pct_change().shift(-1)
    
    # Убираем NaN
    clean_data = data_with_indicators.dropna()
    target_clean = target.loc[clean_data.index].dropna()
    clean_data = clean_data.loc[target_clean.index]
    
    # Выбираем числовые колонки (исключая OHLCV)
    feature_columns = [col for col in clean_data.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume'] 
                      and clean_data[col].dtype in ['float64', 'int64']]
    
    # Рассчитываем корреляции
    correlations = {}
    for col in feature_columns[:20]:  # Берем первые 20 для визуализации
        try:
            corr, p_value = pearsonr(clean_data[col], target_clean)
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except:
            continue
    
    if not correlations:
        return None
    
    # Сортируем по важности
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Создаем график
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    features = [item[0] for item in sorted_features]
    importances = [item[1] for item in sorted_features]
    
    # Цветовая схема
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Абсолютная корреляция с целевой переменной')
    ax.set_title('🔍 Предварительная важность признаков (корреляция)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Добавляем значения
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        width = bar.get_width()
        ax.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def main():
    """Основная функция для запуска визуализации."""
    print("🚀 Запуск визуализации данных обучения STAS_ML...")
    
    # Создаем конфигурацию (используем последние настройки из логов)
    config = MLConfig(
        exchange='binance',
        symbol='BTCUSDT', 
        timeframe='1d',
        model_type='xgboost',
        target_type='direction',
        lookback_window=30
    )
    
    print(f"📊 Конфигурация: {config.symbol} ({config.exchange}) - {config.timeframe}")
    print(f"🎯 Цель: {config.target_type}")
    print(f"📈 Окно lookback: {config.lookback_window}")
    
    # Создаем процессор данных
    processor = CryptoDataProcessor(config)
    
    try:
        # Загружаем исходные данные
        print("\n🔄 Загрузка исходных данных...")
        raw_data = processor.load_data()
        
        # Добавляем технические индикаторы
        print("🔧 Добавление технических индикаторов...")
        data_with_indicators = processor.add_technical_indicators(raw_data)
        
        # Создаем целевую переменную
        print("🎯 Создание целевой переменной...")
        target = processor.create_target(data_with_indicators)
        
        print(f"✅ Данные подготовлены:")
        print(f"   📊 Исходных записей: {len(raw_data)}")
        print(f"   🔧 С индикаторами: {len(data_with_indicators)} записей, {len(data_with_indicators.columns)} признаков")
        print(f"   🎯 Целевых значений: {len(target)}")
        
        # Создаем графики
        print("\n📊 Создание визуализаций...")
        
        # 1. Исходные данные
        print("   📈 График исходных данных...")
        fig1 = visualize_raw_data(raw_data, config.symbol)
        
        # 2. Технические индикаторы
        print("   🔧 График технических индикаторов...")
        fig2 = visualize_technical_indicators(data_with_indicators, config.symbol)
        
        # 3. Целевая переменная
        print("   🎯 График целевой переменной...")
        fig3 = visualize_target_distribution(target, config)
        
        # 4. Важность признаков
        print("   🔍 Анализ важности признаков...")
        fig4 = visualize_feature_importance_preview(data_with_indicators, config)
        
        # Показываем графики
        print("\n✅ Визуализация готова! Показываю графики...")
        plt.show()
        
        # Сохраняем графики
        output_dir = Path('CryptoTrade/ai/STAS_ML/logs/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1.savefig(output_dir / f'{config.symbol}_{config.timeframe}_raw_data.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / f'{config.symbol}_{config.timeframe}_indicators.png', dpi=300, bbox_inches='tight')
        fig3.savefig(output_dir / f'{config.symbol}_{config.timeframe}_target.png', dpi=300, bbox_inches='tight')
        if fig4:
            fig4.savefig(output_dir / f'{config.symbol}_{config.timeframe}_feature_importance.png', dpi=300, bbox_inches='tight')
        
        print(f"💾 Графики сохранены в: {output_dir}")
        
        # Выводим резюме
        print(f"\n📋 РЕЗЮМЕ ДАННЫХ ДЛЯ ОБУЧЕНИЯ:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🏢 Биржа: {config.exchange.upper()}")
        print(f"💰 Торговая пара: {config.symbol}")
        print(f"⏰ Таймфрейм: {config.timeframe}")
        print(f"📅 Период данных: {raw_data.index[0].strftime('%Y-%m-%d')} - {raw_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"📊 Всего записей: {len(raw_data):,}")
        print(f"🔧 Технические индикаторы: {len(data_with_indicators.columns) - len(raw_data.columns)}")
        print(f"🎯 Тип цели: {config.target_type}")
        print(f"📈 Окно lookback: {config.lookback_window} периодов")
        
        if config.target_type == 'direction':
            positive_ratio = np.sum(target == 1) / len(target) * 100
            print(f"📈 Положительных сигналов: {positive_ratio:.1f}%")
            print(f"📉 Отрицательных сигналов: {100-positive_ratio:.1f}%")
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()