"""
Визуализация Bitcoin с 2018 года на двух таймфреймах: дневном (1d) и 15-минутном (15m).
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
from data_processing.feature_engineering import FeatureEngineer

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.facecolor'] = 'white'

class BitcoinDualTimeframeVisualizer:
    """Класс для визуализации Bitcoin на двух таймфреймах."""
    
    def __init__(self):
        self.setup_matplotlib()
        
    def setup_matplotlib(self):
        """Настройка matplotlib для корректного отображения."""
        plt.rcParams['axes.unicode_minus'] = False
        
    def create_realistic_bitcoin_data(self, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
        """Создание реалистичных данных Bitcoin."""
        print(f"📊 Создание реалистичных данных Bitcoin {timeframe} с {start_date} по {end_date}...")
        
        if timeframe == '1d':
            freq = '1D'
        elif timeframe == '15m':
            freq = '15T'
        else:
            freq = '1H'
            
        dates = pd.date_range(start_date, end_date, freq=freq)
        np.random.seed(42)
        
        n_periods = len(dates)
        
        # Начальная цена Bitcoin в 2018 году (~$3,200)
        initial_price = 3200
        
        # Создаем реалистичные ценовые движения
        price_trend = np.zeros(n_periods)
        
        # Добавляем долгосрочный тренд роста
        for i in range(n_periods):
            if timeframe == '1d':
                years_passed = i / 365.25
            elif timeframe == '15m':
                years_passed = i / (365.25 * 24 * 4)  # 15-минутные интервалы в году
            else:
                years_passed = i / (365.25 * 24)
                
            # Экспоненциальный рост с циклами
            trend_multiplier = 1 + years_passed * 0.8  # Базовый рост
            
            # Добавляем циклы (bull/bear markets)
            cycle_component = 1 + 0.5 * np.sin(years_passed * 2 * np.pi / 4)  # 4-летний цикл
            
            price_trend[i] = initial_price * trend_multiplier * cycle_component
        
        # Добавляем волатильность в зависимости от таймфрейма
        if timeframe == '1d':
            volatility = 0.05  # 5% дневная волатильность
        elif timeframe == '15m':
            volatility = 0.008  # 0.8% 15-минутная волатильность
        else:
            volatility = 0.02
            
        price_changes = np.random.normal(0, volatility, n_periods)
        
        # Применяем изменения к тренду
        prices = np.zeros(n_periods)
        prices[0] = initial_price
        
        for i in range(1, n_periods):
            trend_price = price_trend[i]
            daily_change = price_changes[i]
            
            # Применяем изменение к предыдущей цене с притяжением к тренду
            prices[i] = prices[i-1] * (1 + daily_change) * 0.95 + trend_price * 0.05
        
        # Создаем OHLCV данные
        data = []
        
        for i in range(n_periods):
            close_price = prices[i]
            
            # Open цена
            if i == 0:
                open_price = close_price
            else:
                gap = np.random.normal(0, 0.005) * close_price
                open_price = prices[i-1] + gap
            
            # High и Low на основе внутридневной волатильности
            if timeframe == '1d':
                intraday_range = abs(close_price * np.random.uniform(0.02, 0.08))
            elif timeframe == '15m':
                intraday_range = abs(close_price * np.random.uniform(0.003, 0.015))
            else:
                intraday_range = abs(close_price * np.random.uniform(0.01, 0.03))
                
            high_price = max(open_price, close_price) + np.random.uniform(0, intraday_range * 0.5)
            low_price = min(open_price, close_price) - np.random.uniform(0, intraday_range * 0.5)
            
            # Volume коррелирует с волатильностью и ценой
            base_volume = 1000000000 if timeframe == '1d' else 50000000  # Меньше объема для 15m
            price_factor = close_price / initial_price
            volatility_factor = 1 + abs(close_price - open_price) / open_price * 10
            volume = int(base_volume * price_factor * volatility_factor * np.random.uniform(0.5, 2.0))
            
            data.append({
                'open': max(0.01, open_price),
                'high': max(0.01, high_price),
                'low': max(0.01, low_price),
                'close': max(0.01, close_price),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        
        print(f"✅ Создано {len(df)} записей данных {timeframe}")
        print(f"💰 Цена: от ${df['close'].min():.2f} до ${df['close'].max():.2f}")
        print(f"📈 Общий рост: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1)*100:.1f}%")
        
        return df
    
    def collect_or_create_data(self, symbol: str, timeframe: str, start_date: str) -> pd.DataFrame:
        """Собрать данные с API или создать реалистичные демо-данные."""
        print(f"🔄 Попытка сбора реальных данных {symbol} {timeframe}...")
        
        data_config = DataConfig(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            exchange='binance'
        )
        
        collector = CryptoDataCollector(data_config)
        real_data = collector.collect_ohlcv_data()
        
        if real_data.empty or len(real_data) < 100:
            print(f"⚠️ Реальных данных недостаточно для {timeframe}, создаем демо-данные...")
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Для 15m берем данные за последний год для разумного размера
            if timeframe == '15m':
                start_demo = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            else:
                start_demo = start_date
                
            demo_data = self.create_realistic_bitcoin_data(start_demo, end_date, timeframe)
            return demo_data
        else:
            print(f"✅ Получено {len(real_data)} реальных записей {timeframe}")
            return real_data
    
    def create_dual_timeframe_chart(self, daily_data: pd.DataFrame, minute_data: pd.DataFrame):
        """Создание графика с двумя таймфреймами."""
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        fig.suptitle('📈 Bitcoin (BTC/USDT) - Анализ двух таймфреймов с 2018 года', 
                     fontsize=18, fontweight='bold')
        
        # График 1: Дневной график цены
        ax1 = axes[0, 0]
        ax1.plot(daily_data.index, daily_data['close'], color='#1f77b4', linewidth=1.5, label='Close Price')
        ax1.fill_between(daily_data.index, daily_data['low'], daily_data['high'], 
                        alpha=0.2, color='gray', label='High-Low Range')
        ax1.set_title('📊 Дневной график (1D) - Долгосрочный тренд', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Цена (USDT)', fontsize=12)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # График 2: 15-минутный график (последние данные)
        ax2 = axes[0, 1]
        # Берем последние 1000 точек для читаемости
        recent_15m = minute_data.tail(1000) if len(minute_data) > 1000 else minute_data
        ax2.plot(recent_15m.index, recent_15m['close'], color='#ff7f0e', linewidth=1, label='Close Price')
        ax2.fill_between(recent_15m.index, recent_15m['low'], recent_15m['high'], 
                        alpha=0.2, color='orange', label='High-Low Range')
        ax2.set_title('⚡ 15-минутный график - Детальная динамика', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Цена (USDT)', fontsize=12)
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # График 3: Объем торгов - дневной
        ax3 = axes[1, 0]
        ax3.bar(daily_data.index, daily_data['volume'], alpha=0.7, color='green', width=1)
        ax3.set_title('📊 Объем торгов - Дневной', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Объем', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # График 4: Объем торгов - 15-минутный
        ax4 = axes[1, 1]
        recent_15m_vol = minute_data.tail(1000) if len(minute_data) > 1000 else minute_data
        ax4.bar(recent_15m_vol.index, recent_15m_vol['volume'], alpha=0.7, color='purple', width=0.01)
        ax4.set_title('⚡ Объем торгов - 15-минутный', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Объем', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # График 5: Сравнение волатильности
        ax5 = axes[2, 0]
        daily_returns = daily_data['close'].pct_change().dropna()
        minute_returns = minute_data['close'].pct_change().dropna()
        
        ax5.hist(daily_returns * 100, bins=50, alpha=0.7, color='blue', 
                label=f'Дневные изменения (σ={daily_returns.std()*100:.2f}%)', density=True)
        ax5.hist(minute_returns * 100, bins=100, alpha=0.5, color='red', 
                label=f'15-мин изменения (σ={minute_returns.std()*100:.2f}%)', density=True)
        ax5.set_title('📊 Распределение изменений цены', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Изменение цены (%)', fontsize=12)
        ax5.set_ylabel('Плотность', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # График 6: Статистика и информация
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Статистика
        daily_stats = {
            'start_price': daily_data['close'].iloc[0],
            'end_price': daily_data['close'].iloc[-1],
            'min_price': daily_data['close'].min(),
            'max_price': daily_data['close'].max(),
            'total_return': ((daily_data['close'].iloc[-1] / daily_data['close'].iloc[0]) - 1) * 100,
            'avg_volume': daily_data['volume'].mean(),
            'records': len(daily_data)
        }
        
        minute_stats = {
            'start_price': minute_data['close'].iloc[0],
            'end_price': minute_data['close'].iloc[-1],
            'min_price': minute_data['close'].min(),
            'max_price': minute_data['close'].max(),
            'total_return': ((minute_data['close'].iloc[-1] / minute_data['close'].iloc[0]) - 1) * 100,
            'avg_volume': minute_data['volume'].mean(),
            'records': len(minute_data)
        }
        
        stats_text = f"""📊 СТАТИСТИКА BITCOIN (BTC/USDT):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 ДНЕВНОЙ ТАЙМФРЕЙМ (1D):
   Период: {daily_data.index[0].strftime('%Y-%m-%d')} - {daily_data.index[-1].strftime('%Y-%m-%d')}
   Записей: {daily_stats['records']:,}
   Начальная цена: ${daily_stats['start_price']:,.2f}
   Конечная цена: ${daily_stats['end_price']:,.2f}
   Мин/Макс: ${daily_stats['min_price']:,.2f} / ${daily_stats['max_price']:,.2f}
   Общий рост: {daily_stats['total_return']:+.1f}%
   Средний объем: {daily_stats['avg_volume']:,.0f}
   Волатильность: {daily_returns.std()*100:.2f}%

⚡ 15-МИНУТНЫЙ ТАЙМФРЕЙМ (15M):
   Период: {minute_data.index[0].strftime('%Y-%m-%d %H:%M')} - {minute_data.index[-1].strftime('%Y-%m-%d %H:%M')}
   Записей: {minute_stats['records']:,}
   Начальная цена: ${minute_stats['start_price']:,.2f}
   Конечная цена: ${minute_stats['end_price']:,.2f}
   Мин/Макс: ${minute_stats['min_price']:,.2f} / ${minute_stats['max_price']:,.2f}
   Общий рост: {minute_stats['total_return']:+.1f}%
   Средний объем: {minute_stats['avg_volume']:,.0f}
   Волатильность: {minute_returns.std()*100:.2f}%

💡 ВЫВОДЫ:
   Соотношение волатильности: {(minute_returns.std()/daily_returns.std()):,.1f}x
   Разница в ценовом диапазоне: {((daily_stats['max_price']-daily_stats['min_price'])/daily_stats['min_price']*100):.1f}%
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                linespacing=1.5)
        
        plt.tight_layout()
        return fig
    
    def create_technical_analysis_chart(self, daily_data: pd.DataFrame, minute_data: pd.DataFrame):
        """Создание графика с техническим анализом."""
        print("🔧 Добавление технических индикаторов...")
        
        feature_engineer = FeatureEngineer()
        daily_enhanced = feature_engineer.add_technical_indicators(daily_data)
        minute_enhanced = feature_engineer.add_technical_indicators(minute_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('🔧 Bitcoin - Технический анализ на двух таймфреймах', 
                     fontsize=16, fontweight='bold')
        
        # График 1: Дневной с техническими индикаторами
        ax1 = axes[0, 0]
        ax1.plot(daily_enhanced.index, daily_enhanced['close'], label='Close', color='black', linewidth=2)
        
        if 'sma_21' in daily_enhanced.columns:
            ax1.plot(daily_enhanced.index, daily_enhanced['sma_21'], label='SMA 21', color='red', linewidth=1)
        if 'sma_50' in daily_enhanced.columns:
            ax1.plot(daily_enhanced.index, daily_enhanced['sma_50'], label='SMA 50', color='blue', linewidth=1)
        if 'ema_12' in daily_enhanced.columns:
            ax1.plot(daily_enhanced.index, daily_enhanced['ema_12'], label='EMA 12', color='orange', linewidth=1, linestyle='--')
        
        ax1.set_title('📊 Дневной график с Moving Averages', fontweight='bold')
        ax1.set_ylabel('Цена (USDT)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # График 2: 15-минутный с техническими индикаторами
        ax2 = axes[0, 1]
        recent_minute = minute_enhanced.tail(1000) if len(minute_enhanced) > 1000 else minute_enhanced
        ax2.plot(recent_minute.index, recent_minute['close'], label='Close', color='black', linewidth=1.5)
        
        if 'sma_21' in recent_minute.columns:
            ax2.plot(recent_minute.index, recent_minute['sma_21'], label='SMA 21', color='red', linewidth=1)
        if 'ema_12' in recent_minute.columns:
            ax2.plot(recent_minute.index, recent_minute['ema_12'], label='EMA 12', color='orange', linewidth=1, linestyle='--')
        
        ax2.set_title('⚡ 15-минутный график с Moving Averages', fontweight='bold')
        ax2.set_ylabel('Цена (USDT)')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # График 3: RSI сравнение
        ax3 = axes[1, 0]
        if 'rsi_14' in daily_enhanced.columns:
            ax3.plot(daily_enhanced.index, daily_enhanced['rsi_14'], label='RSI 14 (Daily)', color='purple', linewidth=1.5)
        if 'rsi_14' in minute_enhanced.columns:
            recent_rsi = minute_enhanced['rsi_14'].tail(1000) if len(minute_enhanced) > 1000 else minute_enhanced['rsi_14']
            ax3.plot(recent_rsi.index, recent_rsi, label='RSI 14 (15m)', color='cyan', linewidth=1, alpha=0.7)
        
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Перекупленность (70)')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Перепроданность (30)')
        ax3.set_title('📊 RSI - Сравнение таймфреймов', fontweight='bold')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # График 4: MACD
        ax4 = axes[1, 1]
        if 'macd' in daily_enhanced.columns:
            ax4.plot(daily_enhanced.index, daily_enhanced['macd'], label='MACD (Daily)', color='blue', linewidth=1.5)
        if 'macd_signal' in daily_enhanced.columns:
            ax4.plot(daily_enhanced.index, daily_enhanced['macd_signal'], label='Signal (Daily)', color='red', linewidth=1.5)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('📊 MACD - Дневной таймфрейм', fontweight='bold')
        ax4.set_ylabel('MACD')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    """Главная функция для создания графиков Bitcoin на двух таймфреймах."""
    print("🚀 Запуск визуализации Bitcoin на двух таймфреймах...")
    print("📊 Таймфреймы: Дневной (1D) и 15-минутный (15M)")
    print("📅 Период: с 2018 года по настоящее время")
    print("=" * 80)
    
    visualizer = BitcoinDualTimeframeVisualizer()
    
    try:
        # Сбор данных на двух таймфреймах
        print("📈 Получение данных на дневном таймфрейме...")
        daily_data = visualizer.collect_or_create_data('BTC/USDT', '1d', '2018-01-01')
        
        print("⚡ Получение данных на 15-минутном таймфрейме...")
        minute_data = visualizer.collect_or_create_data('BTC/USDT', '15m', '2018-01-01')
        
        print("📊 Создание основных графиков...")
        fig1 = visualizer.create_dual_timeframe_chart(daily_data, minute_data)
        
        print("🔧 Создание графиков технического анализа...")
        fig2 = visualizer.create_technical_analysis_chart(daily_data, minute_data)
        
        # Сохранение графиков
        output_dir = Path('CryptoTrade/ai/DRL/logs/bitcoin_charts')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig1.savefig(output_dir / f'bitcoin_dual_timeframe_{timestamp}.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / f'bitcoin_technical_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        print("✅ Визуализация готова! Показываю графики...")
        plt.show()
        
        print(f"\n💾 Графики сохранены в: {output_dir}")
        print(f"📊 Файлы:")
        print(f"   - bitcoin_dual_timeframe_{timestamp}.png")
        print(f"   - bitcoin_technical_analysis_{timestamp}.png")
        
        print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📈 Дневных записей: {len(daily_data):,}")
        print(f"⚡ 15-минутных записей: {len(minute_data):,}")
        print(f"💰 Диапазон цен: ${min(daily_data['close'].min(), minute_data['close'].min()):,.2f} - ${max(daily_data['close'].max(), minute_data['close'].max()):,.2f}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()