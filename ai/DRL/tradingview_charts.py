"""
Создание профессиональных графиков в стиле TradingView для DRL системы торговли криптовалютой.
Использует японские свечи и технические индикаторы с профессиональным оформлением.
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к модулям DRL
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_processing.data_collector import CryptoDataCollector, DataConfig
from data_processing.feature_engineering import FeatureEngineer


class TradingViewStyleCharts:
    """Класс для создания графиков в стиле TradingView."""
    
    def __init__(self):
        # Настройка стиля TradingView
        self.tradingview_style = mpf.make_marketcolors(
            up='#26A69A',      # Зеленый для роста (как в TradingView)
            down='#EF5350',    # Красный для падения
            edge='inherit',
            wick={'up': '#26A69A', 'down': '#EF5350'},
            volume='in',
            ohlc='i'
        )
        
        # Конфигурация для дополнительных графиков будет создаваться по мере необходимости
        
        # Профессиональная тема
        self.dark_style = mpf.make_mpf_style(
            marketcolors=self.tradingview_style,
            gridstyle='-',
            gridcolor='#2E2E2E',
            facecolor='#1E1E1E',
            figcolor='#1E1E1E',
            edgecolor='white',
            gridaxis='both'
        )
        
        self.light_style = mpf.make_mpf_style(
            marketcolors=self.tradingview_style,
            gridstyle='-',
            gridcolor='#E0E0E0',
            facecolor='white',
            figcolor='white',
            edgecolor='black',
            gridaxis='both'
        )
        
    def create_realistic_bitcoin_data(self, start_date='2018-01-01', timeframe='1d'):
        """Создание реалистичных данных Bitcoin с 2018 года."""
        print(f"🔄 Создание реалистичных данных Bitcoin с {start_date}...")
        
        if timeframe == '1d':
            freq = '1D'
            n_periods = (datetime.now() - pd.Timestamp(start_date)).days
        elif timeframe == '15m':
            freq = '15T'
            # Для 15-минутных данных берем только последние 3 месяца
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            n_periods = 90 * 24 * 4  # 90 дней * 24 часа * 4 интервала по 15 минут
        elif timeframe == '1h':
            freq = '1H'
            # Для часовых данных берем последний год
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            n_periods = 365 * 24
        else:
            freq = '1D'
            n_periods = (datetime.now() - pd.Timestamp(start_date)).days
        
        dates = pd.date_range(start_date, periods=n_periods, freq=freq)
        np.random.seed(42)
        
        # Начальная цена зависит от периода
        if start_date.startswith('2018'):
            initial_price = 3200  # Bitcoin в начале 2018
        else:
            initial_price = 45000  # Более поздний период
        
        # Создаем реалистичные ценовые движения
        prices = np.zeros(n_periods)
        prices[0] = initial_price
        
        # Параметры волатильности в зависимости от таймфрейма
        if timeframe == '15m':
            base_volatility = 0.008  # 0.8% для 15-минутных интервалов
        elif timeframe == '1h':
            base_volatility = 0.015  # 1.5% для часовых интервалов
        else:
            base_volatility = 0.035  # 3.5% для дневных интервалов
        
        # Генерация реалистичных цен Bitcoin
        for i in range(1, n_periods):
            # Базовое случайное изменение
            random_change = np.random.normal(0, base_volatility)
            
            # Добавляем кластеризацию волатильности
            if i > 10:
                recent_returns = [prices[j] / prices[j-1] - 1 for j in range(max(1, i-10), i)]
                recent_volatility = np.std(recent_returns)
                volatility_factor = max(0.5, min(2.0, 1 + recent_volatility * 2))
            else:
                volatility_factor = 1
            
            # Реалистичная модель Bitcoin с учетом исторических данных
            if start_date.startswith('2018') and timeframe == '1d':
                years_passed = i / 365.25
                
                # Основан на реальной истории Bitcoin
                # 2018: Bear market (падение с 17k до 3k)
                # 2019-2020: Накопление (~3k-10k)
                # 2021: Bull run (10k-67k)
                # 2022: Bear market (67k-15k)
                # 2023-2024: Восстановление (15k-70k)
                
                if years_passed < 1:  # 2018: медвежий рынок
                    target_price_factor = 0.8 - years_passed * 0.6  # От 0.8 до 0.2
                elif years_passed < 2.5:  # 2019-2020: накопление
                    target_price_factor = 0.2 + (years_passed - 1) * 0.4  # От 0.2 до 0.8
                elif years_passed < 3.5:  # 2021: бычий рынок
                    target_price_factor = 0.8 + (years_passed - 2.5) * 2.8  # От 0.8 до 3.6
                elif years_passed < 4.5:  # 2022: коррекция
                    target_price_factor = 3.6 - (years_passed - 3.5) * 2.1  # От 3.6 до 1.5
                else:  # 2023+: восстановление
                    target_price_factor = 1.5 + (years_passed - 4.5) * 1.0  # От 1.5 до 2.5+
                
                # Целевая цена на основе исторической модели
                target_price = initial_price * target_price_factor
                
                # Притяжение к целевой цене (mean reversion)
                price_diff = target_price - prices[i-1]
                trend_component = price_diff / prices[i-1] * 0.001  # Слабое притяжение
                
                # Добавляем циклы внутри периодов
                cycle_component = 0.05 * np.sin(years_passed * 2 * np.pi * 2)  # Полугодовые циклы
                
                daily_trend = trend_component + cycle_component * 0.0001
            else:
                daily_trend = 0.0001  # Слабый восходящий тренд для других таймфреймов
            
            # Применяем изменения
            price_change = random_change * volatility_factor + daily_trend
            
            # Ограничиваем экстремальные изменения (реалистично для Bitcoin)
            price_change = max(-0.15, min(0.15, price_change))  # Максимум ±15% за день
            
            new_price = prices[i-1] * (1 + price_change)
            
            # Реалистичные границы для Bitcoin
            if timeframe == '1d':
                min_price, max_price = 500, 100000  # Bitcoin исторический диапазон
            else:
                min_price, max_price = prices[0] * 0.2, prices[0] * 5  # Более консервативный диапазон
            
            prices[i] = max(min_price, min(max_price, new_price))
        
        # Создаем OHLCV данные
        data = []
        for i in range(n_periods):
            close_price = max(0.01, prices[i])
            
            # Open цена
            if i == 0:
                open_price = close_price
            else:
                gap_factor = np.random.normal(0, 0.002)  # Небольшие гэпы
                open_price = max(0.01, prices[i-1] * (1 + gap_factor))
            
            # Внутридневная волатильность
            if timeframe == '15m':
                intraday_volatility = 0.003  # 0.3% для 15 минут
            elif timeframe == '1h':
                intraday_volatility = 0.005  # 0.5% для часа
            else:
                intraday_volatility = 0.02   # 2% для дня
            
            intraday_range = abs(close_price * np.random.uniform(0.005, intraday_volatility))
            
            # High и Low
            high_price = max(open_price, close_price) + np.random.uniform(0, intraday_range)
            low_price = min(open_price, close_price) - np.random.uniform(0, intraday_range)
            
            # Убеждаемся в логичности данных
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume зависит от волатильности и времени
            base_volume = 500000000 if timeframe == '1d' else (50000000 if timeframe == '1h' else 15000000)
            price_factor = close_price / initial_price
            volatility_factor = 1 + abs(close_price - open_price) / open_price * 20
            
            # Добавляем циклические паттерны для объема (больше в рабочее время)
            if timeframe in ['15m', '1h']:
                hour = dates[i].hour
                if 8 <= hour <= 22:  # Рабочие часы
                    time_factor = 1.5
                else:
                    time_factor = 0.6
            else:
                weekday = dates[i].weekday()
                if weekday < 5:  # Рабочие дни
                    time_factor = 1.2
                else:
                    time_factor = 0.8
            
            volume = int(base_volume * price_factor * volatility_factor * time_factor * np.random.uniform(0.5, 2.0))
            
            data.append({
                'Open': max(0.01, open_price),
                'High': max(0.01, high_price),
                'Low': max(0.01, low_price),
                'Close': max(0.01, close_price),
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        
        print(f"📊 Создано {len(df)} записей для таймфрейма {timeframe}")
        print(f"💰 Цена: от ${df['Close'].min():.2f} до ${df['Close'].max():.2f}")
        print(f"📈 Общий рост: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1)*100:.1f}%")
        
        return df
    
    def add_technical_indicators_for_chart(self, df: pd.DataFrame):
        """Добавление технических индикаторов для графика."""
        data = df.copy()
        
        # SMA (Простые скользящие средние)
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean() if len(data) > 200 else None
        
        # EMA (Экспоненциальные скользящие средние)
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        delta = data['Close'].diff()
        gains = delta.where(delta > 0, 0).rolling(window=14).mean()
        losses = (-delta).where(delta < 0, 0).rolling(window=14).mean()
        rs = gains / losses
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
        bb_std = data['Close'].rolling(window=bb_period).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume SMA
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        return data
    
    def create_tradingview_chart(self, df: pd.DataFrame, symbol: str, timeframe: str, style='light'):
        """Создание основного графика в стиле TradingView."""
        
        # Подготовка данных с техническими индикаторами
        data = self.add_technical_indicators_for_chart(df)
        
        # Выбор стиля
        chart_style = self.light_style if style == 'light' else self.dark_style
        
        # Обрезаем данные до последних записей для лучшей видимости
        if len(data) > 500:
            display_data = data.tail(500).copy()
        else:
            display_data = data.copy()
        
        # Удаляем NaN значения
        display_data = display_data.dropna()
        
        if len(display_data) < 10:
            print("⚠️ Недостаточно данных для создания графика")
            return None
        
        # Подготовка дополнительных линий для графика
        addplots = []
        
        # SMA линии - убеждаемся что данные той же длины
        if 'SMA_20' in display_data.columns and not display_data['SMA_20'].isna().all():
            sma_20_clean = display_data['SMA_20'].dropna()
            if len(sma_20_clean) > 0:
                addplots.append(mpf.make_addplot(display_data['SMA_20'], color='blue', width=1.5, alpha=0.8))
        
        if 'SMA_50' in display_data.columns and not display_data['SMA_50'].isna().all():
            sma_50_clean = display_data['SMA_50'].dropna()
            if len(sma_50_clean) > 0:
                addplots.append(mpf.make_addplot(display_data['SMA_50'], color='orange', width=1.5, alpha=0.8))
        
        if 'SMA_200' in display_data.columns and display_data['SMA_200'] is not None and not display_data['SMA_200'].isna().all():
            sma_200_clean = display_data['SMA_200'].dropna()
            if len(sma_200_clean) > 0:
                addplots.append(mpf.make_addplot(display_data['SMA_200'], color='red', width=2, alpha=0.9))
        
        # Bollinger Bands
        if all(col in display_data.columns for col in ['BB_Upper', 'BB_Lower']) and not display_data['BB_Upper'].isna().all():
            addplots.append(mpf.make_addplot(display_data['BB_Upper'], color='gray', width=1, alpha=0.6, linestyle='--'))
            addplots.append(mpf.make_addplot(display_data['BB_Lower'], color='gray', width=1, alpha=0.6, linestyle='--'))
        
        # Создание графика
        title = f"{symbol} - {timeframe.upper()} Timeframe | TradingView Style"
        
        # Сохранение в файл
        output_dir = Path('CryptoTrade/ai/DRL/logs/tradingview_charts')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'tradingview_{symbol.replace("/", "_")}_{timeframe}_{timestamp}.png'
        
        # Конфигурация графика
        config = dict(
            type='candle',
            style=chart_style,
            title=title,
            ylabel='Цена (USDT)',
            ylabel_lower='Объем',
            volume=True,
            addplot=addplots if addplots else None,
            figsize=(16, 12),
            savefig=dict(fname=filename, dpi=300, bbox_inches='tight'),
            show_nontrading=False,
            returnfig=True
        )
        

        
        print(f"📊 Создание TradingView графика для {symbol} ({timeframe})...")
        
        try:
            fig, axes = mpf.plot(display_data, **config)
            
            # Добавляем дополнительную информацию
            current_price = display_data['Close'].iloc[-1]
            price_change = display_data['Close'].iloc[-1] - display_data['Close'].iloc[-2]
            price_change_pct = (price_change / display_data['Close'].iloc[-2]) * 100
            
            info_text = f"Текущая цена: ${current_price:,.2f}\n"
            info_text += f"Изменение: ${price_change:+.2f} ({price_change_pct:+.2f}%)\n"
            info_text += f"Объем: {display_data['Volume'].iloc[-1]:,.0f}"
            
            # Добавляем информационный блок на график
            axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
            print(f"✅ График сохранен: {filename}")
            return fig, filename
            
        except Exception as e:
            print(f"❌ Ошибка создания графика: {e}")
            return None
    
    def create_multi_indicator_chart(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Создание графика с множественными индикаторами в отдельных панелях."""
        
        data = self.add_technical_indicators_for_chart(df)
        
        # Обрезаем данные
        if len(data) > 300:
            display_data = data.tail(300)
        else:
            display_data = data
        
        display_data = display_data.dropna()
        
        if len(display_data) < 20:
            print("⚠️ Недостаточно данных для создания мульти-индикаторного графика")
            return None
        
        # Подготовка панелей для индикаторов
        addplots = []
        
        # SMA на основной панели
        if 'SMA_20' in display_data.columns and not display_data['SMA_20'].isna().all():
            addplots.append(mpf.make_addplot(display_data['SMA_20'], color='blue', width=1.5, panel=0))
        
        if 'SMA_50' in display_data.columns and not display_data['SMA_50'].isna().all():
            addplots.append(mpf.make_addplot(display_data['SMA_50'], color='orange', width=1.5, panel=0))
        
        # RSI на отдельной панели
        if 'RSI' in display_data.columns and not display_data['RSI'].isna().all():
            addplots.append(mpf.make_addplot(display_data['RSI'], color='purple', width=1.5, panel=1, ylabel='RSI'))
            
            # Линии перекупленности/перепроданности для RSI
            rsi_70 = pd.Series([70] * len(display_data), index=display_data.index)
            rsi_30 = pd.Series([30] * len(display_data), index=display_data.index)
            addplots.append(mpf.make_addplot(rsi_70, color='red', width=1, linestyle='--', panel=1, alpha=0.7))
            addplots.append(mpf.make_addplot(rsi_30, color='green', width=1, linestyle='--', panel=1, alpha=0.7))
        
        # MACD на отдельной панели
        if all(col in display_data.columns for col in ['MACD', 'MACD_Signal']) and not display_data['MACD'].isna().all():
            addplots.append(mpf.make_addplot(display_data['MACD'], color='blue', width=1.5, panel=2, ylabel='MACD'))
            addplots.append(mpf.make_addplot(display_data['MACD_Signal'], color='red', width=1.5, panel=2))
            
            # MACD гистограмма
            if 'MACD_Histogram' in display_data.columns:
                addplots.append(mpf.make_addplot(display_data['MACD_Histogram'], color='gray', width=1, 
                                               panel=2, type='bar', alpha=0.6))
        
        # Сохранение
        output_dir = Path('CryptoTrade/ai/DRL/logs/tradingview_charts')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'multi_indicators_{symbol.replace("/", "_")}_{timeframe}_{timestamp}.png'
        
        title = f"{symbol} - {timeframe.upper()} | Мульти-индикаторный анализ"
        
        print(f"📊 Создание мульти-индикаторного графика для {symbol} ({timeframe})...")
        
        try:
            fig, axes = mpf.plot(
                display_data,
                type='candle',
                style=self.light_style,
                title=title,
                ylabel='Цена (USDT)',
                ylabel_lower='Объем',
                volume=True,
                addplot=addplots,
                figsize=(16, 14),
                panel_ratios=(3, 1, 1),  # Основная панель больше, индикаторы меньше
                savefig=dict(fname=filename, dpi=300, bbox_inches='tight'),
                returnfig=True
            )
            
            print(f"✅ Мульти-индикаторный график сохранен: {filename}")
            return fig, filename
            
        except Exception as e:
            print(f"❌ Ошибка создания мульти-индикаторного графика: {e}")
            return None
    
    def create_comparison_charts(self, daily_data: pd.DataFrame, hourly_data: pd.DataFrame, symbol: str):
        """Создание сравнительных графиков разных таймфреймов."""
        
        print(f"📊 Создание сравнительных графиков {symbol}...")
        
        # Создаем графики для обоих таймфреймов
        daily_result = self.create_tradingview_chart(daily_data, symbol, '1d')
        hourly_result = self.create_tradingview_chart(hourly_data, symbol, '1h')
        
        results = []
        
        if daily_result:
            results.append(('Daily', daily_result[1]))
        
        if hourly_result:
            results.append(('Hourly', hourly_result[1]))
        
        # Создаем также мульти-индикаторные графики
        daily_multi = self.create_multi_indicator_chart(daily_data, symbol, '1d')
        hourly_multi = self.create_multi_indicator_chart(hourly_data, symbol, '1h')
        
        if daily_multi:
            results.append(('Daily Multi-Indicators', daily_multi[1]))
        
        if hourly_multi:
            results.append(('Hourly Multi-Indicators', hourly_multi[1]))
        
        return results


def main():
    """Основная функция для создания TradingView-стиля графиков."""
    print("🚀 Запуск создания профессиональных TradingView графиков...")
    print("=" * 80)
    
    # Создаем класс для TradingView графиков
    tv_charts = TradingViewStyleCharts()
    
    symbol = 'BTC/USDT'
    
    try:
        # 1. Создаем данные для дневного таймфрейма (с 2018)
        print("🔄 Создание данных для дневного таймфрейма (с 2018)...")
        daily_data = tv_charts.create_realistic_bitcoin_data('2018-01-01', '1d')
        
        # 2. Создаем данные для часового таймфрейма (последний год)
        print("🔄 Создание данных для часового таймфрейма (последний год)...")
        hourly_data = tv_charts.create_realistic_bitcoin_data('2024-01-01', '1h')
        
        # 3. Создаем профессиональные графики
        print("📊 Создание профессиональных TradingView графиков...")
        results = tv_charts.create_comparison_charts(daily_data, hourly_data, symbol)
        
        # 4. Выводим результаты
        print("\n" + "=" * 80)
        print("🎉 TRADINGVIEW ГРАФИКИ СОЗДАНЫ УСПЕШНО!")
        print("=" * 80)
        
        print("📋 Созданные графики:")
        for chart_type, filepath in results:
            print(f"  ✅ {chart_type}: {filepath}")
        
        # 5. Статистика данных
        print(f"\n📊 СТАТИСТИКА ДАННЫХ:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"💰 Символ: {symbol}")
        print(f"📅 Дневные данные: {daily_data.index[0].strftime('%Y-%m-%d')} - {daily_data.index[-1].strftime('%Y-%m-%d')} ({len(daily_data)} свечей)")
        print(f"🕐 Часовые данные: {hourly_data.index[0].strftime('%Y-%m-%d %H:%M')} - {hourly_data.index[-1].strftime('%Y-%m-%d %H:%M')} ({len(hourly_data)} свечей)")
        print(f"💵 Цена (дневные): ${daily_data['Close'].iloc[0]:.2f} → ${daily_data['Close'].iloc[-1]:.2f}")
        print(f"💵 Цена (часовые): ${hourly_data['Close'].iloc[0]:.2f} → ${hourly_data['Close'].iloc[-1]:.2f}")
        print(f"📈 Общий рост (дневные): {((daily_data['Close'].iloc[-1] / daily_data['Close'].iloc[0]) - 1)*100:.1f}%")
        print(f"📈 Рост (часовые): {((hourly_data['Close'].iloc[-1] / hourly_data['Close'].iloc[0]) - 1)*100:.1f}%")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        print(f"\n🎯 ОСОБЕННОСТИ TRADINGVIEW ГРАФИКОВ:")
        print(f"  ✅ Японские свечи (candlesticks) с профессиональной цветовой схемой")
        print(f"  ✅ Технические индикаторы: SMA, EMA, Bollinger Bands, RSI, MACD")
        print(f"  ✅ Объемные индикаторы внизу графика")
        print(f"  ✅ Мульти-панельные графики с отдельными индикаторами")
        print(f"  ✅ Профессиональное оформление как в TradingView")
        print(f"  ✅ Высокое разрешение (300 DPI) для четкости")
        
        print(f"\n💡 Для просмотра графиков откройте файлы PNG в директории:")
        print(f"   📁 CryptoTrade/ai/DRL/logs/tradingview_charts/")
        
    except Exception as e:
        print(f"❌ Ошибка создания TradingView графиков: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()