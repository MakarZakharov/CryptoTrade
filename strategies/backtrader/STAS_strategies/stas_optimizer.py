import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pandas as pd
import itertools
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings
import os
import sys

# Добавляем путь к UniversalBacktester
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'backtest', 'HistoricalBacktest', 'backtrader', 'UniversalBacktest'))

from universal_backtester import UniversalBacktester
from STAS_strategy import STASStrategy

warnings.filterwarnings('ignore')


class OptimizedSTASStrategy(bt.Strategy):
    """
    Улучшенная STAS стратегия с оптимизациями на основе backtrader best practices
    """
    
    params = (
        # Основные индикаторы (расширенный диапазон для оптимизации)
        ('ema_fast', 8),           # 5-21
        ('ema_slow', 21),          # 15-55  
        ('ema_trend', 50),         # 30-100
        
        # RSI параметры (расширенный диапазон)
        ('rsi_period', 14),        # 10-21
        ('rsi_oversold_strong', 15),    # 10-20
        ('rsi_oversold', 20),           # 15-30
        ('rsi_overbought', 75),         # 65-85
        ('rsi_overbought_strong', 85),  # 80-90
        
        # MACD параметры
        ('macd_fast', 12),         # 8-16
        ('macd_slow', 26),         # 20-35
        ('macd_signal', 9),        # 7-14
        
        # Управление рисками (экстремальная оптимизация)
        ('position_size', 0.90),   # 0.80-0.95
        ('stop_loss', 0.12),       # 0.08-0.20
        ('take_profit', 3.00),     # 2.00-5.00
        ('trailing_stop', 0.75),   # 0.50-1.00
        ('trailing_dist', 0.20),   # 0.15-0.30
        
        # Фильтры качества (расширенные)
        ('volume_filter', True),
        ('trend_strength_min', 0.5),     # 0.3-0.8
        ('signal_quality_min', 5.0),     # 4.0-8.0
        ('max_risk_per_trade', 0.08),    # 0.05-0.15
        
        # Новые параметры для оптимизации
        ('volatility_filter', 0.05),     # 0.02-0.10 - фильтр по волатильности
        ('trend_confirmation', 3),       # 2-5 - количество баров для подтверждения тренда
        ('rsi_divergence', True),        # включить дивергенцию RSI
        ('profit_lock_pct', 0.25),       # 0.15-0.40 - блокировка прибыли
    )

    def __init__(self):
        # Основные индикаторы
        self.ema_fast = btind.EMA(period=self.p.ema_fast)
        self.ema_slow = btind.EMA(period=self.p.ema_slow) 
        self.ema_trend = btind.EMA(period=self.p.ema_trend)
        
        self.rsi = btind.RSI(period=self.p.rsi_period)
        self.macd = btind.MACD(
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        self.atr = btind.ATR(period=14)
        
        # Дополнительные индикаторы для улучшения качества сигналов
        self.stoch = btind.Stochastic(period=14, period_dfast=3)  # Стохастик
        self.bb = btind.BollingerBands(period=20)  # Полосы Боллинджера
        self.adx = btind.ADX(period=14)  # Индекс направленного движения
        
        # Безопасная проверка volume
        try:
            if hasattr(self.data, 'volume') and len(self.data.volume) > 0:
                self.volume_sma = btind.SMA(self.data.volume, period=20)
                self.volume_ratio = self.data.volume / self.volume_sma
            else:
                self.volume_sma = None
                self.volume_ratio = None
        except (AttributeError, IndexError):
            self.volume_sma = None
            self.volume_ratio = None
        
        # Кроссы и сигналы
        self.ema_cross_up = btind.CrossUp(self.ema_fast, self.ema_slow)
        self.ema_cross_down = btind.CrossDown(self.ema_fast, self.ema_slow)
        self.macd_cross_up = btind.CrossUp(self.macd.macd, self.macd.signal)
        self.macd_cross_down = btind.CrossDown(self.macd.macd, self.macd.signal)
        
        # Состояние торговли
        self.order = None
        self.entry_price = None
        self.trailing_stop_price = None
        self.highest_price = None
        self.profit_locked = False
        
        # Счетчики для статистики
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Минимальный период
        self.min_period = max(
            self.p.ema_trend,
            self.p.rsi_period,
            self.p.macd_slow + self.p.macd_signal,
            20,  # BB period
            14   # ADX period
        )

    def log(self, txt, dt=None):
        """Логирование"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: STAS_OPT - {txt}')

    def next(self):
        if len(self.data) < self.min_period:
            return

        if self.order:
            return

        current_price = self.data.close[0]
        if not current_price or current_price <= 0 or not self._is_data_valid():
            return

        # ВХОД В ПОЗИЦИЮ
        if not self.position:
            signal_quality = self._calculate_enhanced_signal_quality()
            
            if signal_quality >= self.p.signal_quality_min:
                size = self._calculate_adaptive_position_size(signal_quality)
                
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.highest_price = current_price
                    self.trailing_stop_price = None
                    self.profit_locked = False
                    self.trade_count += 1
                    
                    self.log(f"📈 BUY: {current_price:.4f}, Quality: {signal_quality:.1f}/10, Size: {size:.0f}")

        # УПРАВЛЕНИЕ ПОЗИЦИЕЙ
        elif self.position and self.entry_price:
            current_profit_pct = (current_price - self.entry_price) / self.entry_price
            
            # Обновляем максимальную цену
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Блокировка прибыли при достижении определенного уровня
            if not self.profit_locked and current_profit_pct >= self.p.profit_lock_pct:
                self.profit_locked = True
                # Ужесточаем стоп-лосс
                self.trailing_stop_price = current_price * (1 - self.p.trailing_dist * 0.5)
                
            # Активация трейлинг стопа
            if current_profit_pct >= self.p.trailing_stop:
                trailing_price = self.highest_price * (1 - self.p.trailing_dist)
                if not self.trailing_stop_price or trailing_price > self.trailing_stop_price:
                    self.trailing_stop_price = trailing_price

            # Проверка условий выхода
            exit_reason = self._should_exit_enhanced(current_price, current_profit_pct)
            
            if exit_reason:
                self.order = self.close()
                
                if current_profit_pct > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.log(f"📉 SELL: {current_price:.4f}, P&L: {current_profit_pct*100:.1f}%, Reason: {exit_reason}")
                
                # Сброс состояния
                self.entry_price = None
                self.highest_price = None  
                self.trailing_stop_price = None
                self.profit_locked = False

    def _calculate_enhanced_signal_quality(self) -> float:
        """Улучшенный расчет качества сигнала с дополнительными индикаторами"""
        try:
            score = 0.0
            
            # 1. Трендовый анализ с усилением (0-4 балла)
            if len(self.ema_fast) > 0 and len(self.ema_slow) > 0 and len(self.ema_trend) > 0:
                if self.ema_fast[0] > self.ema_slow[0] > self.ema_trend[0]:
                    score += 3.0  # Идеальное выравнивание
                    
                    # Проверяем силу тренда через ADX
                    if len(self.adx) > 0 and self.adx[0] > 25:
                        score += 1.0  # Сильный тренд
                elif self.ema_fast[0] > self.ema_slow[0]:
                    score += 1.5  # Частичный тренд
            
            # 2. RSI анализ с дивергенцией (0-3 балла)
            if len(self.rsi) > 0:
                rsi_val = self.rsi[0]
                
                if rsi_val < self.p.rsi_oversold_strong:
                    score += 3.0
                elif rsi_val < self.p.rsi_oversold:
                    score += 2.0
                elif 45 <= rsi_val <= 55:
                    score += 0.5
                
                # Дивергенция RSI (если включена)
                if self.p.rsi_divergence and len(self.rsi) > 5:
                    if self._check_rsi_divergence():
                        score += 0.5
            
            # 3. MACD с улучшенным анализом (0-2.5 балла)
            if len(self.macd_cross_up) > 0 and self.macd_cross_up[0]:
                score += 2.5
                
                # Проверяем позицию MACD относительно нуля
                if len(self.macd.macd) > 0 and self.macd.macd[0] > 0:
                    score += 0.5  # MACD выше нуля - дополнительный бонус
            elif len(self.macd.macd) > 0 and len(self.macd.signal) > 0 and self.macd.macd[0] > self.macd.signal[0]:
                score += 1.0
            
            # 4. Стохастик для подтверждения (0-1.5 балла) 
            if len(self.stoch.percK) > 0:
                stoch_k = self.stoch.percK[0]
                if stoch_k < 20:  # Перепроданность
                    score += 1.5
                elif stoch_k < 30:
                    score += 1.0
            
            # 5. Полосы Боллинджера (0-1 балл)
            if len(self.bb.lines.bot) > 0:
                current_price = self.data.close[0]
                if current_price <= self.bb.lines.bot[0]:  # Цена у нижней полосы
                    score += 1.0
                elif current_price <= self.bb.lines.mid[0]:  # Цена ниже средней линии
                    score += 0.5
            
            # 6. Объемный анализ (0-1 балл)
            if self.volume_ratio and len(self.volume_ratio) > 0:
                vol_ratio = self.volume_ratio[0]
                if vol_ratio > 1.5:  # Повышенный объем
                    score += 1.0
                elif vol_ratio > 1.2:
                    score += 0.5
            
            # 7. Волатильность (0-1 балл)
            if len(self.atr) > 0:
                volatility = self.atr[0] / self.data.close[0]
                if self.p.volatility_filter * 0.5 <= volatility <= self.p.volatility_filter * 2:
                    score += 1.0  # Оптимальная волатильность
                elif volatility > self.p.volatility_filter * 3:
                    score *= 0.7  # Штраф за высокую волатильность
            
            # Штрафы за неблагоприятные условия
            if len(self.rsi) > 0 and self.rsi[0] > self.p.rsi_overbought_strong:
                score *= 0.1  # Сильный штраф за перекупленность
            elif len(self.rsi) > 0 and self.rsi[0] > self.p.rsi_overbought:
                score *= 0.5
                
            return min(max(score, 0.0), 10.0)
            
        except Exception:
            return 0.0

    def _check_rsi_divergence(self) -> bool:
        """Проверка дивергенции RSI"""
        try:
            if len(self.rsi) < 10 or len(self.data.close) < 10:
                return False
            
            # Простая проверка дивергенции: цена падает, RSI растет
            price_trend = self.data.close[0] - self.data.close[-5]
            rsi_trend = self.rsi[0] - self.rsi[-5]
            
            return price_trend < 0 and rsi_trend > 0
        except Exception:
            return False

    def _calculate_adaptive_position_size(self, signal_quality: float) -> float:
        """Адаптивный расчет размера позиции с учетом качества сигнала и волатильности"""
        try:
            current_price = self.data.close[0]
            if current_price <= 0:
                return 0
            
            # Базовый размер
            base_size = (self.broker.cash * self.p.position_size) / current_price
            
            # Множитель качества сигнала (0.6 - 1.4)
            quality_multiplier = 0.6 + (signal_quality / 10) * 0.8
            
            # Корректировка на волатильность
            volatility_adj = 1.0
            if len(self.atr) > 0 and self.atr[0] > 0:
                volatility = self.atr[0] / current_price
                if volatility > self.p.volatility_filter * 2:
                    volatility_adj = 0.7  # Уменьшаем позицию при высокой волатильности
                elif volatility < self.p.volatility_filter * 0.5:
                    volatility_adj = 1.3  # Увеличиваем при низкой волатильности
            
            # Корректировка на ADX (силу тренда)
            trend_adj = 1.0
            if len(self.adx) > 0:
                adx_val = self.adx[0]
                if adx_val > 40:  # Очень сильный тренд
                    trend_adj = 1.2
                elif adx_val < 20:  # Слабый тренд
                    trend_adj = 0.8
            
            # Финальный размер
            final_size = base_size * quality_multiplier * volatility_adj * trend_adj
            
            # Ограничения безопасности
            max_size = self.broker.cash * 0.95 / current_price
            return min(final_size, max_size) if final_size > 0 else 0
            
        except Exception:
            return 0

    def _is_data_valid(self) -> bool:
        """Улучшенная проверка данных"""
        try:
            # Базовые проверки OHLC
            if (self.data.open[0] <= 0 or self.data.high[0] <= 0 or
                self.data.low[0] <= 0 or self.data.close[0] <= 0):
                return False
            
            # Логическая проверка OHLC
            if (self.data.high[0] < max(self.data.open[0], self.data.close[0]) or
                self.data.low[0] > min(self.data.open[0], self.data.close[0])):
                return False
            
            # Проверка индикаторов
            required_indicators = [self.ema_fast, self.ema_slow, self.ema_trend, self.rsi, self.macd.macd]
            for indicator in required_indicators:
                if len(indicator) == 0:
                    return False
            
            return True
            
        except Exception:
            return False

    def _should_exit_enhanced(self, current_price: float, profit_pct: float) -> str:
        """Улучшенная логика выхода с дополнительными условиями"""
        try:
            # 1. Стоп-лосс
            if profit_pct <= -self.p.stop_loss:
                return "STOP_LOSS"
            
            # 2. Трейлинг стоп
            if self.trailing_stop_price and current_price <= self.trailing_stop_price:
                return "TRAILING_STOP"
            
            # 3. Тейк-профит
            if profit_pct >= self.p.take_profit:
                return "TAKE_PROFIT"
            
            # 4. RSI сигналы выхода
            if len(self.rsi) > 0:
                rsi_val = self.rsi[0]
                if rsi_val > self.p.rsi_overbought_strong and profit_pct > 0.05:
                    return "RSI_EXTREME_OVERBOUGHT"
                elif rsi_val > self.p.rsi_overbought and profit_pct > 0.15:
                    return "RSI_OVERBOUGHT"
            
            # 5. Стохастик сигналы выхода
            if len(self.stoch.percK) > 0 and self.stoch.percK[0] > 80 and profit_pct > 0.10:
                return "STOCH_OVERBOUGHT"
            
            # 6. EMA кроссы
            if len(self.ema_cross_down) > 0 and self.ema_cross_down[0] and profit_pct > 0.03:
                return "EMA_CROSS_DOWN"
            
            # 7. MACD кроссы
            if len(self.macd_cross_down) > 0 and self.macd_cross_down[0] and profit_pct > 0.08:
                return "MACD_CROSS_DOWN"
            
            # 8. Полосы Боллинджера - выход при достижении верхней полосы
            if len(self.bb.lines.top) > 0 and current_price >= self.bb.lines.top[0] and profit_pct > 0.05:
                return "BOLLINGER_TOP"
            
            # 9. Слабый тренд по ADX при прибыли
            if len(self.adx) > 0 and self.adx[0] < 15 and profit_pct > 0.05:
                return "WEAK_TREND"
            
            return None
            
        except Exception:
            return None

    def stop(self):
        """Финальная статистика"""
        final_value = self.broker.getvalue()
        total_return = (final_value / self.broker.startingcash - 1) * 100
        
        win_rate = (self.winning_trades / max(self.trade_count, 1)) * 100
        
        self.log(f'🏁 Strategy Complete!')
        self.log(f'📊 Trades: {self.trade_count} | Wins: {self.winning_trades} | Losses: {self.losing_trades}')
        self.log(f'🎯 Win Rate: {win_rate:.1f}%')
        self.log(f'📈 Total Return: {total_return:+.2f}%')
        self.log(f'💰 Final Value: ${final_value:.2f}')


class STASParameterOptimizer:
    """Оптимизатор параметров для STAS стратегии с целью достижения 500% прибыли"""
    
    def __init__(self, target_return: float = 500.0):
        self.target_return = target_return
        self.best_results = []
        self.optimization_history = []
        
        # Диапазоны параметров для оптимизации
        self.param_ranges = {
            'ema_fast': [5, 8, 10, 13, 15, 18, 21],
            'ema_slow': [15, 21, 26, 34, 42, 50, 55],
            'ema_trend': [30, 40, 50, 60, 75, 89, 100],
            
            'rsi_period': [10, 12, 14, 16, 18, 21],
            'rsi_oversold_strong': [10, 12, 15, 18, 20],
            'rsi_oversold': [15, 20, 25, 30],
            'rsi_overbought': [65, 70, 75, 80, 85],
            'rsi_overbought_strong': [80, 83, 85, 87, 90],
            
            'macd_fast': [8, 10, 12, 14, 16],
            'macd_slow': [20, 23, 26, 29, 32, 35],
            'macd_signal': [7, 8, 9, 11, 12, 14],
            
            'position_size': [0.80, 0.85, 0.90, 0.93, 0.95],
            'stop_loss': [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
            'take_profit': [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            'trailing_stop': [0.50, 0.60, 0.75, 0.85, 1.00],
            'trailing_dist': [0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
            
            'signal_quality_min': [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
            'volatility_filter': [0.02, 0.03, 0.05, 0.07, 0.08, 0.10],
            'trend_confirmation': [2, 3, 4, 5],
            'profit_lock_pct': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        }
        
    def generate_random_params(self) -> Dict[str, Any]:
        """Генерация случайных параметров"""
        params = {}
        
        for param_name, param_range in self.param_ranges.items():
            params[param_name] = random.choice(param_range)
        
        # Логическая проверка параметров
        params = self._validate_params(params)
        
        return params
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация и коррекция параметров"""
        # EMA: fast < slow < trend
        if params['ema_fast'] >= params['ema_slow']:
            params['ema_fast'] = min(params['ema_slow'] - 1, 5)
        if params['ema_slow'] >= params['ema_trend']:
            params['ema_slow'] = min(params['ema_trend'] - 1, 21)
            
        # RSI: oversold_strong < oversold < overbought < overbought_strong
        if params['rsi_oversold_strong'] >= params['rsi_oversold']:
            params['rsi_oversold_strong'] = params['rsi_oversold'] - 5
        if params['rsi_overbought'] >= params['rsi_overbought_strong']:
            params['rsi_overbought'] = params['rsi_overbought_strong'] - 5
            
        # MACD: fast < slow
        if params['macd_fast'] >= params['macd_slow']:
            params['macd_fast'] = params['macd_slow'] - 2
            
        return params
    
    def grid_search_optimization(self, 
                                exchange: str = "binance",
                                symbol: str = "BTCUSDT", 
                                timeframe: str = "15m",
                                max_iterations: int = 50) -> pd.DataFrame:
        """Grid search оптимизация с ограниченным количеством итераций"""
        
        print(f"\n🎯 GRID SEARCH OPTIMIZATION для достижения {self.target_return}% прибыли")
        print("=" * 80)
        print(f"📊 Максимум итераций: {max_iterations}")
        print(f"📈 Тестируемые данные: {exchange}:{symbol} ({timeframe})")
        
        # Создаем бэктестер
        backtester = UniversalBacktester(
            initial_cash=100000,
            commission=0.001,
            spread=0.0005, 
            slippage=0.0002,
            require_position_size=False  # Отключаем проверку для оптимизированной стратегии
        )
        
        # Добавляем оптимизированную стратегию в реестр
        backtester.strategies_registry['OptimizedSTAS'] = {
            'class': OptimizedSTASStrategy,
            'module': 'stas_optimizer',
            'file': 'stas_optimizer.py',
            'default_params': {},
            'description': 'Optimized STAS Strategy',
            'original_name': 'OptimizedSTAS'
        }
        
        results = []
        best_return = -100
        iterations_without_improvement = 0
        max_iterations_without_improvement = 20
        
        for iteration in range(max_iterations):
            # Генерируем случайные параметры
            test_params = self.generate_random_params()
            
            print(f"\n⏳ [{iteration + 1}/{max_iterations}] Тестирование комбинации параметров...")
            
            try:
                # Запускаем бэктест
                result = backtester.run_single_backtest(
                    strategy_name='OptimizedSTAS',
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_params=test_params,
                    show_plot=False,
                    verbose=False,
                    suppress_strategy_errors=True
                )
                
                result['parameters'] = test_params.copy()
                results.append(result)
                
                # Проверяем улучшение
                if result['total_return'] > best_return:
                    best_return = result['total_return']
                    iterations_without_improvement = 0
                    print(f"🚀 НОВЫЙ РЕКОРД! Доходность: {result['total_return']:+.2f}% | Сделки: {result.get('total_trades', 0)}")
                    
                    # Проверяем достижение цели
                    if result['total_return'] >= self.target_return:
                        print(f"🎯 ЦЕЛЬ ДОСТИГНУТА! {result['total_return']:+.2f}% >= {self.target_return}%")
                        break
                else:
                    iterations_without_improvement += 1
                    print(f"📊 Доходность: {result['total_return']:+.2f}% | Лучший: {best_return:+.2f}%")
                
                # Ранний выход если долго нет улучшений
                if iterations_without_improvement >= max_iterations_without_improvement:
                    print(f"\n⏹️ Останавливаем оптимизацию: {max_iterations_without_improvement} итераций без улучшения")
                    break
                    
            except Exception as e:
                print(f"❌ Ошибка в итерации {iteration + 1}: {str(e)}")
                continue
        
        if not results:
            print("❌ Нет успешных результатов оптимизации")
            return pd.DataFrame()
        
        # Анализ результатов
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        print(f"\n🏆 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
        print("=" * 100)
        print(f"📊 Всего тестов: {len(results)}")
        print(f"🥇 Лучший результат: {results_df.iloc[0]['total_return']:+.2f}%")
        print(f"📈 Средний результат: {results_df['total_return'].mean():+.2f}%")
        print(f"🎯 Цель ({self.target_return}%): {'ДОСТИГНУТА' if best_return >= self.target_return else 'НЕ ДОСТИГНУТА'}")
        
        # Топ-5 результатов
        print(f"\n🔥 ТОП-5 РЕЗУЛЬТАТОВ:")
        print("-" * 80)
        for i, (_, row) in enumerate(results_df.head().iterrows(), 1):
            print(f"{i}. 📈 {row['total_return']:+.2f}% | 🔄 {row.get('total_trades', 0)} сделок | 🎯 {row.get('win_rate', 0):.1f}% винрейт")
        
        # Сохраняем лучшие результаты
        self.best_results = results_df.head(10).copy()
        self.optimization_history.extend(results)
        
        return results_df
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Получение лучших параметров"""
        if not self.best_results.empty:
            best_result = self.best_results.iloc[0]
            return best_result.get('parameters', {})
        return {}
    
    def save_results(self, filepath: str):
        """Сохранение результатов оптимизации"""
        if not self.optimization_history:
            print("❌ Нет результатов для сохранения")
            return
            
        # Подготовка данных для сохранения
        save_data = []
        for result in self.optimization_history:
            row = {
                'total_return': result['total_return'],
                'profit_loss': result['profit_loss'],
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'profit_factor': result.get('profit_factor', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0)
            }
            
            # Добавляем параметры
            params = result.get('parameters', {})
            row.update(params)
            
            save_data.append(row)
        
        df = pd.DataFrame(save_data)
        df.to_csv(filepath, index=False)
        print(f"✅ Результаты сохранены в: {filepath}")
    
    def analyze_parameter_importance(self):
        """Анализ важности параметров"""
        if not self.optimization_history:
            print("❌ Нет данных для анализа")
            return
        
        print(f"\n🔍 АНАЛИЗ ВАЖНОСТИ ПАРАМЕТРОВ:")
        print("=" * 60)
        
        # Собираем данные
        data = []
        for result in self.optimization_history:
            row = {'return': result['total_return']}
            row.update(result.get('parameters', {}))
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Корреляция параметров с доходностью
        correlations = {}
        for param in self.param_ranges.keys():
            if param in df.columns:
                corr = df[param].corr(df['return'])
                if not pd.isna(corr):
                    correlations[param] = abs(corr)
        
        # Сортируем по важности
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("📊 Корреляция параметров с доходностью:")
        print("-" * 60)
        for i, (param, corr) in enumerate(sorted_correlations[:10], 1):
            print(f"{i:2d}. {param:20s}: {corr:+.3f}")
        
        return sorted_correlations


def main():
    """Основная функция для запуска оптимизации"""
    
    print("🚀 STAS STRATEGY PARAMETER OPTIMIZER")
    print("=" * 50)
    print("🎯 Цель: достижение 500%+ прибыли")
    print("🔧 Метод: случайный поиск параметров")
    print("📊 Основа: улучшенная STAS стратегия")
    print("=" * 50)
    
    # Создаем оптимизатор
    optimizer = STASParameterOptimizer(target_return=500.0)
    
    # Запускаем оптимизацию
    results_df = optimizer.grid_search_optimization(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="15m", 
        max_iterations=100  # Увеличиваем количество итераций
    )
    
    # Анализируем важность параметров
    optimizer.analyze_parameter_importance()
    
    # Получаем лучшие параметры
    best_params = optimizer.get_best_parameters()
    if best_params:
        print(f"\n🏆 ЛУЧШИЕ ПАРАМЕТРЫ:")
        print("=" * 50)
        for param, value in best_params.items():
            print(f"{param:20s}: {value}")
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"stas_optimization_results_{timestamp}.csv"
    optimizer.save_results(filepath)
    
    print(f"\n🎊 Оптимизация завершена!")
    print(f"📁 Результаты сохранены: {filepath}")


if __name__ == "__main__":
    main()