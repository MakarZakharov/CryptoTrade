#!/usr/bin/env python3
"""
IMPROVED STAS Trading Strategies for Backtrader
Покращені торгові стратегії для криптовалют
"""

import backtrader as bt
import math


class ImprovedTrendFollowStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА Трендова стратегія:
    - Додано фільтр волатильності
    - Покращений риск-менеджмент
    - Динамічний стоп-лосс
    """
    
    params = (
        ('ema_period', 21),
        ('trend_strength', 1.015),  # Знижено з 1.02 до 1.015 (1.5%)
        ('position_size', 0.95),
        ('stop_loss', 0.04),        # Знижено з 5% до 4%
        ('take_profit', 0.12),      # Знижено з 15% до 12%
        ('atr_period', 14),         # Додано ATR фільтр
        ('min_atr', 0.01),          # Мінімальна волатільність 1%
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.ema = bt.indicators.EMA(period=self.params.ema_period)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        self.order = None
        self.entry_price = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        ema_value = self.ema[0]
        atr_value = self.atr[0]
        volatility = atr_value / current_price if current_price > 0 else 0
        
        if not self.position:
            # Сигнал на покупку: ціна вище EMA + зростання + достатня волатильність
            if (current_price > ema_value and 
                current_price > self.data.close[-1] * self.params.trend_strength and
                ema_value > self.ema[-1] and
                volatility > self.params.min_atr):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BUY: Price={current_price:.2f}, EMA={ema_value:.2f}, ATR={volatility:.3f}')
                
        else:
            # Динамічний стоп-лосс на основі ATR
            dynamic_stop = max(self.params.stop_loss, volatility * 2)
            stop_price = self.entry_price * (1 - dynamic_stop)
            profit_price = self.entry_price * (1 + self.params.take_profit)
            
            if (current_price <= stop_price or 
                current_price >= profit_price or
                (current_price < ema_value and current_price < self.data.close[-1])):
                
                self.order = self.close()
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.log(f'SELL: Price={current_price:.2f}, PnL={pnl:.2f}%')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ImprovedMomentumStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА Моментум стратегія:
    - Послаблені умови входу
    - Додано фільтр об'єму
    - Кращий тайминг виходу
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_threshold', 55),      # Підвищено з 45 до 55
        ('momentum_period', 3),
        ('momentum_threshold', 1.02), # Знижено з 1.03 до 1.02
        ('position_size', 0.90),
        ('volume_filter', True),
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.momentum = bt.indicators.RateOfChange(period=self.params.momentum_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=10)
        self.order = None
        self.entry_price = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        rsi_value = self.rsi[0]
        momentum_value = self.momentum[0]
        
        # Перевірка об'єму
        volume_ok = True
        if self.params.volume_filter and len(self.data.volume) > 0:
            volume_ok = self.data.volume[0] > self.volume_sma[0] * 0.8
        
        if not self.position:
            # Послаблені умови входу
            if (rsi_value < self.params.rsi_threshold and
                momentum_value > (self.params.momentum_threshold - 1) * 100 and
                volume_ok):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BUY: Price={current_price:.2f}, RSI={rsi_value:.2f}, Mom={momentum_value:.2f}')
                
        else:
            # Покращений вихід
            profit_target = self.entry_price * 1.05  # 5% прибуток
            stop_loss = self.entry_price * 0.97      # 3% стоп
            
            if (current_price >= profit_target or
                current_price <= stop_loss or
                rsi_value > 75 or 
                momentum_value < -1):
                
                self.order = self.close()
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.log(f'SELL: Price={current_price:.2f}, PnL={pnl:.2f}%')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ImprovedBreakoutStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА Breakout стратегія:
    - Зменшена консервативність
    - Додано більше рівнів входу
    - Покращений риск-менеджмент
    """
    
    params = (
        ('lookback_period', 15),    # Знижено з 20 до 15
        ('breakout_threshold', 1.008), # Знижено з 1.01 до 1.008
        ('volume_confirm', True),
        ('position_size', 0.85),
        ('stop_loss', 0.06),        # Знижено з 8% до 6%
        ('take_profit', 0.15),      # Додано тейк-профіт
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.highest = bt.indicators.Highest(period=self.params.lookback_period)
        self.lowest = bt.indicators.Lowest(period=self.params.lookback_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=10)
        self.order = None
        self.entry_price = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        high_breakout = self.highest[-1] * self.params.breakout_threshold
        volume_ok = True
        
        if self.params.volume_confirm and len(self.data.volume) > 0:
            volume_ok = self.data.volume[0] > self.volume_sma[0] * 0.7  # Послаблено з 1.0 до 0.7
        
        if not self.position:
            # Більш агресивний вхід
            if (current_price > high_breakout and volume_ok):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BUY BREAKOUT: Price={current_price:.2f}, High={high_breakout:.2f}')
                
        else:
            # Покращене управління позицією
            stop_price = self.entry_price * (1 - self.params.stop_loss)
            profit_price = self.entry_price * (1 + self.params.take_profit)
            
            if (current_price <= stop_price or 
                current_price >= profit_price):
                
                self.order = self.close()
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.log(f'SELL: Price={current_price:.2f}, PnL={pnl:.2f}%')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ImprovedQuickScalpStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА Quick Scalp стратегія:
    - Послаблені умови входу
    - Більш реалістичні цілі
    - Покращений фільтр сигналів
    """
    
    params = (
        ('fast_ema', 5),            # Збільшено з 3 до 5
        ('slow_ema', 12),           # Збільшено з 8 до 12
        ('rsi_period', 10),         # Збільшено з 7 до 10
        ('scalp_target', 0.015),    # Знижено з 2% до 1.5%
        ('quick_stop', 0.01),       # Знижено з 1.5% до 1%
        ('position_size', 1.0),
        ('rsi_min', 35),            # Додано мінімальний RSI
        ('rsi_max', 70),            # Додано максимальний RSI
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.fast_ema = bt.indicators.EMA(period=self.params.fast_ema)
        self.slow_ema = bt.indicators.EMA(period=self.params.slow_ema)
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.order = None
        self.entry_price = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        fast_ema = self.fast_ema[0]
        slow_ema = self.slow_ema[0]
        rsi_value = self.rsi[0]
        
        if not self.position:
            # Послаблені умови входу
            if (fast_ema > slow_ema and 
                rsi_value > self.params.rsi_min and 
                rsi_value < self.params.rsi_max and
                current_price > self.data.close[-1]):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'SCALP BUY: Price={current_price:.2f}, RSI={rsi_value:.2f}')
                
        else:
            # Реалістичні цілі
            profit_price = self.entry_price * (1 + self.params.scalp_target)
            stop_price = self.entry_price * (1 - self.params.quick_stop)
            
            if (current_price >= profit_price or 
                current_price <= stop_price or
                rsi_value > 75):
                
                self.order = self.close()
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.log(f'SCALP EXIT: Price={current_price:.2f}, PnL={pnl:.2f}%')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ImprovedSimpleStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА Simple стратегія:
    - Послаблені умови
    - Додано SMA фільтр
    - Кращий тайминг
    """
    
    params = (
        ('lookback_period', 2),     # Знижено з 3 до 2
        ('min_growth', 1.005),      # Знижено з 1.01 до 1.005 (0.5%)
        ('position_size', 1.0),
        ('max_hold_days', 8),       # Збільшено з 5 до 8
        ('sma_period', 20),         # Додано SMA фільтр
        ('use_sma_filter', True),
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.params.sma_period)
        self.order = None
        self.entry_price = 0
        self.entry_bar = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        
        if not self.position:
            # Послаблені умови входу
            if len(self.data.close) >= self.params.lookback_period:
                old_price = self.data.close[-self.params.lookback_period]
                growth_ratio = current_price / old_price
                
                # SMA фільтр
                sma_ok = True
                if self.params.use_sma_filter:
                    sma_ok = current_price > self.sma[0]
                
                if growth_ratio >= self.params.min_growth and sma_ok:
                    cash_amount = self.broker.getcash() * self.params.position_size
                    size = max(cash_amount / current_price, 0.001)
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.entry_bar = len(self.data)
                    self.log(f'SIMPLE BUY: Price={current_price:.2f}, Growth={growth_ratio:.3f}')
                    
        else:
            # Покращений вихід
            bars_held = len(self.data) - self.entry_bar
            profit_target = self.entry_price * 1.03  # 3% прибуток
            
            if (current_price >= profit_target or
                current_price < self.entry_price * 0.98 or  # 2% стоп
                bars_held >= self.params.max_hold_days):
                
                self.order = self.close()
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.log(f'SIMPLE EXIT: Price={current_price:.2f}, PnL={pnl:.2f}%, Held={bars_held}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ImprovedPriceActionStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА Price Action стратегія:
    - Додані фільтри для зменшення кількості угод
    - Кращі умови входу
    - Покращений риск-менеджмент
    """
    
    params = (
        ('min_body_ratio', 0.7),    # Збільшено з 0.6
        ('shadow_ratio', 0.25),     # Зменшено з 0.3
        ('confirmation_bars', 2),   # Збільшено з 1
        ('position_size', 0.50),    # Зменшено з 0.85
        ('sma_filter', True),       # Додано SMA фільтр
        ('sma_period', 20),
        ('min_volume_ratio', 1.2),  # Додано об'ємний фільтр
        ('take_profit', 0.08),      # Додано тейк-профіт
        ('stop_loss', 0.04),        # Додано стоп-лосс
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.params.sma_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=10)
        self.order = None
        self.entry_price = 0
        self.bullish_count = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def is_strong_bullish_candle(self, index=0):
        """Перевіряє чи є свічка сильно бичачою"""
        open_price = self.data.open[index]
        close_price = self.data.close[index]
        high_price = self.data.high[index]
        low_price = self.data.low[index]
        
        if close_price <= open_price:
            return False
            
        body_size = close_price - open_price
        total_range = high_price - low_price
        
        if total_range == 0:
            return False
            
        body_ratio = body_size / total_range
        upper_shadow = high_price - close_price
        lower_shadow = open_price - low_price
        shadow_ratio = max(upper_shadow, lower_shadow) / total_range
        
        return (body_ratio >= self.params.min_body_ratio and 
                shadow_ratio <= self.params.shadow_ratio)
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        
        # Фільтри
        sma_ok = True
        volume_ok = True
        
        if self.params.sma_filter:
            sma_ok = current_price > self.sma[0]
            
        if len(self.data.volume) > 0:
            volume_ok = self.data.volume[0] > self.volume_sma[0] * self.params.min_volume_ratio
        
        if not self.position:
            # Підрахунок сильних бичачих свічок
            if self.is_strong_bullish_candle():
                self.bullish_count += 1
            else:
                self.bullish_count = 0
            
            # Суворіші умови входу з фільтрами
            if (self.bullish_count >= self.params.confirmation_bars and
                sma_ok and volume_ok):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.bullish_count = 0
                self.log(f'PA BUY: Price={current_price:.2f}, Bullish={self.bullish_count}')
                
        else:
            # Покращене управління ризиками
            profit_price = self.entry_price * (1 + self.params.take_profit)
            stop_price = self.entry_price * (1 - self.params.stop_loss)
            
            if (current_price >= profit_price or
                current_price <= stop_price or
                (self.data.close[0] < self.data.open[0] and current_price < self.sma[0])):
                
                self.order = self.close()
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.log(f'PA EXIT: Price={current_price:.2f}, PnL={pnl:.2f}%')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ImprovedDCAStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА DCA стратегія:
    - Більш активні покупки
    - Кращий фільтр тренду
    - Динамічне управління позиціями
    """
    
    params = (
        ('buy_interval', 3),        # Зменшено з 5 до 3
        ('buy_amount_pct', 0.15),   # Збільшено з 0.10 до 0.15
        ('trend_filter', True),
        ('sma_period', 30),         # Зменшено з 50 до 30
        ('max_positions', 8),       # Зменшено з 10 до 8
        ('profit_threshold', 1.15), # Знижено з 1.20 до 1.15
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.params.sma_period)
        self.order = None
        self.bar_count = 0
        self.buy_count = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        self.bar_count += 1
        
        # Більш активна регулярна покупка
        if self.bar_count % self.params.buy_interval == 0:
            trend_ok = True
            
            if self.params.trend_filter:
                # Послаблений фільтр тренду
                trend_ok = current_price > self.sma[0] * 0.98
            
            if (trend_ok and 
                self.buy_count < self.params.max_positions):
                
                cash_amount = self.broker.getcash() * self.params.buy_amount_pct
                if cash_amount > current_price:
                    size = max(cash_amount / current_price, 0.001)
                    self.order = self.buy(size=size)
                    self.buy_count += 1
                    self.log(f'DCA BUY #{self.buy_count}: Price={current_price:.2f}')
        
        # Вихід при досягненні цілі
        if (self.position and 
            current_price > self.sma[0] * self.params.profit_threshold):
            
            self.order = self.close()
            self.buy_count = 0
            self.log(f'DCA SELL ALL: Price={current_price:.2f}, Profit target reached')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ImprovedGridTradingStrategy(bt.Strategy):
    """
    ПОКРАЩЕНА Grid Trading стратегія:
    - Менші інтервали сітки
    - Більш активна торгівля
    - Покращене управління ризиками
    """
    
    params = (
        ('grid_size', 0.015),       # Зменшено з 0.02 до 0.015
        ('grid_levels', 4),         # Зменшено з 5 до 4
        ('position_size', 0.25),    # Збільшено з 0.20 до 0.25
        ('base_price_period', 15),  # Зменшено з 20 до 15
        ('profit_target', 0.03),    # Додано цільовий прибуток
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.params.base_price_period)
        self.order = None
        self.grid_positions = {}    # Словник замість списку
        self.base_price = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        
        # Встановлюємо базову ціну
        if self.base_price == 0:
            self.base_price = self.sma[0]
            return
        
        # Розраховуємо рівні сітки
        for level in range(1, self.params.grid_levels + 1):
            buy_level = self.base_price * (1 - self.params.grid_size * level)
            sell_level = self.base_price * (1 + self.params.grid_size * level * 0.8)
            
            # Покупка на рівні сітки
            if (current_price <= buy_level and 
                level not in self.grid_positions):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                if cash_amount > current_price:
                    size = max(cash_amount / current_price, 0.001)
                    self.order = self.buy(size=size)
                    self.grid_positions[level] = current_price
                    self.log(f'GRID BUY L{level}: Price={current_price:.2f}')
                    break
        
        # Продаж при досягненні цільового прибутку
        if (self.position and 
            len(self.grid_positions) > 0 and
            current_price > self.base_price * (1 + self.params.profit_target)):
            
            self.order = self.close()
            avg_entry = sum(self.grid_positions.values()) / len(self.grid_positions)
            profit = (current_price - avg_entry) / avg_entry * 100
            self.grid_positions.clear()
            self.log(f'GRID SELL ALL: Price={current_price:.2f}, Profit={profit:.2f}%')
        
        # Періодичне оновлення базової ціни
        if len(self.data) % self.params.base_price_period == 0:
            self.base_price = self.sma[0]
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


# Експорт покращених стратегій
__all__ = [
    'ImprovedTrendFollowStrategy',
    'ImprovedMomentumStrategy', 
    'ImprovedBreakoutStrategy',
    'ImprovedQuickScalpStrategy',
    'ImprovedSimpleStrategy',
    'ImprovedPriceActionStrategy',
    'ImprovedDCAStrategy',
    'ImprovedGridTradingStrategy'
]