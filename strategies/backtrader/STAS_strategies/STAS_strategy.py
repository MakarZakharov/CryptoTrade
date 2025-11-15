#!/usr/bin/env python3
"""
STAS Trading Strategies for Backtrader
Колекція торгових стратегій для криптовалют
"""

import backtrader as bt
import math


class TrendFollowStrategy(bt.Strategy):
    """
    Трендова стратегія: купуємо вище EMA + зростання
    Використовує EMA для визначення тренду та подає сигнали на покупку при висхідному тренді
    """
    
    params = (
        ('ema_period', 21),
        ('trend_strength', 1.02),  # 2% зростання для підтвердження тренду
        ('position_size', 0.95),   # 95% від доступного капіталу
        ('stop_loss', 0.05),       # 5% стоп-лосс
        ('take_profit', 0.15),     # 15% тейк-профіт
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.ema = bt.indicators.EMA(period=self.params.ema_period)
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
        
        if not self.position:
            # Сигнал на покупку: ціна вище EMA та зростає
            if (current_price > ema_value and 
                current_price > self.data.close[-1] * self.params.trend_strength and
                ema_value > self.ema[-1]):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BUY: Price={current_price:.2f}, EMA={ema_value:.2f}')
                
        else:
            # Управління позицією
            stop_price = self.entry_price * (1 - self.params.stop_loss)
            profit_price = self.entry_price * (1 + self.params.take_profit)
            
            if (current_price <= stop_price or 
                current_price >= profit_price or
                current_price < ema_value):
                
                self.order = self.close()
                self.log(f'SELL: Price={current_price:.2f}, Entry={self.entry_price:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MomentumStrategy(bt.Strategy):
    """
    Швидка моментум стратегія: купуємо при сильному зростанні
    Використовує RSI та швидкість зміни ціни для виявлення моментуму
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_threshold', 45),     # RSI нижче цього значення для входу
        ('momentum_period', 3),    # Період для розрахунку моментуму
        ('momentum_threshold', 1.03), # 3% зростання за період
        ('position_size', 0.90),
        ('quick_exit', True),      # Швидкий вихід з позицій
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.momentum = bt.indicators.RateOfChange(period=self.params.momentum_period)
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
        
        if not self.position:
            # Сигнал на покупку: RSI не перекуплено + сильний моментум
            if (rsi_value < self.params.rsi_threshold and
                momentum_value > (self.params.momentum_threshold - 1) * 100 and
                current_price > self.data.close[-1]):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BUY: Price={current_price:.2f}, RSI={rsi_value:.2f}, Mom={momentum_value:.2f}')
                
        else:
            # Швидкий вихід при зміні моментуму
            if (rsi_value > 70 or 
                momentum_value < 0 or
                current_price < self.data.close[-1] * 0.98):
                
                self.order = self.close()
                self.log(f'SELL: Price={current_price:.2f}, RSI={rsi_value:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class BreakoutStrategy(bt.Strategy):
    """
    Пробій екстремумів: купуємо при пробої максимуму
    Визначає локальні екстремуми та торгує пробої
    """
    
    params = (
        ('lookback_period', 20),   # Період для пошуку екстремумів
        ('breakout_threshold', 1.01), # 1% пробій
        ('volume_confirm', True),  # Підтвердження об'ємом
        ('position_size', 0.85),
        ('stop_loss', 0.08),
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
            volume_ok = self.data.volume[0] > self.volume_sma[0]
        
        if not self.position:
            # Сигнал на покупку: пробій максимуму з об'ємом
            if (current_price > high_breakout and volume_ok):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BUY BREAKOUT: Price={current_price:.2f}, High={high_breakout:.2f}')
                
        else:
            # Стоп-лосс
            stop_price = self.entry_price * (1 - self.params.stop_loss)
            if current_price <= stop_price:
                self.order = self.close()
                self.log(f'STOP LOSS: Price={current_price:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class QuickScalpStrategy(bt.Strategy):
    """
    Швидкий скальпінг: купуємо при швидкому зростанні
    Короткотермінова стратегія для швидких прибутків
    """
    
    params = (
        ('fast_ema', 3),
        ('slow_ema', 8),
        ('rsi_period', 7),
        ('scalp_target', 0.02),    # 2% ціль для скальпінгу
        ('quick_stop', 0.015),     # 1.5% швидкий стоп
        ('position_size', 1.0),
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
            # Швидкий сигнал: EMA кросс + RSI
            if (fast_ema > slow_ema and 
                self.fast_ema[-1] <= self.slow_ema[-1] and
                rsi_value > 40 and rsi_value < 65):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'SCALP BUY: Price={current_price:.2f}, RSI={rsi_value:.2f}')
                
        else:
            # Швидкий вихід
            profit_price = self.entry_price * (1 + self.params.scalp_target)
            stop_price = self.entry_price * (1 - self.params.quick_stop)
            
            if (current_price >= profit_price or 
                current_price <= stop_price or
                fast_ema < slow_ema):
                
                self.order = self.close()
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.log(f'SCALP EXIT: Price={current_price:.2f}, PnL={pnl:.2f}%')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class RSIBounceStrategy(bt.Strategy):
    """
    RSI відбиття: купуємо при RSI<40 та зростанні
    Стратегія на відбиття від перепроданих рівнів
    """
    
    params = (
        ('rsi_period', 14),
        ('oversold_level', 35),
        ('overbought_level', 70),
        ('bounce_confirm', 2),     # Кількість барів для підтвердження відбиття
        ('position_size', 0.80),
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.order = None
        self.entry_price = 0
        self.bounce_count = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        rsi_value = self.rsi[0]
        
        if not self.position:
            # Перевіряємо відбиття від перепроданості
            if rsi_value < self.params.oversold_level:
                self.bounce_count = 0
            elif rsi_value > self.rsi[-1] and self.bounce_count < self.params.bounce_confirm:
                self.bounce_count += 1
            
            # Сигнал на покупку: RSI відбивається від перепроданості
            if (self.bounce_count >= self.params.bounce_confirm and
                rsi_value > self.params.oversold_level and
                rsi_value < 50):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.bounce_count = 0
                self.log(f'RSI BOUNCE BUY: Price={current_price:.2f}, RSI={rsi_value:.2f}')
                
        else:
            # Вихід при перекупленості або розвороті RSI
            if (rsi_value > self.params.overbought_level or
                (rsi_value < self.rsi[-1] and rsi_value > 60)):
                
                self.order = self.close()
                self.log(f'RSI EXIT: Price={current_price:.2f}, RSI={rsi_value:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class VolumeSpreadStrategy(bt.Strategy):
    """
    Об'ємний спред: купуємо при високому об'ємі
    Використовує об'єм та спред для визначення сили руху
    """
    
    params = (
        ('volume_period', 20),
        ('volume_threshold', 1.5), # В 1.5 рази вище середнього об'єму
        ('spread_threshold', 0.005), # 0.5% мінімальний спред
        ('position_size', 0.75),
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)
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
        current_volume = self.data.volume[0] if len(self.data.volume) > 0 else 1
        avg_volume = self.volume_sma[0] if self.volume_sma[0] > 0 else 1
        
        # Розрахунок спреду
        high_low_spread = (self.data.high[0] - self.data.low[0]) / current_price
        
        if not self.position:
            # Сигнал: високий об'єм + достатній спред + зростання
            if (current_volume > avg_volume * self.params.volume_threshold and
                high_low_spread > self.params.spread_threshold and
                current_price > self.data.open[0]):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'VOLUME BUY: Price={current_price:.2f}, Vol={current_volume:.0f}, Spread={high_low_spread:.3f}')
                
        else:
            # Вихід при зниженні об'єму або ціни
            if (current_volume < avg_volume * 0.8 or
                current_price < self.data.low[-1]):
                
                self.order = self.close()
                self.log(f'VOLUME EXIT: Price={current_price:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class PriceActionStrategy(bt.Strategy):
    """
    Прайс екшн: купуємо при бичачих свічках
    Аналізує формації свічок для торгових сигналів
    """
    
    params = (
        ('min_body_ratio', 0.6),   # Мінімальний розмір тіла свічки
        ('shadow_ratio', 0.3),     # Максимальний розмір тіні
        ('confirmation_bars', 1),   # Кількість барів для підтвердження
        ('position_size', 0.85),
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.order = None
        self.entry_price = 0
        self.bullish_count = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def is_bullish_candle(self, index=0):
        """Перевіряє чи є свічка бичачою"""
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
        
        if not self.position:
            # Підрахунок бичачих свічок
            if self.is_bullish_candle():
                self.bullish_count += 1
            else:
                self.bullish_count = 0
            
            # Сигнал на покупку: достатньо бичячих свічок підряд
            if (self.bullish_count >= self.params.confirmation_bars and
                current_price > self.data.close[-1]):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.bullish_count = 0
                self.log(f'PRICE ACTION BUY: Price={current_price:.2f}')
                
        else:
            # Вихід при ведмежій свічці
            if (self.data.close[0] < self.data.open[0] and
                not self.is_bullish_candle()):
                
                self.order = self.close()
                self.log(f'PRICE ACTION EXIT: Price={current_price:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class SimpleStrategy(bt.Strategy):
    """
    Найпростіша: купуємо при зростанні ціни
    Базова стратегія для початківців
    """
    
    params = (
        ('lookback_period', 3),    # Період для перевірки зростання
        ('min_growth', 1.01),      # Мінімальне зростання 1%
        ('position_size', 1.0),
        ('max_hold_days', 5),      # Максимальний час утримання позиції  
        ('log_enabled', False),
    )
    
    def __init__(self):
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
            # Перевіряємо зростання за останні бари
            if len(self.data.close) >= self.params.lookback_period:
                old_price = self.data.close[-self.params.lookback_period]
                growth_ratio = current_price / old_price
                
                if growth_ratio >= self.params.min_growth:
                    cash_amount = self.broker.getcash() * self.params.position_size
                    size = max(cash_amount / current_price, 0.001)
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.entry_bar = len(self.data)
                    self.log(f'SIMPLE BUY: Price={current_price:.2f}, Growth={growth_ratio:.3f}')
                    
        else:
            # Вихід при падінні або максимальному часі утримання
            bars_held = len(self.data) - self.entry_bar
            
            if (current_price < self.data.close[-1] or 
                bars_held >= self.params.max_hold_days):
                
                self.order = self.close()
                self.log(f'SIMPLE EXIT: Price={current_price:.2f}, Held={bars_held} bars')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MACDStrategy(bt.Strategy):
    """
    MACD стратегія: купуємо при перетині MACD вгору
    Використовує MACD індикатор для генерації торгових сигналів
    """
    
    params = (
        ('fast_period', 12),
        ('slow_period', 26),
        ('signal_period', 9),
        ('position_size', 0.90),
        ('stop_loss', 0.06),
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.macd = bt.indicators.MACDHisto(
            self.data.close,
            period_me1=self.params.fast_period,
            period_me2=self.params.slow_period,
            period_signal=self.params.signal_period
        )
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
        macd_line = self.macd.macd[0]
        signal_line = self.macd.signal[0]
        histogram = self.macd.histo[0]
        
        if not self.position:
            # Сигнал на покупку: MACD перетинає сигнальну лінію вгору
            if (macd_line > signal_line and 
                self.macd.macd[-1] <= self.macd.signal[-1] and
                histogram > 0):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'MACD BUY: Price={current_price:.2f}, MACD={macd_line:.4f}')
                
        else:
            # Вихід при зміні тренду або стоп-лосс
            stop_price = self.entry_price * (1 - self.params.stop_loss)
            
            if (macd_line < signal_line or 
                current_price <= stop_price or
                histogram < 0):
                
                self.order = self.close()
                self.log(f'MACD SELL: Price={current_price:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class BollingerBandsStrategy(bt.Strategy):
    """
    Bollinger Bands стратегія: купуємо при відбитті від нижньої смуги
    Використовує смуги Боллінжера для визначення зон перекупленості/перепроданості
    """
    
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('position_size', 0.85),
        ('squeeze_threshold', 0.02),  # Поріг для визначення стиснення смуг
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.bb = bt.indicators.BollingerBands(
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )
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
        bb_top = self.bb.lines.top[0]
        bb_mid = self.bb.lines.mid[0]
        bb_bot = self.bb.lines.bot[0]
        
        # Розрахунок ширини смуг
        bb_width = (bb_top - bb_bot) / bb_mid
        
        if not self.position:
            # Сигнал на покупку: ціна торкається нижньої смуги та відбивається
            if (current_price <= bb_bot and 
                current_price > self.data.close[-1] and
                bb_width > self.params.squeeze_threshold):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BB BUY: Price={current_price:.2f}, Lower={bb_bot:.2f}')
                
        else:
            # Вихід при досягненні середньої лінії або верхньої смуги
            if (current_price >= bb_mid or 
                current_price >= bb_top * 0.98):
                
                self.order = self.close()
                self.log(f'BB SELL: Price={current_price:.2f}, Target reached')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class StochasticStrategy(bt.Strategy):
    """
    Стохастична стратегія: купуємо при виході з перепроданості
    Використовує стохастичний осцилятор для визначення моментів входу
    """
    
    params = (
        ('stoch_period', 14),
        ('stoch_period_dfast', 3),
        ('oversold_level', 20),
        ('overbought_level', 80),
        ('position_size', 0.80),
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.stoch = bt.indicators.Stochastic(
            period=self.params.stoch_period,
            period_dfast=self.params.stoch_period_dfast
        )
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
        stoch_k = self.stoch.percK[0]
        stoch_d = self.stoch.percD[0]
        
        if not self.position:
            # Сигнал на покупку: стохастик виходить з зони перепроданості
            if (stoch_k < self.params.oversold_level and 
                stoch_d < self.params.oversold_level and
                stoch_k > stoch_d and
                stoch_k > self.stoch.percK[-1]):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'STOCH BUY: Price={current_price:.2f}, K={stoch_k:.1f}, D={stoch_d:.1f}')
                
        else:
            # Вихід при досягненні зони перекупленості
            if (stoch_k > self.params.overbought_level and 
                stoch_d > self.params.overbought_level and
                stoch_k < stoch_d):
                
                self.order = self.close()
                self.log(f'STOCH SELL: Price={current_price:.2f}, K={stoch_k:.1f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MeanReversionStrategy(bt.Strategy):
    """
    Стратегія повернення до середнього: купуємо при відхиленні від SMA
    Торгує на відновленні ціни до середнього значення
    """
    
    params = (
        ('sma_period', 20),
        ('deviation_threshold', 0.05),  # 5% відхилення від SMA
        ('position_size', 0.75),
        ('max_positions', 3),           # Максимальна кількість відкритих позицій
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.params.sma_period)
        self.order = None
        self.entry_price = 0
        self.position_count = 0
        
    def log(self, txt, dt=None):
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}: {txt}')
    
    def next(self):
        if self.order:
            return
            
        current_price = self.data.close[0]
        sma_value = self.sma[0]
        deviation = (current_price - sma_value) / sma_value
        
        if not self.position:
            # Сигнал на покупку: ціна значно нижче SMA
            if (deviation < -self.params.deviation_threshold and
                current_price > self.data.close[-1] and
                self.position_count < self.params.max_positions):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.position_count += 1
                self.log(f'MEAN REV BUY: Price={current_price:.2f}, SMA={sma_value:.2f}, Dev={deviation:.3f}')
                
        else:
            # Вихід при поверненні до SMA або вище
            if (current_price >= sma_value or 
                deviation > 0.02):
                
                self.order = self.close()
                self.position_count = max(0, self.position_count - 1)
                self.log(f'MEAN REV SELL: Price={current_price:.2f}, Back to mean')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class DCAStrategy(bt.Strategy):
    """
    DCA (Dollar Cost Averaging) стратегія: регулярне усереднення
    Купує фіксовану суму через регулярні інтервали
    """
    
    params = (
        ('buy_interval', 5),        # Купуємо кожні 5 барів
        ('buy_amount_pct', 0.10),   # 10% від капіталу за раз
        ('trend_filter', True),     # Фільтр тренду
        ('sma_period', 50),
        ('max_positions', 10),
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
        
        # Регулярна покупка
        if self.bar_count % self.params.buy_interval == 0:
            trend_ok = True
            
            if self.params.trend_filter:
                trend_ok = current_price > self.sma[0]
            
            if (trend_ok and 
                self.buy_count < self.params.max_positions):
                
                cash_amount = self.broker.getcash() * self.params.buy_amount_pct
                if cash_amount > current_price:
                    size = max(cash_amount / current_price, 0.001)
                    self.order = self.buy(size=size)
                    self.buy_count += 1
                    self.log(f'DCA BUY #{self.buy_count}: Price={current_price:.2f}, Amount=${cash_amount:.2f}')
        
        # Вихід при сильному зростанні (взяття прибутку)
        if (self.position and 
            current_price > self.sma[0] * 1.20):  # 20% вище SMA
            
            self.order = self.close()
            self.buy_count = 0
            self.log(f'DCA SELL ALL: Price={current_price:.2f}, Strong growth')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class GridTradingStrategy(bt.Strategy):
    """
    Грід-трейдінг стратегія: створює сітку ордерів
    Купує при падінні та продає при зростанні в межах сітки
    """
    
    params = (
        ('grid_size', 0.02),        # 2% між рівнями сітки
        ('grid_levels', 5),         # Кількість рівнів сітки
        ('position_size', 0.20),    # 20% на кожен рівень
        ('base_price_period', 20),  # Період для базової ціни
        ('log_enabled', False),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.params.base_price_period)
        self.order = None
        self.grid_positions = []
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
            sell_level = self.base_price * (1 + self.params.grid_size * level)
            
            # Покупка на рівні сітки
            if (current_price <= buy_level and 
                level not in self.grid_positions and
                len(self.grid_positions) < self.params.grid_levels):
                
                cash_amount = self.broker.getcash() * self.params.position_size
                if cash_amount > current_price:
                    size = max(cash_amount / current_price, 0.001)
                    self.order = self.buy(size=size)
                    self.grid_positions.append(level)
                    self.log(f'GRID BUY L{level}: Price={current_price:.2f}, Target={buy_level:.2f}')
                    break
            
            # Продаж на рівні сітки
            elif (current_price >= sell_level and 
                  self.position and
                  len(self.grid_positions) > 0):
                
                self.order = self.close()
                if self.grid_positions:
                    sold_level = self.grid_positions.pop(0)
                    self.log(f'GRID SELL L{sold_level}: Price={current_price:.2f}, Target={sell_level:.2f}')
                break
        
        # Оновлюємо базову ціну періодично
        if len(self.data) % self.params.base_price_period == 0:
            self.base_price = self.sma[0]
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


# Експорт всіх стратегій
__all__ = [
    'TrendFollowStrategy',
    'MomentumStrategy', 
    'BreakoutStrategy',
    'QuickScalpStrategy',
    'RSIBounceStrategy',
    'VolumeSpreadStrategy',
    'PriceActionStrategy',
    'SimpleStrategy',
    'MACDStrategy',
    'BollingerBandsStrategy',
    'StochasticStrategy',
    'MeanReversionStrategy',
    'DCAStrategy',
    'GridTradingStrategy'
]