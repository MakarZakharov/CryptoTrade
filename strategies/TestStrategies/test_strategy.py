import os
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class SimpleMovingAverageStrategy(bt.Strategy):
    """Проста стратегія на скользящих середніх"""

    params = (
        ('fast_ma', 10),
        ('slow_ma', 20),
        ('position_size', 0.95),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_ma)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_ma)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:  # Prevent division by zero
            return

        if not self.position and self.crossover > 0:
            size = int(self.broker.get_cash() * self.params.position_size / current_price)
            if size > 0:
                self.order = self.buy(size=size)
        elif self.position and self.crossover < 0:
            self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class HighFrequencyTradingStrategy(bt.Strategy):
    """Високочастотна стратегія"""
    params = (
        ('ema_fast', 3), ('ema_slow', 8), ('rsi_period', 5),
        ('rsi_overbought', 60), ('rsi_oversold', 40), ('position_size', 0.8),
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)
        self.ema_cross = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:  # Prevent division by zero
            return

        if not self.position:
            buy_signals = sum([
                self.ema_cross > 0,
                self.rsi[0] < self.params.rsi_oversold
            ])
            if buy_signals >= 1:
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.ema_cross < 0 or self.rsi[0] > self.params.rsi_overbought:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ScalpingMACDStrategy(bt.Strategy):
    """Скальпінгова стратегія з MACD"""
    params = (
        ('macd_fast', 5), ('macd_slow', 13), ('macd_signal', 8),
        ('ema_period', 9), ('position_size', 0.9),
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close, period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow, period_signal=self.params.macd_signal
        )
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_period)
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.macd_cross > 0 and self.data.close[0] > self.ema[0]:
                size = int(self.broker.get_cash() * self.params.position_size / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.macd_cross < 0 or self.data.close[0] < self.ema[0]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class UltraHighFrequencyStrategy(bt.Strategy):
    """Ультра високочастотна стратегія з множинними сигналами - ПОКРАЩЕНА"""
    params = (
        ('ema1', 1), ('ema2', 2), ('ema3', 3),  # Ще швидші EMA (було 2,3,5)
        ('rsi_period', 2), ('rsi_ob', 53), ('rsi_os', 47),  # Ще вужчі рівні RSI (було 55/45)
        ('stoch_period', 2), ('stoch_ob', 72), ('stoch_os', 28),  # Швидший Stochastic
        ('volume_ma', 3),  # Ще коротший період для volume (було 5)
        ('position_size', 0.65),  # Трохи менший розмір (було 0.7)
        ('atr_period', 3),  # Додали ATR для волатильності
    )

    def __init__(self):
        # Максимально швидкі EMA індикатори
        self.ema1 = bt.indicators.EMA(self.data.close, period=self.params.ema1)
        self.ema2 = bt.indicators.EMA(self.data.close, period=self.params.ema2)
        self.ema3 = bt.indicators.EMA(self.data.close, period=self.params.ema3)

        # Швидкі осцилятори
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.stoch = bt.indicators.Stochastic(self.data, period=self.params.stoch_period)

        # Volume та волатильність
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

        # Кросовери
        self.ema_cross_fast = bt.indicators.CrossOver(self.ema1, self.ema2)
        self.ema_cross_med = bt.indicators.CrossOver(self.ema2, self.ema3)

        # Додаткові сигнали
        self.price_above_ema1 = self.data.close > self.ema1
        self.volume_spike = self.data.volume > self.volume_ma * 1.1

        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Гіперагресивні умови для покупки
        if not self.position:
            buy_signals = sum([
                self.ema_cross_fast > 0,  # EMA1 перетинає EMA2 вгору
                self.ema_cross_med > 0,   # EMA2 перетинає EMA3 вгору
                self.rsi[0] < self.params.rsi_os,  # Дуже вузькі RSI рівні
                self.stoch.percK[0] < self.params.stoch_os,  # Швидкий Stoch
                self.volume_spike[0],  # Спайк volume
                self.data.close[0] > self.data.close[-1],  # Зростання ціни
                self.price_above_ema1[0],  # Ціна вище EMA1
                self.atr[0] > self.atr[-1],  # Зростання волатильності
            ])

            # Входимо при 1+ сигналі (максимально агресивно, було 2+)
            if buy_signals >= 1:
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        # Миттєві умови для продажу
        else:
            sell_signals = sum([
                self.ema_cross_fast < 0,  # EMA1 перетинає EMA2 вниз
                self.rsi[0] > self.params.rsi_ob,  # Вузькі RSI рівні
                self.stoch.percK[0] > self.params.stoch_ob,  # Stoch високий
                self.data.close[0] < self.data.close[-1],  # Падіння ціни
                not self.price_above_ema1[0],  # Ціна нижче EMA1
            ])

            # Виходимо при першому ж негативному сигналі (було 1+, залишаємо)
            if sell_signals >= 1:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class HyperFrequencyStrategy(bt.Strategy):
    """Гіпер високочастотна стратегія з мінімальними періодами"""
    params = (
        ('ema1', 1), ('ema2', 2), ('ema3', 3),  # Максимально швидкі EMA
        ('rsi_period', 2), ('rsi_ob', 52), ('rsi_os', 48),  # Дуже швидкий RSI з дуже вузькими рівнями
        ('stoch_period', 2), ('stoch_ob', 70), ('stoch_os', 30),  # Дуже швидкий Stochastic
        ('williams_period', 2), ('williams_ob', -20), ('williams_os', -80),  # Williams %R
        ('cci_period', 2), ('cci_ob', 50), ('cci_os', -50),  # Швидкий CCI
        ('volume_ma', 3),  # Дуже короткий період для volume
        ('momentum_period', 1),  # Momentum з періодом 1
        ('position_size', 0.6),  # Менший розмір через дуже високий ризик
    )

    def __init__(self):
        # Максимально швидкі EMA індикатори
        self.ema1 = bt.indicators.EMA(self.data.close, period=self.params.ema1)
        self.ema2 = bt.indicators.EMA(self.data.close, period=self.params.ema2)
        self.ema3 = bt.indicators.EMA(self.data.close, period=self.params.ema3)

        # Швидкі осцилятори
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.stoch = bt.indicators.Stochastic(self.data, period=self.params.stoch_period)
        self.williams = bt.indicators.WilliamsR(self.data, period=self.params.williams_period)
        self.cci = bt.indicators.CommodityChannelIndex(self.data, period=self.params.cci_period)

        # Volume та momentum індикатори
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma)
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)

        # Кросовери
        self.ema_cross_1_2 = bt.indicators.CrossOver(self.ema1, self.ema2)
        self.ema_cross_2_3 = bt.indicators.CrossOver(self.ema2, self.ema3)

        # Ціновий momentum
        self.price_change = self.data.close - self.data.close(-1)

        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Гіперагресивні умови для покупки
        if not self.position:
            buy_signals = sum([
                self.ema_cross_1_2 > 0,  # EMA1 перетинає EMA2 вгору
                self.ema_cross_2_3 > 0,  # EMA2 перетинає EMA3 вгору
                self.rsi[0] < self.params.rsi_os,  # Дуже вузькі RSI рівні
                self.stoch.percK[0] < self.params.stoch_os,  # Швидкий Stoch
                self.williams[0] < self.params.williams_os,  # Williams %R
                self.cci[0] < self.params.cci_os,  # CCI
                self.data.volume[0] > self.volume_ma[0] * 1.1,  # Невеликий приріст volume
                self.price_change[0] > 0,  # Позитивна зміна ціни
                self.momentum[0] > 0,  # Позитивний momentum
                self.data.close[0] > self.ema1[0],  # Ціна вище найшвидшої EMA
            ])

            # Вхід при 1+ сигналі (максимально агресивно)
            if buy_signals >= 1:
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        # Миттєві умови для продажу
        else:
            sell_signals = sum([
                self.ema_cross_1_2 < 0,  # EMA1 перетинає EMA2 вниз
                self.rsi[0] > self.params.rsi_ob,  # Вузькі RSI рівні
                self.stoch.percK[0] > self.params.stoch_ob,  # Stoch
                self.williams[0] > self.params.williams_ob,  # Williams %R
                self.cci[0] > self.params.cci_ob,  # CCI
                self.price_change[0] < 0,  # Негативна зміна ціни
                self.momentum[0] < 0,  # Негативний momentum
                self.data.close[0] < self.ema1[0],  # Ціна нижче найшвидшої EMA
            ])

            # Вихід при першому ж негативному сигналі
            if sell_signals >= 1:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ScalpingTickStrategy(bt.Strategy):
    """Тік-скальпінгова стратегія для максимальної частоти"""
    params = (
        ('price_threshold', 0.001),  # Мінімальна зміна ціни для сигналу (0.1%)
        ('volume_spike', 1.05),  # Мінімальний спайк volume (5%)
        ('position_size', 0.5),  # Малий розмір через дуже високий ризик
        ('stop_loss', 0.005),  # 0.5% стоп-лос
        ('take_profit', 0.01),  # 1% тейк-профіт
    )

    def __init__(self):
        # Мінімальні індикатори для швидкості
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=2)
        self.price_change_pct = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.volume_change = self.data.volume / self.volume_ma

        self.order = None
        self.entry_price = 0

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Тік-скальпінг: реакція на мінімальні зміни
        if not self.position:
            # Вхід на малих рухах з високим volume
            if (abs(self.price_change_pct[0]) > self.params.price_threshold and
                self.volume_change[0] > self.params.volume_spike and
                self.price_change_pct[0] > 0):  # Тільки на зростанні

                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price

        else:
            # Швидкий вихід по стоп-лосу або тейк-профіту
            profit_loss_pct = (current_price - self.entry_price) / self.entry_price

            if (profit_loss_pct <= -self.params.stop_loss or  # Стоп-лос
                profit_loss_pct >= self.params.take_profit or  # Тейк-профіт
                self.price_change_pct[0] < -self.params.price_threshold):  # Негативна зміна
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class NanoFrequencyStrategy(bt.Strategy):
    """Нано-частотна стратегія з максимальною чутливістю"""
    params = (
        ('price_threshold', 0.0001),  # 0.01% мінімальна зміна
        ('volume_threshold', 1.01),   # 1% збільшення volume
        ('position_size', 0.3),       # Малий розмір через високий ризик
        ('max_hold_periods', 3),      # Максимум 3 періоди утримання
    )

    def __init__(self):
        self.price_change = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.volume_change = self.data.volume / self.data.volume(-1)
        self.hold_periods = 0
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # Вхід на найменших змінах
            if (abs(self.price_change[0]) > self.params.price_threshold and
                self.volume_change[0] > self.params.volume_threshold and
                self.price_change[0] > 0):

                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
                    self.hold_periods = 0
        else:
            self.hold_periods += 1
            # Вихід через максимальний час утримання або негативну зміну
            if (self.hold_periods >= self.params.max_hold_periods or
                self.price_change[0] < -self.params.price_threshold):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MultiSignalHFTStrategy(bt.Strategy):
    """Мультисигнальна HFT стратегія з 15+ індикаторами"""
    params = (
        ('ema_ultra', 1), ('ema_fast', 2), ('ema_med', 3),
        ('rsi_period', 2), ('rsi_neutral', 50),
        ('stoch_period', 2), ('williams_period', 2), ('cci_period', 2),
        ('momentum_period', 1), ('roc_period', 1), ('trix_period', 3),
        ('volume_sma', 2), ('atr_period', 2), ('adx_period', 3),
        ('position_size', 0.4), ('signal_threshold', 3),
    )

    def __init__(self):
        # Множина індикаторів для максимальної точності
        self.ema_ultra = bt.indicators.EMA(self.data.close, period=self.params.ema_ultra)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_med = bt.indicators.EMA(self.data.close, period=self.params.ema_med)

        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.stoch = bt.indicators.Stochastic(self.data, period=self.params.stoch_period)
        self.williams = bt.indicators.WilliamsR(self.data, period=self.params.williams_period)
        self.cci = bt.indicators.CCI(self.data, period=self.params.cci_period)

        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.roc = bt.indicators.ROC(self.data.close, period=self.params.roc_period)
        self.trix = bt.indicators.TRIX(self.data.close, period=self.params.trix_period)

        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_sma)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)

        # Кросовери
        self.ema_cross_ultra_fast = bt.indicators.CrossOver(self.ema_ultra, self.ema_fast)
        self.ema_cross_fast_med = bt.indicators.CrossOver(self.ema_fast, self.ema_med)

        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # 15 різних бичачих сигналів
            buy_signals = sum([
                self.ema_cross_ultra_fast > 0,                    # EMA кросовер
                self.ema_cross_fast_med > 0,                      # EMA кросовер 2
                self.data.close[0] > self.ema_ultra[0],           # Ціна > Ultra EMA
                self.rsi[0] < self.params.rsi_neutral,            # RSI нейтральний
                self.stoch.percK[0] < 50,                         # Stoch < 50
                self.williams[0] < -50,                           # Williams < -50
                self.cci[0] < 0,                                  # CCI < 0
                self.momentum[0] > 0,                             # Позитивний momentum
                self.roc[0] > 0,                                  # Позитивний ROC
                self.trix[0] > self.trix[-1],                     # TRIX зростає
                self.data.volume[0] > self.volume_sma[0],         # Volume > SMA
                self.atr[0] > self.atr[-1],                       # ATR зростає
                self.adx[0] > 20,                                 # ADX > 20 (тренд)
                self.data.close[0] > self.data.close[-1],         # Зростання ціни
                self.data.high[0] > self.data.high[-1],           # Новий максимум
            ])

            if buy_signals >= self.params.signal_threshold:
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        else:
            # Швидкий вихід при негативних сигналах
            sell_signals = sum([
                self.ema_cross_ultra_fast < 0,
                self.data.close[0] < self.ema_ultra[0],
                self.momentum[0] < 0,
                self.roc[0] < 0,
                self.data.close[0] < self.data.close[-1],
            ])

            if sell_signals >= 1:  # Вихід при першому негативному сигналі
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class PriceActionScalpingStrategy(bt.Strategy):
    """Скальпінг на price action з мінімальними затримками"""
    params = (
        ('min_body_size', 0.0005),      # Мінімальний розмір тіла свічки (0.05%)
        ('wick_ratio', 0.3),            # Співвідношення фітиля до тіла
        ('volume_spike', 1.05),         # Спайк volume 5%
        ('position_size', 0.35),        # Консервативний розмір
        ('consecutive_candles', 2),      # Кількість послідовних свічок
    )

    def __init__(self):
        self.body_size = abs(self.data.close - self.data.open) / self.data.open
        self.upper_wick = self.data.high - bt.Max(self.data.open, self.data.close)
        self.lower_wick = bt.Min(self.data.open, self.data.close) - self.data.low
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=3)

        self.consecutive_green = 0
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Визначення зеленої свічки
        is_green = self.data.close[0] > self.data.open[0]
        if is_green:
            self.consecutive_green += 1
        else:
            self.consecutive_green = 0

        if not self.position:
            # Price action сигнали
            conditions = [
                self.body_size[0] > self.params.min_body_size,                    # Достатній розмір тіла
                is_green,                                                         # Зелена свічка
                self.lower_wick[0] < self.body_size[0] * self.params.wick_ratio, # Малий нижній фітиль
                self.data.volume[0] > self.volume_ma[0] * self.params.volume_spike, # Volume спайк
                self.consecutive_green >= self.params.consecutive_candles,        # Послідовні зелені свічки
                self.data.close[0] > self.data.high[-1],                         # Пробій попереднього максимуму
            ]

            if sum(conditions) >= 3:  # Мінімум 3 умови
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        else:
            # Вихід при червоній свічці або великому фітилі
            if (not is_green or
                self.upper_wick[0] > self.body_size[0] * 2 or
                self.data.close[0] < self.data.low[-1]):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class VolatilityBreakoutStrategy(bt.Strategy):
    """Стратегія на пробої волатильності"""
    params = (
        ('atr_period', 2), ('atr_multiplier', 0.5),
        ('volume_period', 2), ('volume_multiplier', 1.2),
        ('position_size', 0.4), ('trailing_stop', 0.003),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)
        self.highest = bt.indicators.Highest(self.data.high, period=self.params.atr_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.params.atr_period)

        self.order = None
        self.entry_price = 0
        self.trailing_price = 0

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # Пробій волатільності
            breakout_level = self.highest[-1] + self.atr[0] * self.params.atr_multiplier
            volume_condition = self.data.volume[0] > self.volume_ma[0] * self.params.volume_multiplier

            if (self.data.close[0] > breakout_level and volume_condition):
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.trailing_price = current_price

        else:
            # Trailing stop
            if current_price > self.trailing_price:
                self.trailing_price = current_price

            stop_price = self.trailing_price * (1 - self.params.trailing_stop)

            if (current_price < stop_price or
                current_price < self.lowest[-1]):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ExtremeFrequencyStrategy(bt.Strategy):
    """Екстремально високочастотна стратегія"""
    params = (
        ('ema_period', 1),      # Найшвидша EMA
        ('momentum_period', 1), # Миттєвий momentum
        ('volume_ma', 2),       # Дуже швидкий volume MA
        ('price_change_threshold', 0.0001),  # 0.01% зміна ціни
        ('position_size', 0.4), # Малий розмір через екстремальний ризик
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_period)
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma)
        self.price_change = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Вхід на найменших змінах
        if not self.position:
            buy_conditions = [
                abs(self.price_change[0]) > self.params.price_change_threshold,
                self.data.close[0] > self.ema[0],
                self.momentum[0] > 0,
                self.data.volume[0] > self.volume_ma[0],
                self.price_change[0] > 0  # Тільки на зростанні
            ]

            if sum(buy_conditions) >= 2:  # Мінімум 2 умови
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        # Миттєвий вихід
        else:
            exit_conditions = [
                self.data.close[0] < self.ema[0],
                self.momentum[0] < 0,
                self.price_change[0] < -self.params.price_change_threshold
            ]

            if any(exit_conditions):  # Вихід при першому негативному сигналі
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class CSVBacktester:
    """Простий бектестер для CSV файлів"""

    def __init__(self, csv_file: str, initial_cash: float = 100000, commission: float = 0.001, verbose: bool = True):
        # Покращена логіка визначення шляху
        if not os.path.isabs(csv_file):
            # Спробуємо різні варіанти розташування проекту
            possible_roots = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),  # Від test_strategy.py
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),   # Альтернативний варіант
                os.getcwd(),  # Поточна директорія
            ]

            found = False
            for root in possible_roots:
                full_path = os.path.join(root, csv_file)
                if os.path.exists(full_path):
                    csv_file = full_path
                    found = True
                    break

            if not found:
                # Якщо не знайшли, залишаємо оригінальний шлях
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
                csv_file = os.path.join(project_root, csv_file)

        self.csv_file = csv_file
        self.initial_cash = initial_cash
        self.commission = commission
        self.verbose = verbose

    def load_data(self) -> pd.DataFrame:
        """Завантаження та підготовка даних з CSV"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Файл {self.csv_file} не знайдено")

        try:
            df = pd.read_csv(self.csv_file)

            # Пошук колонки з датою
            date_col = None

            for col in df.columns:
                if any(word in col.lower() for word in ['date', 'time', 'timestamp']):
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.drop_duplicates(subset=[date_col], keep='last')
                df.set_index(date_col, inplace=True)
            else:
                df.set_index(df.columns[0], inplace=True)

            # Стандартизація колонок
            df.columns = df.columns.str.lower().str.strip()
            mapping = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
            df = df.rename(columns=mapping)

            # Перевірка необхідних колонок
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Відсутні колонки: {[col for col in required if col not in df.columns]}")

            if 'volume' not in df.columns:
                df['volume'] = 1000

            # Очищення даних
            df = df[required + ['volume']].dropna()
            df = df[(df > 0).all(axis=1)]
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)

            if len(df) == 0:
                raise ValueError("Після очищення не залишилося даних")

            if self.verbose:
                print(f"✅ Завантажено {len(df)} записів")
            return df

        except Exception as e:
            raise ValueError(f"Помилка завантаження CSV: {str(e)}")

    def run_backtest(self, strategy_class=SimpleMovingAverageStrategy, **strategy_params):
        """Запуск бектестування"""
        data = self.load_data()

        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=data))
        cerebro.addstrategy(strategy_class, **strategy_params)
        cerebro.broker.set_cash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        final_value = cerebro.broker.get_value()

        # Отримання статистики
        trade_analysis = results[0].analyzers.trades.get_analysis()
        total_trades = getattr(getattr(trade_analysis, 'total', None), 'total', 0)
        won_trades = getattr(getattr(trade_analysis, 'won', None), 'total', 0)

        # Prevent division by zero in return calculation
        if self.initial_cash <= 0:
            return_pct = 0.0
        else:
            return_pct = ((final_value - self.initial_cash) / self.initial_cash) * 100

        result = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'profit_loss': final_value - self.initial_cash,
            'return_pct': return_pct,
            'total_trades': total_trades,
            'won_trades': won_trades
        }

        if self.verbose:
            print(f"💰 P&L: ${result['profit_loss']:+,.2f} ({result['return_pct']:+.2f}%)")
            print(f"🔄 Угоди: {total_trades} (виграші: {won_trades})")

        return result


def main():
    """Головна функція для тестування стратегій"""
    CSV_FILE = "CryptoTrade/data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    print("🚀 ТЕСТУВАННЯ ВИСОКОЧАСТОТНИХ СТРАТЕГІЙ")
    print("=" * 60)

    strategies_to_test = [
        {
            'name': '📊 Simple Moving Average',
            'class': SimpleMovingAverageStrategy,
            'params': {
                'fast_ma': 10, 'slow_ma': 20, 'position_size': 0.95
            }
        },
        {
            'name': '⚡ High Frequency Trading',
            'class': HighFrequencyTradingStrategy,
            'params': {
                'ema_fast': 3, 'ema_slow': 8, 'rsi_period': 5,
                'rsi_overbought': 60, 'rsi_oversold': 40, 'position_size': 0.8
            }
        },
        {
            'name': '🎯 Scalping MACD',
            'class': ScalpingMACDStrategy,
            'params': {
                'macd_fast': 5, 'macd_slow': 13, 'macd_signal': 8,
                'ema_period': 9, 'position_size': 0.9
            }
        },
        {
            'name': '🚀 Ultra High Frequency',
            'class': UltraHighFrequencyStrategy,
            'params': {
                'ema1': 1, 'ema2': 2, 'ema3': 3,
                'rsi_period': 2, 'rsi_ob': 53, 'rsi_os': 47,
                'position_size': 0.65
            }
        },
        {
            'name': '🔥 Hyper Frequency',
            'class': HyperFrequencyStrategy,
            'params': {
                'ema1': 1, 'ema2': 2, 'ema3': 3,
                'rsi_period': 2, 'rsi_ob': 52, 'rsi_os': 48,
                'position_size': 0.6
            }
        },
        {
            'name': '⭐ Nano Frequency',
            'class': NanoFrequencyStrategy,
            'params': {
                'price_threshold': 0.0001, 'volume_threshold': 1.01,
                'position_size': 0.3, 'max_hold_periods': 3
            }
        },
        {
            'name': '🎪 Multi-Signal HFT',
            'class': MultiSignalHFTStrategy,
            'params': {
                'ema_ultra': 1, 'ema_fast': 2, 'ema_med': 3,
                'signal_threshold': 3, 'position_size': 0.4
            }
        },
        {
            'name': '📈 Price Action Scalping',
            'class': PriceActionScalpingStrategy,
            'params': {
                'min_body_size': 0.0005, 'volume_spike': 1.05,
                'position_size': 0.35, 'consecutive_candles': 2
            }
        },
        {
            'name': '💥 Volatility Breakout',
            'class': VolatilityBreakoutStrategy,
            'params': {
                'atr_period': 2, 'atr_multiplier': 0.5,
                'position_size': 0.4, 'trailing_stop': 0.003
            }
        },
        {
            'name': '⚡ Extreme Frequency',
            'class': ExtremeFrequencyStrategy,
            'params': {
                'ema_period': 1, 'momentum_period': 1,
                'price_change_threshold': 0.0001, 'position_size': 0.4
            }
        }
    ]

    best_result = None
    best_score = 0
    all_results = []

    try:
        backtester = CSVBacktester(csv_file=CSV_FILE, initial_cash=100000, commission=0.001)

        for strategy in strategies_to_test:
            print(f"\n🔥 ТЕСТУВАННЯ: {strategy['name']}")
            print("-" * 50)

            try:
                result = backtester.run_backtest(
                    strategy_class=strategy['class'],
                    **strategy['params']
                )

                win_rate = (result['won_trades']/max(result['total_trades'],1)*100)
                profit_per_trade = result['return_pct'] / max(result['total_trades'], 1)

                # Комплексна оцінка
                frequency_bonus = min(2.0, result['total_trades'] / 100)
                complex_score = (result['return_pct'] * 0.4 +          # 40% - прибутковість
                               result['total_trades'] * 0.3 +          # 30% - частота торгівлі
                               win_rate * 0.2 +                        # 20% - процент виграшів
                               frequency_bonus * 10)                   # 10% - бонус за частоту

                strategy_result = {
                    'name': strategy['name'],
                    'result': result,
                    'params': strategy['params'],
                    'win_rate': win_rate,
                    'profit_per_trade': profit_per_trade,
                    'complex_score': complex_score
                }
                all_results.append(strategy_result)

                print(f"📊 Трейдів: {result['total_trades']}")
                print(f"📈 Win Rate: {win_rate:.1f}%")
                print(f"💰 Прибуток: {result['return_pct']:+.2f}%")
                print(f"⚡ Прибуток/трейд: {profit_per_trade:.3f}%")
                print(f"🎯 Комплексний бал: {complex_score:.2f}")

                if complex_score > best_score:
                    best_score = complex_score
                    best_result = strategy_result

            except Exception as e:
                print(f"❌ Помилка в {strategy['name']}: {e}")

        # Виведення результатів
        if best_result:
            print(f"\n🏆 НАЙКРАЩА СТРАТЕГІЯ:")
            print("=" * 60)
            print(f"📛 Назва: {best_result['name']}")
            print(f"🔄 Трейдів: {best_result['result']['total_trades']}")
            print(f"📈 Win Rate: {best_result['win_rate']:.1f}%")
            print(f"💰 Прибуток: {best_result['result']['return_pct']:+.2f}%")
            print(f"⚡ Прибуток/трейд: {best_result['profit_per_trade']:.3f}%")
            print(f"🎯 Комплексний бал: {best_result['complex_score']:.2f}")
            print(f"⚙️ Параметри: {best_result['params']}")

        # ТОП-3 стратегії
        all_results.sort(key=lambda x: x['complex_score'], reverse=True)
        print(f"\n📊 ТОП-3 СТРАТЕГІЙ:")
        print("-" * 40)
        for i, strategy in enumerate(all_results[:3], 1):
            print(f"{i}. {strategy['name']}: {strategy['complex_score']:.2f} балів")
            print(f"   Прибуток: {strategy['result']['return_pct']:+.2f}% | Трейдів: {strategy['result']['total_trades']}")

        print("\n✅ ТЕСТУВАННЯ ЗАВЕРШЕНО!")

    except Exception as e:
        print(f"❌ Критична помилка: {e}")


if __name__ == "__main__":
    main()


class OptimizedBTCAnalysisStrategy(bt.Strategy):
    """Оптимізована стратегія на основі аналізу історичних даних BTC 2018-2025"""
    params = (
        # Адаптивні параметри для різних ринкових циклів
        ('ema_ultra', 1), ('ema_fast', 2), ('ema_medium', 5), ('ema_slow', 13),
        ('rsi_period', 3), ('rsi_oversold', 25), ('rsi_overbought', 75), # Ширші рівні для BTC
        ('bb_period', 10), ('bb_std', 2.2), # Болінгер з більшим стандартним відхиленням
        ('volume_short', 3), ('volume_long', 10), ('volume_spike_ratio', 2.5),
        ('atr_period', 7), ('atr_multiplier', 1.8),
        # Спеціальні BTC паттерни
        ('gap_threshold', 0.03), # 3% гепи характерні для BTC
        ('momentum_period', 2), ('roc_period', 3),
        ('volatility_window', 7), ('high_vol_threshold', 0.05), # 5% денна волатильність
        # Управління ризиками
        ('position_size', 0.45), ('max_dd_percent', 0.08), # 8% максимальна просадка
        ('profit_target', 0.04), ('stop_loss', 0.025), # 4% профіт, 2.5% стоп
        # Фільтри ринкових режимів
        ('bull_trend_days', 5), ('bear_trend_days', 7),
        ('sideways_volatility', 0.02), # 2% для бокового руху
    )

    def __init__(self):
        # Трендові індикатори з різними швидкостями
        self.ema_ultra = bt.indicators.EMA(self.data.close, period=self.params.ema_ultra)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_medium = bt.indicators.EMA(self.data.close, period=self.params.ema_medium)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)

        # Осцилятори оптимізовані для BTC
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.rsi_ema = bt.indicators.EMA(self.rsi, period=3) # Згладжений RSI

        # Bollinger Bands для волатильності
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.params.bb_period, devfactor=self.params.bb_std
        )
        self.bb_percent = (self.data.close - self.bb.bot) / (self.bb.top - self.bb.bot)

        # Volume індикатори
        self.volume_short = bt.indicators.SMA(self.data.volume, period=self.params.volume_short)
        self.volume_long = bt.indicators.SMA(self.data.volume, period=self.params.volume_long)
        self.volume_ratio = self.data.volume / self.volume_long

        # Волатільність та momentum
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.roc = bt.indicators.ROC(self.data.close, period=self.params.roc_period)

        # Ринкові режими
        self.daily_return = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.volatility = bt.indicators.StdDev(self.daily_return, period=self.params.volatility_window)

        # Спеціальні BTC паттерни
        self.gap_size = abs(self.data.open - self.data.close(-1)) / self.data.close(-1)
        self.body_size = abs(self.data.close - self.data.open) / self.data.open
        self.upper_shadow = (self.data.high - bt.Max(self.data.open, self.data.close)) / self.data.open
        self.lower_shadow = (bt.Min(self.data.open, self.data.close) - self.data.low) / self.data.open

        # Кросовери для сигналів
        self.ema_cross_ultra = bt.indicators.CrossOver(self.ema_ultra, self.ema_fast)
        self.ema_cross_fast = bt.indicators.CrossOver(self.ema_fast, self.ema_medium)
        self.ema_alignment = (self.ema_ultra > self.ema_fast) and (self.ema_fast > self.ema_medium)

        # Трекінг позиції
        self.order = None
        self.entry_price = 0
        self.entry_bar = 0
        self.highest_price = 0
        self.market_regime = 'sideways'

    def determine_market_regime(self):
        """Визначення поточного ринкового режиму на основі BTC специфіки"""
        # Трендова сила
        price_trend = (self.ema_fast[0] - self.ema_slow[0]) / self.ema_slow[0]
        volatility_level = self.volatility[0]
        volume_strength = self.volume_ratio[0]

        # BTC специфічні умови
        if (price_trend > 0.02 and  # 2%+ тренд
            volatility_level > self.params.high_vol_threshold and
            volume_strength > 1.5):
            return 'strong_bull'
        elif (price_trend > 0.005 and  # 0.5%+ слабкий тренд
              self.ema_alignment):
            return 'bull'
        elif (price_trend < -0.02 and  # -2% сильний ведмежий тренд
              volatility_level > self.params.high_vol_threshold):
            return 'strong_bear'
        elif price_trend < -0.005:  # Слабкий ведмежий тренд
            return 'bear'
        else:
            return 'sideways'

    def calculate_dynamic_position_size(self, regime, volatility):
        """Динамічний розмір позиції залежно від ринкових умов"""
        base_size = self.params.position_size

        # Коригування на волатільність
        vol_adjustment = max(0.5, min(1.5, 0.03 / max(volatility, 0.01)))

        # Коригування на ринковий режим
        regime_multipliers = {
            'strong_bull': 1.2,
            'bull': 1.0,
            'sideways': 0.7,
            'bear': 0.4,
            'strong_bear': 0.3
        }

        return base_size * vol_adjustment * regime_multipliers.get(regime, 0.7)

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Визначення ринкового режиму
        self.market_regime = self.determine_market_regime()
        current_volatility = self.volatility[0]

        if not self.position:
            # Комплексна система сигналів для входу

            # 1. Трендові сигнали (вага 30%)
            trend_signals = sum([
                self.ema_cross_ultra > 0,  # Ультрашвидкий кросовер
                self.ema_cross_fast > 0,   # Швидкий кросовер
                self.data.close[0] > self.ema_fast[0],  # Ціна вище швидкої EMA
                self.ema_alignment,  # Вирівнювання EMA
                self.momentum[0] > 0,  # Позитивний momentum
            ])

            # 2. Осциляторні сигнали (вага 25%)
            oscillator_signals = sum([
                self.rsi[0] < self.params.rsi_oversold,  # RSI перепроданість
                self.rsi_ema[0] > self.rsi_ema[-1],  # RSI тренд вгору
                self.bb_percent[0] < 0.2,  # Ціна в нижній частині BB
                self.roc[0] > 1,  # Позитивний ROC
            ])

            # 3. Volume сигнали (вага 20%)
            volume_signals = sum([
                self.volume_ratio[0] > self.params.volume_spike_ratio,  # Спайк об'єму
                self.data.volume[0] > self.volume_short[0],  # Об'єм вище короткої MA
                self.volume_ratio[0] > self.volume_ratio[-1],  # Зростаючий об'єм
            ])

            # 4. BTC специфічні паттерни (вага 15%)
            btc_patterns = sum([
                self.gap_size[0] > self.params.gap_threshold and self.daily_return[0] > 0,  # Позитивний гап
                self.body_size[0] > 0.015,  # Велике тіло свічки (1.5%)
                self.lower_shadow[0] > self.upper_shadow[0] * 2,  # Довгий нижній тінь
                self.data.close[0] > self.data.high[-1],  # Пробій попереднього максимуму
                current_volatility > self.params.high_vol_threshold,  # Висока волатільність
            ])

            # 5. Ринковий режим бонус (вага 10%)
            regime_bonus = 0
            if self.market_regime in ['strong_bull', 'bull']:
                regime_bonus = 2
            elif self.market_regime == 'sideways':
                regime_bonus = 1

            # Загальний скор
            total_score = (trend_signals * 3 + oscillator_signals * 2.5 +
                          volume_signals * 2 + btc_patterns * 1.5 + regime_bonus)

            # Адаптивний поріг входу
            entry_thresholds = {
                'strong_bull': 8,
                'bull': 10,
                'sideways': 12,
                'bear': 15,
                'strong_bear': 18
            }

            required_score = entry_thresholds.get(self.market_regime, 12)

            if total_score >= required_score:
                position_size = self.calculate_dynamic_position_size(self.market_regime, current_volatility)
                size = int(self.broker.get_cash() * position_size / current_price)

                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.entry_bar = len(self.data)
                    self.highest_price = current_price

        else:
            # Управління відкритою позицією
            current_return = (current_price - self.entry_price) / self.entry_price
            bars_held = len(self.data) - self.entry_bar

            # Оновлення найвищої ціни для trailing stop
            if current_price > self.highest_price:
                self.highest_price = current_price

            # Динамічні стоп-лосс та тейк-профіт
            volatility_multiplier = max(0.5, min(2.0, current_volatility / 0.03))

            dynamic_stop = self.params.stop_loss * volatility_multiplier
            dynamic_profit = self.params.profit_target * volatility_multiplier

            # Trailing stop (30% від максимального прибутку)
            trailing_stop = (self.highest_price - self.entry_price) / self.entry_price * 0.3
            effective_stop = max(dynamic_stop, trailing_stop)

            # Сигнали виходу
            exit_conditions = [
                current_return <= -effective_stop,  # Стоп-лосс
                current_return >= dynamic_profit,  # Тейк-профіт
                self.ema_cross_ultra < 0,  # Ультрашвидкий розворот
                self.rsi[0] > self.params.rsi_overbought,  # Перекупленість
                self.bb_percent[0] > 0.9,  # Ціна біля верхньої BB
                self.volume_ratio[0] < 0.5,  # Падіння об'єму
                bars_held > 15,  # Максимальний час утримання
                self.market_regime == 'strong_bear',  # Сильний ведмежий ринок
            ]

            # Режим-специфічні умови виходу
            regime_exit_rules = {
                'strong_bull': lambda: sum(exit_conditions[:4]) >= 2,
                'bull': lambda: sum(exit_conditions[:5]) >= 2,
                'sideways': lambda: sum(exit_conditions[:6]) >= 2,
                'bear': lambda: sum(exit_conditions) >= 1,
                'strong_bear': lambda: True  # Негайний вихід
            }

            should_exit = regime_exit_rules.get(self.market_regime, lambda: sum(exit_conditions) >= 2)()

            if should_exit:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class BTCVolumeBreakoutStrategy(bt.Strategy):
    """Стратегія пробоїв на основі специфіки об'ємів BTC"""
    params = (
        ('volume_ma_short', 5), ('volume_ma_long', 20),
        ('volume_breakout_ratio', 3.0), # 300% збільшення об'єму
        ('price_breakout_period', 10),
        ('atr_period', 14), ('atr_multiplier', 2.0),
        ('position_size', 0.4), ('max_positions', 1),
        ('rsi_filter', 70), # Фільтр перекупленості
    )

    def __init__(self):
        # Volume індикатори
        self.volume_ma_short = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma_short)
        self.volume_ma_long = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma_long)
        self.volume_ratio = self.data.volume / self.volume_ma_long

        # Price breakout levels
        self.highest_high = bt.indicators.Highest(self.data.high, period=self.params.price_breakout_period)
        self.lowest_low = bt.indicators.Lowest(self.data.low, period=self.params.price_breakout_period)

        # Volatility
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

        # Filters
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # Пробій вгору з підтвердженням об'єму
            price_breakout = current_price > self.highest_high[-1]
            volume_breakout = self.volume_ratio[0] > self.params.volume_breakout_ratio
            rsi_filter = self.rsi[0] < self.params.rsi_filter

            # Додаткові підтвердження
            momentum_confirmation = self.data.close[0] > self.data.open[0]  # Зелена свічка
            gap_up = self.data.open[0] > self.data.close[-1] * 1.005  # Гап вгору 0.5%

            if (price_breakout and volume_breakout and rsi_filter and
                momentum_confirmation):

                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        else:
            # Вихід при падінні нижче стоп-рівня або зменшенні об'єму
            stop_level = self.lowest_low[-1] - self.atr[0] * self.params.atr_multiplier
            volume_exit = self.volume_ratio[0] < 0.7

            if current_price < stop_level or volume_exit:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class BTCSeasonalityStrategy(bt.Strategy):
    """Стратегія на основі сезонних паттернів BTC"""
    params = (
        ('ema_fast', 3), ('ema_slow', 8),
        ('position_size_base', 0.35),
        ('seasonal_multipliers', {
            'january': 1.2,  # Січень зазвичай позитивний
            'february': 0.8, # Лютий часто слабкий
            'march': 1.1,    # Березень recovery
            'april': 1.3,    # Квітень історично сильний
            'may': 0.9,      # "Sell in May"
            'october': 1.4,  # Жовтень Uptober
            'november': 1.3, # Листопад сильний
            'december': 1.1, # Грудень mixed
        }),
        ('week_multipliers', {
            0: 1.0,  # Понеділок
            1: 1.1,  # Вівторок
            2: 1.2,  # Середа - найкращий день
            3: 1.1,  # Четвер
            4: 0.9,  # П'ятниця
            5: 0.7,  # Субота - слабкий
            6: 0.8,  # Неділя - слабкий
        })
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)
        self.ema_cross = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)

        self.order = None

    def get_seasonal_multiplier(self):
        """Отримання сезонного мультиплікатора"""
        current_date = self.data.datetime.date(0)
        month_name = current_date.strftime('%B').lower()
        weekday = current_date.weekday()

        month_mult = self.params.seasonal_multipliers.get(month_name, 1.0)
        week_mult = self.params.week_multipliers.get(weekday, 1.0)

        return month_mult * week_mult

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        seasonal_mult = self.get_seasonal_multiplier()
        adjusted_size = self.params.position_size_base * seasonal_mult

        if not self.position and self.ema_cross > 0 and seasonal_mult > 0.9:
            size = int(self.broker.get_cash() * min(adjusted_size, 0.6) / current_price)
            if size > 0:
                self.order = self.buy(size=size)
        elif self.position and (self.ema_cross < 0 or seasonal_mult < 0.8):
            self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


def main():
    """Головна функція з оптимізованими стратегіями на основі аналізу BTC 2018-2025"""
    CSV_FILE = "CryptoTrade/data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    print("🚀 ОПТИМІЗОВАНІ СТРАТЕГІЇ НА ОСНОВІ АНАЛІЗУ BTC 2018-2025")
    print("=" * 80)
    print("📊 Аналіз періоду:")
    print("   • 2018: Ведмежий ринок після ATH 2017 (~$20k -> ~$3.2k)")
    print("   • 2019: Боковий рух з відновленням (~$3.2k -> ~$7.2k)")
    print("   • 2020-2021: Бичий супер-цикл (~$7.2k -> ~$69k)")
    print("   • 2022: Ведмежий ринок (~$69k -> ~$15.5k)")
    print("   • 2023-2024: Відновлення та новий ATH (~$15.5k -> ~$73k)")
    print("   • 2025: Поточний період")

    strategies_to_test = [
        {
            'name': '🎯 Optimized BTC Analysis Strategy',
            'class': OptimizedBTCAnalysisStrategy,
            'params': {
                'ema_ultra': 1, 'ema_fast': 2, 'ema_medium': 5, 'ema_slow': 13,
                'rsi_period': 3, 'gap_threshold': 0.03,
                'position_size': 0.45, 'profit_target': 0.04, 'stop_loss': 0.025
            }
        },
        {
            'name': '📈 BTC Volume Breakout Strategy',
            'class': BTCVolumeBreakoutStrategy,
            'params': {
                'volume_breakout_ratio': 3.0, 'price_breakout_period': 10,
                'position_size': 0.4, 'atr_multiplier': 2.0
            }
        },
        {
            'name': '📅 BTC Seasonality Strategy',
            'class': BTCSeasonalityStrategy,
            'params': {
                'ema_fast': 3, 'ema_slow': 8, 'position_size_base': 0.35
            }
        },
        {
            'name': '⚡ Enhanced Hyper Frequency',
            'class': HyperFrequencyStrategy,
            'params': {
                'ema1': 1, 'ema2': 2, 'ema3': 3,
                'rsi_period': 2, 'rsi_ob': 70, 'rsi_os': 30,
                'position_size': 0.4
            }
        },
        {
            'name': '🚀 Ultra HFT Enhanced',
            'class': UltraHighFrequencyStrategy,
            'params': {
                'ema1': 1, 'ema2': 2, 'ema3': 3,
                'rsi_period': 2, 'rsi_ob': 55, 'rsi_os': 45,
                'position_size': 0.5
            }
        },
        {
            'name': '🔥 Nano Frequency Strategy',
            'class': NanoFrequencyStrategy,
            'params': {
                'price_threshold': 0.0001, 'volume_threshold': 1.01,
                'position_size': 0.3, 'max_hold_periods': 3
            }
        },
        {
            'name': '🎯 Multi-Signal HFT Strategy',
            'class': MultiSignalHFTStrategy,
            'params': {
                'ema_ultra': 1, 'ema_fast': 2, 'ema_med': 3,
                'signal_threshold': 3, 'position_size': 0.4
            }
        },
        {
            'name': '📊 Price Action Scalping',
            'class': PriceActionScalpingStrategy,
            'params': {
                'min_body_size': 0.0005, 'volume_spike': 1.05,
                'position_size': 0.35, 'consecutive_candles': 2
            }
        },
        {
            'name': '💥 Volatility Breakout',
            'class': VolatilityBreakoutStrategy,
            'params': {
                'atr_period': 2, 'atr_multiplier': 0.5,
                'position_size': 0.4, 'trailing_stop': 0.003
            }
        },
        {
            'name': '⚡ Extreme Frequency Strategy',
            'class': ExtremeFrequencyStrategy,
            'params': {
                'ema_period': 1, 'momentum_period': 1,
                'price_change_threshold': 0.0001, 'position_size': 0.4
            }
        }
    ]

    best_result = None
    best_score = 0
    all_results = []

    try:
        backtester = CSVBacktester(csv_file=CSV_FILE, initial_cash=100000, commission=0.001)

        for strategy in strategies_to_test:
            print(f"\n🔥 ТЕСТУВАННЯ: {strategy['name']}")
            print("-" * 70)

            try:
                result = backtester.run_backtest(
                    strategy_class=strategy['class'],
                    **strategy['params']
                )

                win_rate = (result['won_trades']/max(result['total_trades'],1)*100)
                profit_per_trade = result['return_pct'] / max(result['total_trades'], 1)

                # Спеціальна BTC метрика: прибуток скоригований на ризик
                risk_adjusted_return = result['return_pct'] / max(abs(result['return_pct'] * 0.1), 1)

                # Бонус за частоту (важливо для HFT)
                frequency_score = min(3.0, result['total_trades'] / 50) # Максимум 3x бонус

                # Комплексний BTC скор
                btc_score = (
                    result['return_pct'] * 0.35 +           # 35% - прибутковість
                    result['total_trades'] * 0.25 +         # 25% - частота
                    win_rate * 0.2 +                        # 20% - відсоток виграшів
                    risk_adjusted_return * 0.1 +            # 10% - ризик-скоригована доходність
                    frequency_score * 10                     # 10% - бонус за частоту
                )

                strategy_result = {
                    'name': strategy['name'],
                    'result': result,
                    'params': strategy['params'],
                    'win_rate': win_rate,
                    'profit_per_trade': profit_per_trade,
                    'btc_score': btc_score,
                    'risk_adjusted_return': risk_adjusted_return
                }
                all_results.append(strategy_result)

                print(f"📊 Трейдів: {result['total_trades']}")
                print(f"📈 Win Rate: {win_rate:.1f}%")
                print(f"💰 Прибуток: {result['return_pct']:+.2f}%")
                print(f"⚡ Прибуток/трейд: {profit_per_trade:.3f}%")
                print(f"🛡️ Ризик-скоригований: {risk_adjusted_return:.2f}")
                print(f"🎯 BTC Скор: {btc_score:.2f}")

                if btc_score > best_score:
                    best_score = btc_score
                    best_result = strategy_result

            except Exception as e:
                print(f"❌ Помилка в {strategy['name']}: {e}")

        # Результати
        if best_result:
            print(f"\n🏆 НАЙКРАЩА BTC СТРАТЕГІЯ:")
            print("=" * 80)
            print(f"📛 Назва: {best_result['name']}")
            print(f"🔄 Трейдів: {best_result['result']['total_trades']}")
            print(f"📈 Win Rate: {best_result['win_rate']:.1f}%")
            print(f"💰 Прибуток: {best_result['result']['return_pct']:+.2f}%")
            print(f"⚡ Прибуток/трейд: {best_result['profit_per_trade']:.3f}%")
            print(f"🛡️ Ризик-скоригований: {best_result['risk_adjusted_return']:.2f}")
            print(f"🎯 BTC Скор: {best_result['btc_score']:.2f}")
            print(f"⚙️ Параметри: {best_result['params']}")

        # ТОП-3 для BTC
        all_results.sort(key=lambda x: x['btc_score'], reverse=True)
        print(f"\n📊 ТОП-3 СТРАТЕГІЙ ДЛЯ BTC:")
        print("-" * 50)
        for i, strategy in enumerate(all_results[:3], 1):
            print(f"{i}. {strategy['name']}: {strategy['btc_score']:.2f} BTC скор")
            print(f"   Прибуток: {strategy['result']['return_pct']:+.2f}% | "
                  f"Трейдів: {strategy['result']['total_trades']} | "
                  f"Win Rate: {strategy['win_rate']:.1f}%")

        print(f"\n💡 РЕКОМЕНДАЦІЇ НА ОСНОВІ АНАЛІЗУ BTC:")
        print("   • Використовуйте адаптивні розміри позицій залежно від волатильності")
        print("   • Враховуйте сезонні паттерни (Uptober, Sell in May)")
        print("   • Застосовуйте trailing stops для захисту прибутку")
        print("   • Підтверджуйте сигнали аномальними об'ємами")
        print("   • Адаптуйтеся до ринкових циклів (Bull/Bear/Sideways)")

        print("\n✅ BTC-ОПТИМІЗОВАНЕ ТЕСТУВАННЯ ЗАВЕРШЕНО!")

    except Exception as e:
        print(f"❌ Критична помилка: {e}")


class CSVBacktester:
    """Простий бектестер для CSV файлів"""

    def __init__(self, csv_file: str, initial_cash: float = 100000, commission: float = 0.001, verbose: bool = True):
        # Покращена логіка визначення шляху
        if not os.path.isabs(csv_file):
            # Спробуємо різні варіанти розташування проекту
            possible_roots = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),  # Від test_strategy.py
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),   # Альтернативний варіант
                os.getcwd(),  # Поточна директорія
            ]

            found = False
            for root in possible_roots:
                full_path = os.path.join(root, csv_file)
                if os.path.exists(full_path):
                    csv_file = full_path
                    found = True
                    break

            if not found:
                # Якщо не знайшли, залишаємо оригінальний шлях
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
                csv_file = os.path.join(project_root, csv_file)

        self.csv_file = csv_file
        self.initial_cash = initial_cash
        self.commission = commission
        self.verbose = verbose

    def load_data(self) -> pd.DataFrame:
        """Завантаження та підготовка даних з CSV"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Файл {self.csv_file} не знайдено")

        try:
            df = pd.read_csv(self.csv_file)

            # Пошук колонки з датою
            date_col = None

            for col in df.columns:
                if any(word in col.lower() for word in ['date', 'time', 'timestamp']):
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.drop_duplicates(subset=[date_col], keep='last')
                df.set_index(date_col, inplace=True)
            else:
                df.set_index(df.columns[0], inplace=True)

            # Стандартизація колонок
            df.columns = df.columns.str.lower().str.strip()
            mapping = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
            df = df.rename(columns=mapping)

            # Перевірка необхідних колонок
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Відсутні колонки: {[col for col in required if col not in df.columns]}")

            if 'volume' not in df.columns:
                df['volume'] = 1000

            # Очищення даних
            df = df[required + ['volume']].dropna()
            df = df[(df > 0).all(axis=1)]
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)

            if len(df) == 0:
                raise ValueError("Після очищення не залишилося даних")

            if self.verbose:
                print(f"✅ Завантажено {len(df)} записів")
            return df

        except Exception as e:
            raise ValueError(f"Помилка завантаження CSV: {str(e)}")

    def run_backtest(self, strategy_class=SimpleMovingAverageStrategy, **strategy_params):
        """Запуск бектестування"""
        data = self.load_data()

        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=data))
        cerebro.addstrategy(strategy_class, **strategy_params)
        cerebro.broker.set_cash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        final_value = cerebro.broker.get_value()

        # Отримання статистики
        trade_analysis = results[0].analyzers.trades.get_analysis()
        total_trades = getattr(getattr(trade_analysis, 'total', None), 'total', 0)
        won_trades = getattr(getattr(trade_analysis, 'won', None), 'total', 0)

        # Prevent division by zero in return calculation
        if self.initial_cash <= 0:
            return_pct = 0.0
        else:
            return_pct = ((final_value - self.initial_cash) / self.initial_cash) * 100

        result = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'profit_loss': final_value - self.initial_cash,
            'return_pct': return_pct,
            'total_trades': total_trades,
            'won_trades': won_trades
        }

        if self.verbose:
            print(f"💰 P&L: ${result['profit_loss']:+,.2f} ({result['return_pct']:+.2f}%)")
            print(f"🔄 Угоди: {total_trades} (виграші: {won_trades})")

        return result


def main():
    """Главная функция с оптимизированными стратегиями на основе анализа данных"""
    CSV_FILE = "CryptoTrade/data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    print("🚀 ТЕСТУВАННЯ ОПТИМІЗОВАНИХ СТРАТЕГІЙ НА ОСНОВІ АНАЛІЗУ ДАНИХ BTC 2018-2025")
    print("=" * 80)

    strategies_to_test = [
        {
            'name': '🎯 Optimized Data-Driven Strategy',
            'class': OptimizedDataDrivenStrategy,
            'params': {
                'ema_ultra_fast': 1, 'ema_fast': 2, 'ema_med': 5, 'ema_slow': 8,
                'rsi_period': 3, 'volatility_threshold': 0.025,
                'position_size': 0.4, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            }
        },
        {
            'name': '📊 Advanced Volume Strategy',
            'class': AdvancedVolumeStrategy,
            'params': {
                'vwap_period': 5, 'obv_period': 8, 'volume_rsi_period': 5,
                'position_size': 0.35, 'volume_breakout_multiplier': 2.0
            }
        },
        {
            'name': '🔄 Market Regime Strategy',
            'class': MarketRegimeStrategy,
            'params': {
                'regime_period': 20, 'trend_threshold': 0.02,
                'position_sizes': {'bull': 0.6, 'bear': 0.2, 'sideways': 0.4}
            }
        },
        {
            'name': '⚡ Enhanced HyperFrequency',
            'class': HyperFrequencyStrategy,
            'params': {
                'ema1': 1, 'ema2': 2, 'ema3': 3,
                'rsi_period': 2, 'rsi_ob': 70, 'rsi_os': 30,
                'position_size': 0.45
            }
        },
        {
            'name': '🚀 Multi-Signal Enhanced',
            'class': MultiSignalHFTStrategy,
            'params': {
                'ema_ultra': 1, 'ema_fast': 2, 'ema_med': 3,
                'signal_threshold': 4, 'position_size': 0.4
            }
        }
    ]

    best_result = None
    best_score = 0
    all_results = []

    try:
        backtester = CSVBacktester(csv_file=CSV_FILE, initial_cash=100000, commission=0.001)

        for strategy in strategies_to_test:
            print(f"\n🔥 ТЕСТУВАННЯ: {strategy['name']}")
            print("-" * 70)

            try:
                result = backtester.run_backtest(
                    strategy_class=strategy['class'],
                    **strategy['params']
                )

                win_rate = (result['won_trades']/max(result['total_trades'],1)*100)
                profit_per_trade = result['return_pct'] / max(result['total_trades'], 1)

                # Комплексная оценка с учетом нескольких факторов
                sharpe_like_ratio = result['return_pct'] / max(abs(result['return_pct'] * 0.1), 1)  # Упрощенный Sharpe
                frequency_bonus = min(2.0, result['total_trades'] / 100)  # Бонус за частоту
                consistency_score = win_rate / 100 * frequency_bonus

                # Итоговый комплексный счет
                complex_score = (result['return_pct'] * 0.4 +          # 40% - прибыльность
                               result['total_trades'] * 0.3 +          # 30% - частота торговли
                               win_rate * 0.2 +                        # 20% - процент выигрышей
                               sharpe_like_ratio * 0.1)                # 10% - риск-скорректированная доходность

                strategy_result = {
                    'name': strategy['name'],
                    'result': result,
                    'params': strategy['params'],
                    'win_rate': win_rate,
                    'profit_per_trade': profit_per_trade,
                    'complex_score': complex_score
                }
                all_results.append(strategy_result)

                print(f"📊 Трейдів: {result['total_trades']}")
                print(f"📈 Win Rate: {win_rate:.1f}%")
                print(f"💰 Прибуток: {result['return_pct']:+.2f}%")
                print(f"⚡ Прибуток/трейд: {profit_per_trade:.3f}%")
                print(f"🎯 Комплексний бал: {complex_score:.2f}")

                if complex_score > best_score:
                    best_score = complex_score
                    best_result = strategy_result

            except Exception as e:
                print(f"❌ Помилка в {strategy['name']}: {e}")

        # Вывод результатов
        if best_result:
            print(f"\n🏆 НАЙКРАЩА СТРАТЕГІЯ ЗА КОМПЛЕКСНОЮ ОЦІНКОЮ:")
            print("=" * 80)
            print(f"📛 Назва: {best_result['name']}")
            print(f"🔄 Трейдів: {best_result['result']['total_trades']}")
            print(f"📈 Win Rate: {best_result['win_rate']:.1f}%")
            print(f"💰 Прибуток: {best_result['result']['return_pct']:+.2f}%")
            print(f"⚡ Прибуток/трейд: {best_result['profit_per_trade']:.3f}%")
            print(f"🎯 Комплексний бал: {best_result['complex_score']:.2f}")
            print(f"⚙️ Параметри: {best_result['params']}")

        # Топ-3 стратегии
        all_results.sort(key=lambda x: x['complex_score'], reverse=True)
        print(f"\n📊 ТОП-3 СТРАТЕГІЙ:")
        print("-" * 50)
        for i, strategy in enumerate(all_results[:3], 1):
            print(f"{i}. {strategy['name']}: {strategy['complex_score']:.2f} балів")
            print(f"   Прибуток: {strategy['result']['return_pct']:+.2f}% | Трейдів: {strategy['result']['total_trades']}")

        print("\n✅ ОПТИМІЗОВАНЕ ТЕСТУВАННЯ ЗАВЕРШЕНО!")

    except Exception as e:
        print(f"❌ Критична помилка: {e}")


class OptimizedDataDrivenStrategy(bt.Strategy):
    """Оптимізована стратегія на основі аналізу даних"""
    params = (
        ('ema_ultra_fast', 1), ('ema_fast', 2), ('ema_med', 5), ('ema_slow', 8),
        ('rsi_period', 3), ('rsi_oversold', 30), ('rsi_overbought', 70),
        ('volatility_threshold', 0.025), ('volume_spike', 1.5),
        ('position_size', 0.4), ('stop_loss_pct', 0.03), ('take_profit_pct', 0.06),
        ('momentum_period', 2), ('bb_period', 10), ('bb_std', 2.0),
    )

    def __init__(self):
        # EMA індикатори різних швидкостей
        self.ema_ultra_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_ultra_fast)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_med = bt.indicators.EMA(self.data.close, period=self.params.ema_med)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)

        # Осцилятори
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)

        # Bollinger Bands для волатільності
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.params.bb_period, devfactor=self.params.bb_std)

        # Volume індикатори
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=5)

        # Волатільність
        self.daily_return = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.volatility = bt.indicators.StdDev(self.daily_return, period=7)

        # Кросовери
        self.ema_cross_fast = bt.indicators.CrossOver(self.ema_ultra_fast, self.ema_fast)
        self.ema_cross_med = bt.indicators.CrossOver(self.ema_fast, self.ema_med)

        self.order = None
        self.entry_price = 0

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # Сигнали входу
            trend_signals = sum([
                self.ema_cross_fast > 0,
                self.ema_cross_med > 0,
                self.data.close[0] > self.ema_fast[0],
                self.momentum[0] > 0,
            ])

            volatility_ok = self.volatility[0] > self.params.volatility_threshold
            volume_ok = self.data.volume[0] > self.volume_sma[0] * self.params.volume_spike
            rsi_ok = self.rsi[0] < self.params.rsi_overbought

            if trend_signals >= 2 and volatility_ok and volume_ok and rsi_ok:
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price

        else:
            # Умови виходу
            current_return = (current_price - self.entry_price) / self.entry_price

            exit_conditions = [
                current_return <= -self.params.stop_loss_pct,  # Стоп-лосс
                current_return >= self.params.take_profit_pct,  # Тейк-профіт
                self.ema_cross_fast < 0,  # Розворот тренду
                self.rsi[0] > self.params.rsi_overbought,  # Перекупленість
                self.data.close[0] < self.bb.bot[0],  # Ціна нижче нижньої BB
            ]

            if any(exit_conditions):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class AdvancedVolumeStrategy(bt.Strategy):
    """Розвинена стратегія на основі аналізу об'ємів"""
    params = (
        ('vwap_period', 5), ('obv_period', 8), ('volume_rsi_period', 5),
        ('volume_breakout_multiplier', 2.0), ('position_size', 0.35),
        ('rsi_period', 7), ('ema_period', 3),
    )

    def __init__(self):
        # Volume індикатори
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.vwap_period)
        self.volume_ratio = self.data.volume / self.volume_sma

        # Approximation of OBV
        self.price_change = self.data.close - self.data.close(-1)
        self.obv_raw = bt.indicators.SumN(
            bt.If(self.price_change > 0, self.data.volume,
                 bt.If(self.price_change < 0, -self.data.volume, 0)),
            period=self.params.obv_period
        )

        # Volume RSI approximation
        self.volume_change = self.data.volume - self.data.volume(-1)
        self.volume_rsi = bt.indicators.RSI(self.data.volume, period=self.params.volume_rsi_period)

        # Price індикатори
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_period)

        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # Volume сигнали
            volume_breakout = self.volume_ratio[0] > self.params.volume_breakout_multiplier
            volume_trend = self.obv_raw[0] > self.obv_raw[-1]
            volume_rsi_ok = self.volume_rsi[0] > 50

            # Price сигнали
            price_trend = self.data.close[0] > self.ema[0]
            rsi_ok = 30 < self.rsi[0] < 70

            if (volume_breakout and volume_trend and volume_rsi_ok and
                price_trend and rsi_ok):
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        else:
            # Вихід при падінні volume або негативному тренді
            if (self.volume_ratio[0] < 0.8 or
                self.data.close[0] < self.ema[0] or
                self.rsi[0] > 75):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MarketRegimeStrategy(bt.Strategy):
    """Стратегія адаптації до ринкових режимів"""
    params = (
        ('regime_period', 20), ('trend_threshold', 0.02),
        ('position_sizes', {'bull': 0.6, 'bear': 0.2, 'sideways': 0.4}),
        ('volatility_period', 10), ('volume_period', 5),
    )

    def __init__(self):
        # Індикатори для визначення режиму
        self.ema_fast = bt.indicators.EMA(self.data.close, period=5)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.regime_period)

        # Volatility
        self.daily_return = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.volatility = bt.indicators.StdDev(self.daily_return, period=self.params.volatility_period)

        # Volume
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)

        # Trend strength
        self.trend_strength = (self.ema_fast - self.ema_slow) / self.ema_slow

        self.order = None
        self.current_regime = 'sideways'

    def determine_regime(self):
        """Визначення поточного ринкового режиму"""
        trend = self.trend_strength[0]
        vol = self.volatility[0]

        if trend > self.params.trend_threshold and vol > 0.02:
            return 'bull'
        elif trend < -self.params.trend_threshold and vol > 0.02:
            return 'bear'
        else:
            return 'sideways'

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Визначення режиму
        self.current_regime = self.determine_regime()
        position_size = self.params.position_sizes.get(self.current_regime, 0.3)

        if not self.position:
            # Сигнали входу залежно від режиму
            if self.current_regime == 'bull':
                signal = (self.data.close[0] > self.ema_fast[0] and
                         self.data.volume[0] > self.volume_sma[0])
            elif self.current_regime == 'sideways':
                signal = (abs(self.trend_strength[0]) < 0.01 and
                         self.data.close[0] > self.ema_fast[0])
            else:  # bear
                signal = False  # Не торгуємо в ведмежому ринку

            if signal:
                size = int(self.broker.get_cash() * position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        else:
            # Вихід залежно від режиму
            if (self.current_regime == 'bear' or
                self.data.close[0] < self.ema_fast[0]):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class NanoFrequencyStrategy(bt.Strategy):
    """Нано-частотна стратегія з максимальною чутливістю"""
    params = (
        ('price_threshold', 0.0001),  # 0.01% мінімальна зміна
        ('volume_threshold', 1.01),   # 1% збільшення volume
        ('position_size', 0.3),       # Малий розмір через високий ризик
        ('max_hold_periods', 3),      # Максимум 3 періоди утримання
    )

    def __init__(self):
        self.price_change = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.volume_change = self.data.volume / self.data.volume(-1)
        self.hold_periods = 0
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # Вхід на найменших змінах
            if (abs(self.price_change[0]) > self.params.price_threshold and
                self.volume_change[0] > self.params.volume_threshold and
                self.price_change[0] > 0):

                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
                    self.hold_periods = 0
        else:
            self.hold_periods += 1
            # Вихід через максимальний час утримання або негативну зміну
            if (self.hold_periods >= self.params.max_hold_periods or
                self.price_change[0] < -self.params.price_threshold):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MultiSignalHFTStrategy(bt.Strategy):
    """Мультисигнальна HFT стратегія з 15+ індикаторами"""
    params = (
        ('ema_ultra', 1), ('ema_fast', 2), ('ema_med', 3),
        ('rsi_period', 2), ('rsi_neutral', 50),
        ('stoch_period', 2), ('williams_period', 2), ('cci_period', 2),
        ('momentum_period', 1), ('roc_period', 1), ('trix_period', 3),
        ('volume_sma', 2), ('atr_period', 2), ('adx_period', 3),
        ('position_size', 0.4), ('signal_threshold', 3),
    )

    def __init__(self):
        # Множина індикаторів для максимальної точності
        self.ema_ultra = bt.indicators.EMA(self.data.close, period=self.params.ema_ultra)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_med = bt.indicators.EMA(self.data.close, period=self.params.ema_med)

        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.stoch = bt.indicators.Stochastic(self.data, period=self.params.stoch_period)
        self.williams = bt.indicators.WilliamsR(self.data, period=self.params.williams_period)
        self.cci = bt.indicators.CCI(self.data, period=self.params.cci_period)

        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.roc = bt.indicators.ROC(self.data.close, period=self.params.roc_period)
        self.trix = bt.indicators.TRIX(self.data.close, period=self.params.trix_period)

        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_sma)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)

        # Кросовери
        self.ema_cross_ultra_fast = bt.indicators.CrossOver(self.ema_ultra, self.ema_fast)
        self.ema_cross_fast_med = bt.indicators.CrossOver(self.ema_fast, self.ema_med)

        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # 15 різних бичачих сигналів
            buy_signals = sum([
                self.ema_cross_ultra_fast > 0,                    # EMA кросовер
                self.ema_cross_fast_med > 0,                      # EMA кросовер 2
                self.data.close[0] > self.ema_ultra[0],           # Ціна > Ultra EMA
                self.rsi[0] < self.params.rsi_neutral,            # RSI нейтральний
                self.stoch.percK[0] < 50,                         # Stoch < 50
                self.williams[0] < -50,                           # Williams < -50
                self.cci[0] < 0,                                  # CCI < 0
                self.momentum[0] > 0,                             # Позитивний momentum
                self.roc[0] > 0,                                  # Позитивний ROC
                self.trix[0] > self.trix[-1],                     # TRIX зростає
                self.data.volume[0] > self.volume_sma[0],         # Volume > SMA
                self.atr[0] > self.atr[-1],                       # ATR зростає
                self.adx[0] > 20,                                 # ADX > 20 (тренд)
                self.data.close[0] > self.data.close[-1],         # Зростання ціни
                self.data.high[0] > self.data.high[-1],           # Новий максимум
            ])

            if buy_signals >= self.params.signal_threshold:
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        else:
            # Швидкий вихід при негативних сигналах
            sell_signals = sum([
                self.ema_cross_ultra_fast < 0,
                self.data.close[0] < self.ema_ultra[0],
                self.momentum[0] < 0,
                self.roc[0] < 0,
                self.data.close[0] < self.data.close[-1],
            ])

            if sell_signals >= 1:  # Вихід при першому негативному сигналі
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class PriceActionScalpingStrategy(bt.Strategy):
    """Скальпінг на price action з мінімальними затримками"""
    params = (
        ('min_body_size', 0.0005),      # Мінімальний розмір тіла свічки (0.05%)
        ('wick_ratio', 0.3),            # Співвідношення фітиля до тіла
        ('volume_spike', 1.05),         # Спайк volume 5%
        ('position_size', 0.35),        # Консервативний розмір
        ('consecutive_candles', 2),      # Кількість послідовних свічок
    )

    def __init__(self):
        self.body_size = abs(self.data.close - self.data.open) / self.data.open
        self.upper_wick = self.data.high - bt.Max(self.data.open, self.data.close)
        self.lower_wick = bt.Min(self.data.open, self.data.close) - self.data.low
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=3)

        self.consecutive_green = 0
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Визначення зеленої свічки
        is_green = self.data.close[0] > self.data.open[0]
        if is_green:
            self.consecutive_green += 1
        else:
            self.consecutive_green = 0

        if not self.position:
            # Price action сигнали
            conditions = [
                self.body_size[0] > self.params.min_body_size,                    # Достатній розмір тіла
                is_green,                                                         # Зелена свічка
                self.lower_wick[0] < self.body_size[0] * self.params.wick_ratio, # Малий нижній фітиль
                self.data.volume[0] > self.volume_ma[0] * self.params.volume_spike, # Volume спайк
                self.consecutive_green >= self.params.consecutive_candles,        # Послідовні зелені свічки
                self.data.close[0] > self.data.high[-1],                         # Пробій попереднього максимуму
            ]

            if sum(conditions) >= 3:  # Мінімум 3 умови
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        else:
            # Вихід при червоній свічці або великому фітилі
            if (not is_green or
                self.upper_wick[0] > self.body_size[0] * 2 or
                self.data.close[0] < self.data.low[-1]):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class VolatilityBreakoutStrategy(bt.Strategy):
    """Стратегія на пробої волатильності"""
    params = (
        ('atr_period', 2), ('atr_multiplier', 0.5),
        ('volume_period', 2), ('volume_multiplier', 1.2),
        ('position_size', 0.4), ('trailing_stop', 0.003),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)
        self.highest = bt.indicators.Highest(self.data.high, period=self.params.atr_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.params.atr_period)

        self.order = None
        self.entry_price = 0
        self.trailing_price = 0

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position:
            # Пробій волатильності
            breakout_level = self.highest[-1] + self.atr[0] * self.params.atr_multiplier
            volume_condition = self.data.volume[0] > self.volume_ma[0] * self.params.volume_multiplier

            if (self.data.close[0] > breakout_level and volume_condition):
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.trailing_price = current_price

        else:
            # Trailing stop
            if current_price > self.trailing_price:
                self.trailing_price = current_price

            stop_price = self.trailing_price * (1 - self.params.trailing_stop)

            if (current_price < stop_price or
                current_price < self.lowest[-1]):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class ExtremeFrequencyStrategy(bt.Strategy):
    """Екстремально високочастотна стратегія"""
    params = (
        ('ema_period', 1),      # Найшвидша EMA
        ('momentum_period', 1), # Миттєвий momentum
        ('volume_ma', 2),       # Дуже швидкий volume MA
        ('price_change_threshold', 0.0001),  # 0.01% зміна ціни
        ('position_size', 0.4), # Малий розмір через екстремальний ризик
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_period)
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma)
        self.price_change = (self.data.close - self.data.close(-1)) / self.data.close(-1)
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        # Вхід на найменших змінах
        if not self.position:
            buy_conditions = [
                abs(self.price_change[0]) > self.params.price_change_threshold,
                self.data.close[0] > self.ema[0],
                self.momentum[0] > 0,
                self.data.volume[0] > self.volume_ma[0],
                self.price_change[0] > 0  # Тільки на зростанні
            ]

            if sum(buy_conditions) >= 2:  # Мінімум 2 умови
                size = int(self.broker.get_cash() * self.params.position_size / current_price)
                if size > 0:
                    self.order = self.buy(size=size)

        # Миттєвий вихід
        else:
            exit_conditions = [
                self.data.close[0] < self.ema[0],
                self.momentum[0] < 0,
                self.price_change[0] < -self.params.price_change_threshold
            ]

            if any(exit_conditions):  # Вихід при першому негативному сигналі
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None