import backtrader as bt
import backtrader.indicators as btind


class SafeProfitableBTCStrategy(bt.Strategy):
    """
    Улучшенная и безопасная BTC стратегия с защитой от ошибок индексации
    """
    
    params = (
        ('ema_fast', 12),
        ('ema_slow', 26),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('position_size', 1.0),
        ('stop_loss', 0.10),
        ('take_profit', 0.20),
    )

    def __init__(self):
        # Основные индикаторы
        self.ema_fast = btind.EMA(period=self.p.ema_fast)
        self.ema_slow = btind.EMA(period=self.p.ema_slow)
        self.rsi = btind.RSI(period=self.p.rsi_period)
        
        # Сигналы
        self.ema_bullish = self.ema_fast > self.ema_slow
        self.ema_cross_up = btind.CrossUp(self.ema_fast, self.ema_slow)
        self.ema_cross_down = btind.CrossDown(self.ema_fast, self.ema_slow)
        
        # Состояние
        self.order = None
        self.entry_price = None

    def next(self):
        # Защита от недостаточного количества данных
        if len(self.data) < max(self.p.ema_slow, self.p.rsi_period):
            return
            
        # Скасовуємо попередній ордер якщо є
        if self.order:
            return

        current_price = self.data.close[0]
        
        # Защита от некорректных значений
        if not current_price or current_price <= 0:
            return

        # ВХІД В ПОЗИЦІЮ
        if not self.position:
            # Безопасная проверка сигналов
            try:
                buy_signal = (
                    (len(self.ema_bullish) > 0 and self.ema_bullish[0]) or
                    (len(self.rsi) > 0 and self.rsi[0] < self.p.rsi_oversold) or
                    (len(self.ema_cross_up) > 0 and self.ema_cross_up[0])
                )

                if buy_signal:
                    # Розрахунок розміру позиції
                    size = (self.broker.cash * self.p.position_size) / current_price
                    if size > 0:
                        self.order = self.buy(size=size)
                        self.entry_price = current_price
                        print(f"📈 КУПІВЛЯ: {current_price:.2f}, RSI: {self.rsi[0]:.2f}")
            except (IndexError, TypeError):
                # Пропускаем если недостаточно данных
                pass

        # ВИХІД З ПОЗИЦІЇ
        elif self.position and self.entry_price:
            try:
                profit_pct = (current_price - self.entry_price) / self.entry_price

                # Умови виходу
                exit_signal = (
                    # Стоп-лос
                    profit_pct < -self.p.stop_loss or
                    # Тейк-профіт
                    profit_pct > self.p.take_profit or
                    # RSI перекуплений + тренд вниз
                    (len(self.rsi) > 0 and len(self.ema_bullish) > 0 and 
                     self.rsi[0] > self.p.rsi_overbought and not self.ema_bullish[0]) or
                    # EMA кросс вниз
                    (len(self.ema_cross_down) > 0 and self.ema_cross_down[0])
                )

                if exit_signal:
                    self.order = self.close()
                    print(f"📉 ПРОДАЖ: {current_price:.2f}, Прибуток: {profit_pct*100:.2f}%")
            except (IndexError, TypeError, ZeroDivisionError):
                # Пропускаем если ошибка расчетов
                pass

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MovingAverageCrossStrategy(bt.Strategy):
    """
    Классическая стратегия пересечения скользящих средних
    """
    
    params = (
        ('ma_fast', 20),
        ('ma_slow', 50),
        ('position_size', 1.0),
    )

    def __init__(self):
        self.ma_fast = btind.SMA(period=self.p.ma_fast)
        self.ma_slow = btind.SMA(period=self.p.ma_slow)
        self.crossover = btind.CrossOver(self.ma_fast, self.ma_slow)
        self.order = None

    def next(self):
        if len(self.data) < self.p.ma_slow:
            return
            
        if self.order:
            return

        if not self.position:
            if self.crossover[0] > 0:  # Golden Cross
                size = (self.broker.cash * self.p.position_size) / self.data.close[0]
                self.order = self.buy(size=size)
                print(f"📈 MA Cross UP: {self.data.close[0]:.2f}")
        else:
            if self.crossover[0] < 0:  # Death Cross
                self.order = self.close()
                print(f"📉 MA Cross DOWN: {self.data.close[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class RSIStrategy(bt.Strategy):
    """
    Стратегия на основе RSI с зонами перекупленности/перепроданности
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('position_size', 1.0),
    )

    def __init__(self):
        self.rsi = btind.RSI(period=self.p.rsi_period)
        self.order = None

    def next(self):
        if len(self.data) < self.p.rsi_period:
            return
            
        if self.order:
            return

        if not self.position:
            if self.rsi[0] < self.p.rsi_oversold:
                size = (self.broker.cash * self.p.position_size) / self.data.close[0]
                self.order = self.buy(size=size)
                print(f"📈 RSI BUY: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")
        else:
            if self.rsi[0] > self.p.rsi_overbought:
                self.order = self.close()
                print(f"📉 RSI SELL: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MACDStrategy(bt.Strategy):
    """
    Стратегия на основе MACD
    """
    
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('position_size', 1.0),
    )

    def __init__(self):
        self.macd = btind.MACD(
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.order = None

    def next(self):
        if len(self.data) < self.p.macd_slow + self.p.macd_signal:
            return
            
        if self.order:
            return

        if not self.position:
            # MACD пересекает сигнальную линию снизу вверх
            if (len(self.macd.macd) > 1 and len(self.macd.signal) > 1 and
                self.macd.macd[0] > self.macd.signal[0] and 
                self.macd.macd[-1] <= self.macd.signal[-1]):
                
                size = (self.broker.cash * self.p.position_size) / self.data.close[0]
                self.order = self.buy(size=size)
                print(f"📈 MACD BUY: {self.data.close[0]:.2f}")
        else:
            # MACD пересекает сигнальную линию сверху вниз
            if (len(self.macd.macd) > 1 and len(self.macd.signal) > 1 and
                self.macd.macd[0] < self.macd.signal[0] and 
                self.macd.macd[-1] >= self.macd.signal[-1]):
                
                self.order = self.close()
                print(f"📉 MACD SELL: {self.data.close[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class BollingerBandsStrategy(bt.Strategy):
    """
    Стратегия на основе полос Боллинджера
    """
    
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('position_size', 1.0),
    )

    def __init__(self):
        self.bb = btind.BollingerBands(period=self.p.bb_period, devfactor=self.p.bb_dev)
        self.order = None

    def next(self):
        if len(self.data) < self.p.bb_period:
            return
            
        if self.order:
            return

        current_price = self.data.close[0]

        if not self.position:
            # Покупаем при касании нижней полосы
            if current_price <= self.bb.lines.bot[0]:
                size = (self.broker.cash * self.p.position_size) / current_price
                self.order = self.buy(size=size)
                print(f"📈 BB BUY: {current_price:.2f}")
        else:
            # Продаем при касании верхней полосы
            if current_price >= self.bb.lines.top[0]:
                self.order = self.close()
                print(f"📉 BB SELL: {current_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class MomentumStrategy(bt.Strategy):
    """
    Стратегия на основе моментума
    """
    
    params = (
        ('momentum_period', 14),
        ('momentum_threshold', 0.02),  # 2%
        ('position_size', 1.0),
    )

    def __init__(self):
        self.momentum = btind.Momentum(period=self.p.momentum_period)
        self.order = None

    def next(self):
        if len(self.data) < self.p.momentum_period:
            return
            
        if self.order:
            return

        # Нормализуем моментум к процентам
        momentum_pct = self.momentum[0] / self.data.close[0]

        if not self.position:
            if momentum_pct > self.p.momentum_threshold:
                size = (self.broker.cash * self.p.position_size) / self.data.close[0]
                self.order = self.buy(size=size)
                print(f"📈 MOMENTUM BUY: {self.data.close[0]:.2f}, Mom: {momentum_pct*100:.2f}%")
        else:
            if momentum_pct < -self.p.momentum_threshold:
                self.order = self.close()
                print(f"📉 MOMENTUM SELL: {self.data.close[0]:.2f}, Mom: {momentum_pct*100:.2f}%")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class HybridStrategy(bt.Strategy):
    """
    Гибридная стратегия с несколькими индикаторами
    """
    
    params = (
        ('sma_period', 20),
        ('rsi_period', 14),
        ('rsi_oversold', 35),
        ('rsi_overbought', 65),
        ('position_size', 1.0),
    )

    def __init__(self):
        self.sma = btind.SMA(period=self.p.sma_period)
        self.rsi = btind.RSI(period=self.p.rsi_period)
        self.order = None

    def next(self):
        if len(self.data) < max(self.p.sma_period, self.p.rsi_period):
            return
            
        if self.order:
            return

        price_above_sma = self.data.close[0] > self.sma[0]
        price_below_sma = self.data.close[0] < self.sma[0]

        if not self.position:
            # Покупаем если цена выше SMA и RSI показывает перепроданность
            if price_above_sma and self.rsi[0] < self.p.rsi_oversold:
                size = (self.broker.cash * self.p.position_size) / self.data.close[0]
                self.order = self.buy(size=size)
                print(f"📈 HYBRID BUY: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")
        else:
            # Продаем если цена ниже SMA или RSI показывает перекупленность
            if price_below_sma or self.rsi[0] > self.p.rsi_overbought:
                self.order = self.close()
                print(f"📉 HYBRID SELL: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None