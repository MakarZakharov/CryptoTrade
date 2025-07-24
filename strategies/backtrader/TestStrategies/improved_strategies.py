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
    Оптимизированная стратегия на основе моментума для достижения 100+ трейдов и 2000$+ прибыли
    """
    
    params = (
        ('momentum_period', 7),          # Уменьшен с 14 до 7 для более частых сигналов
        ('momentum_threshold', 0.005),   # Уменьшен с 0.02 до 0.005 (0.5%) для более чувствительных сигналов
        ('rsi_period', 14),              # Добавлен RSI для фильтрации
        ('rsi_oversold', 35),            # RSI фильтр для покупок
        ('rsi_overbought', 65),          # RSI фильтр для продаж
        ('position_size', 0.8),          # Уменьшен с 1.0 до 0.8 для лучшего управления рисками
        ('take_profit', 0.03),           # Добавлен take profit 3%
        ('stop_loss', 0.015),            # Добавлен stop loss 1.5%
    )

    def __init__(self):
        self.momentum = btind.Momentum(period=self.p.momentum_period)
        self.rsi = btind.RSI(period=self.p.rsi_period)
        self.sma_fast = btind.SMA(period=5)   # Быстрая SMA для дополнительных сигналов
        self.sma_slow = btind.SMA(period=10)  # Медленная SMA для тренда
        self.order = None
        self.entry_price = None

    def next(self):
        if len(self.data) < max(self.p.momentum_period, self.p.rsi_period):
            return
            
        if self.order:
            return

        current_price = self.data.close[0]
        # Нормализуем моментум к процентам
        momentum_pct = self.momentum[0] / current_price if current_price > 0 else 0
        
        if not self.position:
            # Множественные условия входа для более частых сигналов
            buy_conditions = [
                momentum_pct > self.p.momentum_threshold,  # Основной моментум сигнал
                self.rsi[0] < self.p.rsi_oversold,         # RSI фильтр
                current_price > self.sma_fast[0],          # Цена выше быстрой SMA
                self.sma_fast[0] > self.sma_slow[0],       # Восходящий тренд
            ]
            
            sell_conditions = [
                momentum_pct < -self.p.momentum_threshold, # Негативный моментум
                self.rsi[0] > self.p.rsi_overbought,       # RSI фильтр
                current_price < self.sma_fast[0],          # Цена ниже быстрой SMA
                self.sma_fast[0] < self.sma_slow[0],       # Нисходящий тренд
            ]
            
            # Покупка если выполнено минимум 2 условия
            if sum(buy_conditions) >= 2:
                size = (self.broker.cash * self.p.position_size) / current_price
                self.order = self.buy(size=size)
                self.entry_price = current_price
                print(f"📈 MOMENTUM BUY: {current_price:.2f}, Mom: {momentum_pct*100:.2f}%, RSI: {self.rsi[0]:.2f}")
            
            # Продажа если выполнено минимум 2 условия
            elif sum(sell_conditions) >= 2:
                size = (self.broker.cash * self.p.position_size) / current_price
                self.order = self.sell(size=size)
                self.entry_price = current_price
                print(f"📉 MOMENTUM SELL: {current_price:.2f}, Mom: {momentum_pct*100:.2f}%, RSI: {self.rsi[0]:.2f}")
                
        else:
            # Управление позициями с stop loss и take profit
            if self.entry_price:
                profit_pct = (current_price - self.entry_price) / self.entry_price
                
                # Условия закрытия LONG позиции
                if self.position.size > 0:
                    close_long_conditions = [
                        momentum_pct < -self.p.momentum_threshold,   # Смена моментума
                        self.rsi[0] > self.p.rsi_overbought,        # RSI перекуплен
                        profit_pct >= self.p.take_profit,           # Take profit
                        profit_pct <= -self.p.stop_loss,            # Stop loss
                        current_price < self.sma_fast[0],           # Цена под SMA
                    ]
                    
                    if any(close_long_conditions):
                        self.order = self.close()
                        print(f"📉 CLOSE LONG: {current_price:.2f}, Profit: {profit_pct*100:.2f}%")
                
                # Условия закрытия SHORT позиции
                elif self.position.size < 0:
                    close_short_conditions = [
                        momentum_pct > self.p.momentum_threshold,    # Смена моментума
                        self.rsi[0] < self.p.rsi_oversold,          # RSI перепродан
                        -profit_pct >= self.p.take_profit,          # Take profit (инвертирован для SHORT)
                        -profit_pct <= -self.p.stop_loss,           # Stop loss (инвертирован для SHORT)
                        current_price > self.sma_fast[0],           # Цена над SMA
                    ]
                    
                    if any(close_short_conditions):
                        self.order = self.close()
                        print(f"📈 CLOSE SHORT: {current_price:.2f}, Profit: {-profit_pct*100:.2f}%")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
            if not self.position:
                self.entry_price = None


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