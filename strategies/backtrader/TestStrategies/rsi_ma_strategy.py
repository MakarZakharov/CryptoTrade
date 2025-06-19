import backtrader as bt


class RSI_MA_Strategy(bt.Strategy):
    """
    Стратегия на основе RSI и скользящих средних
    
    Сигналы на покупку:
    - RSI < oversold_level (перепроданность)
    - Цена выше быстрой MA
    - Быстрая MA выше медленной MA (восходящий тренд)
    
    Сигналы на продажу:
    - RSI > overbought_level (перекупленность)
    - Или стоп-лосс/тейк-профит
    """
    
    params = (
        ('rsi_period', 14),           # Период RSI
        ('ma_fast_period', 10),       # Быстрая скользящая средняя
        ('ma_slow_period', 30),       # Медленная скользящая средняя
        ('oversold_level', 45),       # Уровень перепроданности RSI (еще больше ослаблено)
        ('overbought_level', 55),     # Уровень перекупленности RSI (еще больше ослаблено)
        ('stop_loss', 0.03),          # Стоп-лосс в процентах (3%)
        ('take_profit', 0.08),        # Тейк-профит в процентах (8%)
        ('position_size', 1.0),      # Размер позиции от доступного капитала
        ('min_bars', 35),             # Минимум баров для начала торговли
    )

    def __init__(self):
        # Индикаторы
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.ma_fast = bt.indicators.SMA(self.data.close, period=self.params.ma_fast_period)
        self.ma_slow = bt.indicators.SMA(self.data.close, period=self.params.ma_slow_period)
        
        # Для отслеживания цены входа
        self.entry_price = None
        self.order = None

        # Статистика и диагностика
        self.trade_count = 0
        self.win_count = 0
        self.signal_checks = 0
        self.rsi_signals = 0
        self.ma_signals = 0
        self.combined_signals = 0

    def next(self):
        # Ждем достаточное количество баров для стабилизации индикаторов
        if len(self.data) < self.params.min_bars:
            return

        # Проверяем, есть ли открытые ордера
        if self.order:
            return

        self.signal_checks += 1

        # Если у нас нет позиции
        if not self.position:
            # Упрощенные условия для покупки
            rsi_condition = self.rsi[0] < self.params.oversold_level

            if rsi_condition:
                self.rsi_signals += 1
                self.combined_signals += 1

                # Вычисляем размер позиции
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * self.params.position_size) / price)

                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = price
                    self.log(f'BUY SIGNAL: RSI={self.rsi[0]:.2f}, Price={price:.2f}, '
                            f'MA_Fast={self.ma_fast[0]:.2f}, MA_Slow={self.ma_slow[0]:.2f}, Size={size}')

        # Если у нас есть позиция
        else:
            current_price = self.data.close[0]
            
            # Рассчитываем прибыль/убыток в процентах
            if self.entry_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                # Условия для продажи
                sell_signal = False
                sell_reason = ""

                # 1. RSI перекуплен
                if self.rsi[0] > self.params.overbought_level:
                    sell_signal = True
                    sell_reason = "RSI_OVERBOUGHT"

                # 2. Стоп-лосс
                elif pnl_pct <= -self.params.stop_loss:
                    sell_signal = True
                    sell_reason = "STOP_LOSS"

                # 3. Тейк-профит
                elif pnl_pct >= self.params.take_profit:
                    sell_signal = True
                    sell_reason = "TAKE_PROFIT"

                # 4. Тренд изменился (быстрая MA упала ниже медленной)
                elif self.ma_fast[0] < self.ma_slow[0]:
                    sell_signal = True
                    sell_reason = "TREND_CHANGE"

                if sell_signal:
                    self.order = self.sell()
                    self.log(f'SELL SIGNAL: {sell_reason}, PnL={pnl_pct*100:.2f}%, '
                            f'RSI={self.rsi[0]:.2f}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Value: {order.executed.value:.2f}')
                self.trade_count += 1
                
                # Определяем прибыльность сделки
                if self.entry_price and order.executed.price > self.entry_price:
                    self.win_count += 1
                
                self.entry_price = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
        
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'TRADE CLOSED: Gross P&L: {trade.pnl:.2f}, Net P&L: {trade.pnlcomm:.2f}')

    def log(self, txt, dt=None):
        """Логирование"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')

    def stop(self):
        """Вызывается в конце бэктеста"""
        win_rate = (self.win_count / max(self.trade_count, 1)) * 100
        self.log(f'Strategy Performance: Total Trades: {self.trade_count}, '
                f'Win Rate: {win_rate:.1f}%, Final Value: {self.broker.getvalue():.2f}')
        self.log(f'Signal Diagnostics: Total checks: {self.signal_checks}, '
                f'RSI signals: {self.rsi_signals}, MA signals: {self.ma_signals}, '
                f'Combined signals: {self.combined_signals}')


class SimpleMovingAverageCrossover(bt.Strategy):
    """
    Простая стратегия пересечения скользящих средних
    
    Покупка при пересечении быстрой MA медленной снизу вверх
    Продажа при пересечении быстрой MA медленной сверху вниз
    """
    
    params = (
        ('ma_fast', 10),      # Период быстрой MA
        ('ma_slow', 30),      # Период медленной MA
        ('position_size', 1.0), # Размер позиции
    )

    def __init__(self):
        self.ma_fast = bt.indicators.SMA(self.data.close, period=self.params.ma_fast)
        self.ma_slow = bt.indicators.SMA(self.data.close, period=self.params.ma_slow)
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:  # Быстрая MA пересекла медленную снизу вверх
                size = int((self.broker.getcash() * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.crossover < 0:  # Быстрая MA пересекла медленную сверху вниз
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            pass
        self.order = None


class BollingerBandsStrategy(bt.Strategy):
    """
    Стратегия на основе полос Боллинджера
    
    Покупка при касании нижней полосы (перепроданность)
    Продажа при касании верхней полосы (перекупленность)
    """
    
    params = (
        ('bb_period', 20),        # Период для полос Боллинджера
        ('bb_dev', 2),           # Стандартные отклонения
        ('rsi_period', 14),       # RSI для дополнительного фильтра
        ('rsi_oversold', 30),     # Уровень перепроданности RSI
        ('rsi_overbought', 70),   # Уровень перекупленности RSI
        ('position_size', 0.9),   # Размер позиции
    )

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Покупка при касании нижней полосы + RSI перепродан
            if (self.data.close[0] <= self.bb.lines.bot[0] and 
                self.rsi[0] < self.params.rsi_oversold):
                
                size = int((self.broker.getcash() * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Продажа при касании верхней полосы + RSI перекуплен
            if (self.data.close[0] >= self.bb.lines.top[0] and 
                self.rsi[0] > self.params.rsi_overbought):
                
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            pass
        self.order = None


class MACDStrategy(bt.Strategy):
    """
    Стратегия на основе MACD
    
    Покупка при пересечении MACD сигнальной линии снизу вверх
    Продажа при пересечении MACD сигнальной линии сверху вниз
    """
    
    params = (
        ('macd_fast', 12),        # Быстрая EMA для MACD
        ('macd_slow', 26),        # Медленная EMA для MACD
        ('macd_signal', 9),       # Сигнальная линия
        ('position_size', 0.9),   # Размер позиции
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:  # MACD пересек сигнальную линию снизу вверх
                size = int((self.broker.getcash() * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.crossover < 0:  # MACD пересек сигнальную линию сверху вниз
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            pass
        self.order = None