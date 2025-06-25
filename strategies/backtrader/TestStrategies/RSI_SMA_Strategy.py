import backtrader as bt

class RSI_SMA_Strategy(bt.Strategy):
    """
    Агрессивная стратегия для дневного таймфрейма с частыми сделками
    Использует RSI и SMA для генерации торговых сигналов
    """

    params = (
        # RSI параметры
        ('rsi_period', 18),
        ('rsi_overbought', 35),
        ('rsi_oversold', 65),
        ('rsi_exit_overbought', 75),
        ('rsi_exit_oversold', 25),

        # SMA параметры
        ('sma_fast', 10),
        ('sma_slow', 20),

        # Управление позициями
        ('position_size', 1.0),  # Изменено с 0.12 на 1.0
        ('stop_loss', 0.02),     # 2% стоп-лосс
        ('take_profit', 0.035),   # 3% тейк-профит

        # Логирование
        ('log_enabled', True),   # Включить/выключить логирование
    )

    def __init__(self):
        # Основные индикаторы
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.sma_fast = bt.indicators.SMA(period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SMA(period=self.params.sma_slow)

        # Дополнительные индикаторы для более частой торговли
        self.ema_fast = bt.indicators.EMA(period=8)
        self.ema_slow = bt.indicators.EMA(period=16)

        # Сигналы
        self.sma_crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        self.ema_crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)

        # Переменные для отслеживания позиций
        self.entry_price = 0
        self.order = None

    def log(self, txt, dt=None):
        """Функция для логирования"""
        if self.params.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')

    def next(self):
        # Отменяем предыдущие ордера если есть
        if self.order:
            return

        current_price = self.data.close[0]
        rsi_value = self.rsi[0]

        if not self.position:
            # Условия для LONG позиций (множественные сигналы для частой торговли)
            long_conditions = [
                # RSI сигналы
                rsi_value < self.params.rsi_oversold,
                rsi_value < 35 and rsi_value > self.rsi[-1],  # RSI растет от низких значений

                # SMA сигналы
                self.sma_crossover > 0,  # Быстрая SMA пересекает медленную вверх
                current_price > self.sma_fast[0] and self.sma_fast[0] > self.sma_fast[-1],

                # EMA сигналы
                self.ema_crossover > 0,  # Быстрая EMA пересекает медленную вверх
                current_price > self.ema_fast[0],

                # Дополнительные сигналы
                current_price > self.data.close[-1],  # Цена растет
                rsi_value > 30 and rsi_value < 50,  # RSI в зоне восстановления
            ]

            # Условия для SHORT позиций
            short_conditions = [
                # RSI сигналы
                rsi_value > self.params.rsi_overbought,
                rsi_value > 65 and rsi_value < self.rsi[-1],  # RSI падает от высоких значений

                # SMA сигналы
                self.sma_crossover < 0,  # Быстрая SMA пересекает медленную вниз
                current_price < self.sma_fast[0] and self.sma_fast[0] < self.sma_fast[-1],

                # EMA сигналы
                self.ema_crossover < 0,  # Быстрая EMA пересекает медленную вниз
                current_price < self.ema_fast[0],

                # Дополнительные сигналы
                current_price < self.data.close[-1],  # Цена падает
                rsi_value < 70 and rsi_value > 50,  # RSI в зоне ослабления
            ]

            # Выполняем сделку если выполнено хотя бы 2 условия
            if sum(long_conditions) >= 2:
                # ИСПРАВЛЕННЫЙ расчет размера позиции
                cash_amount = self.broker.getcash() * self.params.position_size
                size = cash_amount / current_price
                # Убеждаемся что размер минимум 0.001 (для крипто)
                size = max(size, 0.001)

                self.order = self.buy(size=size)
                self.entry_price = current_price
                self.log(f'BUY EXECUTED: Size={size:.6f}, Price={current_price:.2f}, RSI={rsi_value:.2f}, Conditions={sum(long_conditions)}')

            elif sum(short_conditions) >= 2:
                # ИСПРАВЛЕННЫЙ расчет размера позиции для SHORT
                cash_amount = self.broker.getcash() * self.params.position_size
                size = cash_amount / current_price
                # Убеждаемся что размер минимум 0.001 (для крипто)
                size = max(size, 0.001)

                self.order = self.sell(size=size)
                self.entry_price = current_price
                self.log(f'SELL EXECUTED: Size={size:.6f}, Price={current_price:.2f}, RSI={rsi_value:.2f}, Conditions={sum(short_conditions)}')

        else:
            # Управление открытыми позициями
            if self.position.size > 0:  # LONG позиция
                # Условия выхода из LONG
                exit_conditions = [
                    rsi_value > self.params.rsi_exit_overbought,
                    self.sma_crossover < 0,
                    self.ema_crossover < 0,
                    current_price < self.sma_fast[0],
                    rsi_value > 70 and rsi_value < self.rsi[-1],  # RSI разворачивается вниз
                ]

                # Стоп-лосс и тейк-профит
                stop_price = self.entry_price * (1 - self.params.stop_loss)
                profit_price = self.entry_price * (1 + self.params.take_profit)

                if (sum(exit_conditions) >= 1 or
                    current_price <= stop_price or
                    current_price >= profit_price):
                    self.order = self.close()
                    self.log(f'CLOSE LONG: Price={current_price:.2f}, Entry={self.entry_price:.2f}, RSI={rsi_value:.2f}')

            elif self.position.size < 0:  # SHORT позиция
                # Условия выхода из SHORT
                exit_conditions = [
                    rsi_value < self.params.rsi_exit_oversold,
                    self.sma_crossover > 0,
                    self.ema_crossover > 0,
                    current_price > self.sma_fast[0],
                    rsi_value < 30 and rsi_value > self.rsi[-1],  # RSI разворачивается вверх
                ]

                # Стоп-лосс и тейк-профит для SHORT
                stop_price = self.entry_price * (1 + self.params.stop_loss)
                profit_price = self.entry_price * (1 - self.params.take_profit)

                if (sum(exit_conditions) >= 1 or
                    current_price >= stop_price or
                    current_price <= profit_price):
                    self.order = self.close()
                    self.log(f'CLOSE SHORT: Price={current_price:.2f}, Entry={self.entry_price:.2f}, RSI={rsi_value:.2f}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY COMPLETED: Price={order.executed.price:.2f}, Size={order.executed.size:.6f}, Cost=${order.executed.value:.2f}')
            elif order.issell():
                self.log(f'SELL COMPLETED: Price={order.executed.price:.2f}, Size={order.executed.size:.6f}, Value=${order.executed.value:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order FAILED: Status={order.getstatusname()}, Reason: Insufficient funds or invalid size')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        pnl = trade.pnl
        
        # Fix for ZeroDivisionError: check if trade.value is zero
        if abs(trade.value) > 0:
            pnl_pct = (trade.pnl / abs(trade.value)) * 100
            self.log(f'TRADE CLOSED: PnL=${pnl:.2f}, PnL%={pnl_pct:.2f}%, Value=${abs(trade.value):.2f}')
        else:
            # Handle case where trade value is zero
            self.log(f'TRADE CLOSED: PnL=${pnl:.2f}, PnL%=N/A (zero value trade), Value=${abs(trade.value):.2f}')


# Добавляем класс ScalpingStrategy с исправлениями
class ScalpingStrategy(bt.Strategy):
    """
    Дополнительная скальпинговая стратегия для очень частой торговли
    """

    params = (
        ('rsi_period', 7),
        ('rsi_upper', 65),
        ('rsi_lower', 35),
        ('bb_period', 10),
        ('bb_devfactor', 1.5),
        ('position_size', 1.0),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.bb = bt.indicators.BollingerBands(period=self.params.bb_period,
                                               devfactor=self.params.bb_devfactor)
        self.ema5 = bt.indicators.EMA(period=5)
        self.ema13 = bt.indicators.EMA(period=13)

        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        rsi_val = self.rsi[0]

        if not self.position:
            # Сигналы на покупку (очень агрессивные)
            if (rsi_val < self.params.rsi_lower and
                current_price <= self.bb.lines.bot[0] and
                current_price > self.ema5[0]):

                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.buy(size=size)

            # Сигналы на продажу
            elif (rsi_val > self.params.rsi_upper and
                  current_price >= self.bb.lines.top[0] and
                  current_price < self.ema5[0]):

                cash_amount = self.broker.getcash() * self.params.position_size
                size = max(cash_amount / current_price, 0.001)
                self.order = self.sell(size=size)
        else:
            # Быстрый выход из позиций
            if self.position.size > 0:
                if rsi_val > 60 or current_price < self.ema5[0]:
                    self.order = self.close()
            elif self.position.size < 0:
                if rsi_val < 40 or current_price > self.ema5[0]:
                    self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None




