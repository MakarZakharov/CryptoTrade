import backtrader as bt

class OptimizedBTCStrategy(bt.Strategy):
    """Активная стратегия для частой торговли BTC"""

    params = (
        ('sma_fast', 7),        # Быстрая SMA
        ('sma_slow', 21),       # Медленная SMA
        ('rsi_period', 14),     # Период RSI
        ('rsi_high', 75),       # Верхний уровень RSI
        ('rsi_low', 25),        # Нижний уровень RSI
        ('bb_period', 20),      # Период Bollinger Bands
        ('bb_devfactor', 2.0),  # Стандартное отклонение BB
        ('position_size', 0.95), # Размер позиции
        ('stop_loss', 0.05),    # Стоп-лосс 5%
        ('take_profit', 0.15),  # Тейк-профит 15%
        ('trail_percent', 0.03), # Трейлинг стоп 3%
    )

    def __init__(self):
        # Скользящие средние
        self.sma_fast = bt.ind.SMA(period=self.p.sma_fast)
        self.sma_slow = bt.ind.SMA(period=self.p.sma_slow)
        self.sma_cross = bt.ind.CrossOver(self.sma_fast, self.sma_slow)

        # RSI для перекупленности/перепроданности
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)

        # Bollinger Bands для волатильности
        self.bb = bt.ind.BollingerBands(period=self.p.bb_period, devfactor=self.p.bb_devfactor)

        # MACD для трендовых сигналов
        self.macd = bt.ind.MACD()
        self.macd_signal = bt.ind.CrossOver(self.macd.macd, self.macd.signal)

        # Переменные для управления позицией
        self.order = None
        self.buy_price = None
        self.highest_price = None
        self.trade_count = 0

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]

        if not self.position:
            # Условия для покупки (множественные сигналы)
            buy_signals = []

            # Сигнал 1: Пересечение SMA вверх
            if self.sma_cross[0] > 0:
                buy_signals.append("SMA_CROSS_UP")

            # Сигнал 2: RSI перепродан и растет
            if (self.rsi[0] < self.p.rsi_low and
                self.rsi[0] > self.rsi[-1]):
                buy_signals.append("RSI_OVERSOLD_RECOVERY")

            # Сигнал 3: Цена касается нижней Bollinger Band
            if (current_price <= self.bb.bot[0] * 1.005 and
                current_price > self.bb.bot[-1]):
                buy_signals.append("BB_LOWER_BOUNCE")

            # Сигнал 4: MACD пересечение вверх
            if self.macd_signal[0] > 0:
                buy_signals.append("MACD_CROSS_UP")

            # Сигнал 5: Быстрая SMA выше медленной и цена выше быстрой SMA
            if (self.sma_fast[0] > self.sma_slow[0] and
                current_price > self.sma_fast[0] and
                self.rsi[0] < 70):
                buy_signals.append("TREND_CONTINUATION")

            # Покупаем если есть хотя бы один сильный сигнал
            if len(buy_signals) >= 1:
                size = (self.broker.cash * self.p.position_size) / current_price
                self.order = self.buy(size=size)
                self.buy_price = current_price
                self.highest_price = current_price
                self.trade_count += 1
                print(f"BUY #{self.trade_count}: ${current_price:.2f} - Signals: {buy_signals}")

        else:
            # Обновляем максимальную цену для трейлинг стопа
            if current_price > self.highest_price:
                self.highest_price = current_price

            # Расчет P&L
            pnl_pct = (current_price - self.buy_price) / self.buy_price

            # Условия для продажи
            sell_signals = []

            # Стоп-лосс
            if pnl_pct <= -self.p.stop_loss:
                sell_signals.append(f"STOP_LOSS_{pnl_pct*100:.1f}%")

            # Тейк-профит
            elif pnl_pct >= self.p.take_profit:
                sell_signals.append(f"TAKE_PROFIT_{pnl_pct*100:.1f}%")

            # Трейлинг стоп
            elif (self.highest_price - current_price) / self.highest_price >= self.p.trail_percent:
                sell_signals.append(f"TRAILING_STOP_{pnl_pct*100:.1f}%")

            # Технические сигналы на продажу
            elif self.sma_cross[0] < 0:  # SMA пересечение вниз
                sell_signals.append(f"SMA_CROSS_DOWN_{pnl_pct*100:.1f}%")

            elif self.rsi[0] > self.p.rsi_high:  # RSI перекуплен
                sell_signals.append(f"RSI_OVERBOUGHT_{pnl_pct*100:.1f}%")

            elif (current_price >= self.bb.top[0] * 0.995):  # Цена у верхней BB
                sell_signals.append(f"BB_UPPER_REJECTION_{pnl_pct*100:.1f}%")

            elif self.macd_signal[0] < 0:  # MACD пересечение вниз
                sell_signals.append(f"MACD_CROSS_DOWN_{pnl_pct*100:.1f}%")

            # Продаем если есть сигнал
            if sell_signals:
                self.order = self.close()
                print(f"SELL #{self.trade_count}: ${current_price:.2f} - {sell_signals[0]}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            # Перевірка на нульове значення trade.value для уникнення ділення на нуль
            if trade.value != 0:
                pnl_pct = (trade.pnl / abs(trade.value)) * 100
                print(f"Trade #{self.trade_count} закрито: PnL = ${trade.pnl:.2f} ({pnl_pct:+.2f}%)")
            else:
                # Якщо trade.value дорівнює нулю, просто показуємо абсолютний PnL
                print(f"Trade #{self.trade_count} закрито: PnL = ${trade.pnl:.2f}")



