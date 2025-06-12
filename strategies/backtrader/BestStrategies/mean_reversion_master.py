import backtrader as bt
import backtrader.indicators as btind


class Makar(bt.Strategy):
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
                    print(f"📉 ПРОДАЖ: {current_price:.2f}, Прибуток: {profit_pct * 100:.2f}%")
            except (IndexError, TypeError, ZeroDivisionError):
                # Пропускаем если ошибка расчетов
                pass

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
