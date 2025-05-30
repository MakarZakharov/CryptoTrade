import backtrader as bt


class ProfitableBTCStrategy(bt.Strategy):
    """Исправленная агрессивная BTC стратегия с защитой от ошибок"""

    params = (
        ('ema_fast', 12),
        ('ema_slow', 26),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('position_size', 0.95),
    )

    def __init__(self):
        # Основні індикатори
        self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)

        # Сигнали
        self.ema_bullish = self.ema_fast > self.ema_slow
        self.ema_cross_up = bt.ind.CrossUp(self.ema_fast, self.ema_slow)

        # Стан
        self.order = None

    def next(self):
        # Защита от недостаточного количества данных
        if len(self.data) < max(self.p.ema_slow, self.p.rsi_period):
            return

        # Скасовуємо попередній ордер якщо є
        if self.order:
            return

        try:
            price = self.data.close[0]

            # Дополнительная проверка на корректность данных
            if not price or price <= 0:
                return

            # ВХІД В ПОЗИЦІЮ
            if not self.position:
                # Безопасная проверка условий с защитой от IndexError
                ema_bullish_signal = len(self.ema_bullish) > 0 and self.ema_bullish[0]
                rsi_oversold_signal = len(self.rsi) > 0 and self.rsi[0] < self.p.rsi_oversold
                ema_cross_signal = len(self.ema_cross_up) > 0 and self.ema_cross_up[0]

                buy_signal = (
                    ema_bullish_signal or
                    rsi_oversold_signal or
                    ema_cross_signal
                )

                if buy_signal:
                    # Розрахунок розміру позиції
                    size = (self.broker.cash * self.p.position_size) / price
                    if size > 0:
                        self.order = self.buy(size=size)
                        rsi_val = self.rsi[0] if len(self.rsi) > 0 else 0
                        ema_fast_val = self.ema_fast[0] if len(self.ema_fast) > 0 else 0
                        ema_slow_val = self.ema_slow[0] if len(self.ema_slow) > 0 else 0
                        print(f"📈 КУПІВЛЯ: {price:.2f}, RSI: {rsi_val:.2f}, EMA Fast: {ema_fast_val:.2f}, EMA Slow: {ema_slow_val:.2f}")

            # ВИХІД З ПОЗИЦІЇ
            elif self.position:
                profit_pct = (price - self.position.price) / self.position.price

                # Умови виходу
                exit_signal = (
                    # Стоп-лос 10%
                    profit_pct < -0.10 or
                    # Тейк-профіт 20%
                    profit_pct > 0.20 or
                    # RSI перекуплений + тренд вниз
                    (len(self.rsi) > 0 and len(self.ema_bullish) > 0 and
                     self.rsi[0] > self.p.rsi_overbought and not self.ema_bullish[0]) or
                    # EMA кросс вниз
                    (len(self.ema_fast) > 0 and len(self.ema_slow) > 0 and
                     bt.ind.CrossDown(self.ema_fast, self.ema_slow)[0])
                )

                if exit_signal:
                    self.order = self.close()
                    rsi_val = self.rsi[0] if len(self.rsi) > 0 else 0
                    print(f"📉 ПРОДАЖ: {price:.2f}, Прибуток: {profit_pct*100:.2f}%, RSI: {rsi_val:.2f}")

        except (IndexError, TypeError, ZeroDivisionError) as e:
            # Пропускаем итерацию если есть ошибки с данными
            print(f"⚠️ Пропуск итерации из-за ошибки: {e}")
            return

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
