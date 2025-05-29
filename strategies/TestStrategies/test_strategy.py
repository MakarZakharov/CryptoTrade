import backtrader as bt


class ProfitableBTCStrategy(bt.Strategy):
    """Спрощена агресивна BTC стратегія для частіших угод"""

    params = (
        ('ema_fast', 12),        # Трохи повільніша швидка EMA
        ('ema_slow', 26),        # Стандартна повільна EMA
        ('rsi_period', 14),      # RSI період
        ('rsi_oversold', 30),    # Перепроданість
        ('rsi_overbought', 70),  # Перекупленість
        ('position_size', 0.95), # 95% капіталу
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
        # Скасовуємо попередній ордер якщо є
        if self.order:
            return

        price = self.data.close[0]

        # ВХІД В ПОЗИЦІЮ
        if not self.position:
            # Простіші умови входу - або тренд вгору або RSI перепроданий
            buy_signal = (
                # Основний сигнал - EMA тренд вгору
                self.ema_bullish[0] or
                # Або RSI показує перепроданість (можливість відскоку)
                self.rsi[0] < self.p.rsi_oversold or
                # Або був кросовер EMA
                self.ema_cross_up[0]
            )

            if buy_signal:
                # Розрахунок розміру позиції
                size = (self.broker.cash * self.p.position_size) / price
                self.order = self.buy(size=size)
                print(f"📈 КУПІВЛЯ: {price:.2f}, RSI: {self.rsi[0]:.2f}, EMA Fast: {self.ema_fast[0]:.2f}, EMA Slow: {self.ema_slow[0]:.2f}")

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
                (self.rsi[0] > self.p.rsi_overbought and not self.ema_bullish[0]) or
                # EMA кросс вниз
                bt.ind.CrossDown(self.ema_fast, self.ema_slow)[0]
            )

            if exit_signal:
                self.order = self.close()
                print(f"📉 ПРОДАЖ: {price:.2f}, Прибуток: {profit_pct*100:.2f}%, RSI: {self.rsi[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
