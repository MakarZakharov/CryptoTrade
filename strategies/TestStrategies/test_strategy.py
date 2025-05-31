import backtrader as bt


class ProfitableBTCStrategy(bt.Strategy):
    """Исправленная агрессивная BTC стратегия с улучшенной защитой от ошибок"""

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
        self.ema_cross_down = bt.ind.CrossDown(self.ema_fast, self.ema_slow)

        # Стан
        self.order = None
        self.entry_price = None

    def next(self):
        # Усиленная защита от недостаточного количества данных
        min_bars = max(self.p.ema_slow, self.p.rsi_period) + 5
        if len(self.data) < min_bars:
            return

        # Скасовуємо попередній ордер якщо є
        if self.order:
            return

        try:
            price = self.data.close[0]

            # Дополнительная проверка на корректность данных
            if not price or price <= 0:
                return

            # Проверяем доступность всех индикаторов
            if (len(self.ema_fast) == 0 or len(self.ema_slow) == 0 or
                len(self.rsi) == 0 or len(self.ema_bullish) == 0):
                return

            # ВХІД В ПОЗИЦІЮ
            if not self.position:
                buy_signal = False

                # Безопасная проверка каждого условия отдельно
                try:
                    if self.ema_bullish[0]:
                        buy_signal = True
                except (IndexError, TypeError):
                    pass

                try:
                    if self.rsi[0] < self.p.rsi_oversold:
                        buy_signal = True
                except (IndexError, TypeError):
                    pass

                try:
                    if len(self.ema_cross_up) > 0 and self.ema_cross_up[0]:
                        buy_signal = True
                except (IndexError, TypeError):
                    pass

                if buy_signal:
                    # Розрахунок розміру позиції
                    size = (self.broker.cash * self.p.position_size) / price
                    if size > 0:
                        self.order = self.buy(size=size)
                        self.entry_price = price
                        try:
                            rsi_val = self.rsi[0] if len(self.rsi) > 0 else 0
                            ema_fast_val = self.ema_fast[0] if len(self.ema_fast) > 0 else 0
                            ema_slow_val = self.ema_slow[0] if len(self.ema_slow) > 0 else 0
                            print(f"📈 КУПІВЛЯ: {price:.2f}, RSI: {rsi_val:.2f}, EMA Fast: {ema_fast_val:.2f}, EMA Slow: {ema_slow_val:.2f}")
                        except:
                            print(f"📈 КУПІВЛЯ: {price:.2f}")

            # ВИХІД З ПОЗИЦІЇ
            elif self.position and self.entry_price:
                try:
                    profit_pct = (price - self.entry_price) / self.entry_price
                    exit_signal = False

                    # Перевіряємо умови виходу по одній
                    if profit_pct < -0.10 or profit_pct > 0.20:
                        exit_signal = True

                    # RSI перекуплений + тренд вниз
                    try:
                        if (self.rsi[0] > self.p.rsi_overbought and
                            len(self.ema_bullish) > 0 and not self.ema_bullish[0]):
                            exit_signal = True
                    except (IndexError, TypeError):
                        pass

                    # EMA кросс вниз
                    try:
                        if len(self.ema_cross_down) > 0 and self.ema_cross_down[0]:
                            exit_signal = True
                    except (IndexError, TypeError):
                        pass

                    if exit_signal:
                        self.order = self.close()
                        self.entry_price = None
                        try:
                            rsi_val = self.rsi[0] if len(self.rsi) > 0 else 0
                            print(f"📉 ПРОДАЖ: {price:.2f}, Прибуток: {profit_pct*100:.2f}%, RSI: {rsi_val:.2f}")
                        except:
                            print(f"📉 ПРОДАЖ: {price:.2f}, Прибуток: {profit_pct*100:.2f}%")

                except (IndexError, TypeError, ZeroDivisionError):
                    # Просто пропускаем итерацию без сообщения
                    pass

        except Exception:
            # Молча пропускаем любые другие ошибки
            pass

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
