# rsi_ema_bbands_strategy.py - Нова стратегія

import backtrader as bt


class RSI_EMA_BBands_ATR_Strategy(bt.Strategy):
    """
    Стратегія на основі RSI, EMA, Bollinger Bands і ATR

    Логіка входу:
    - RSI < 30 (перепроданість)
    - Ціна перетинає нижню лінію Bollinger Bands знизу вгору
    - Ціна вище EMA (підтвердження тренду)

    Логіка виходу:
    - Stop Loss / Take Profit на основі ATR
    - RSI > 70 (перекупленість)
    """

    params = (
        ("rsi_period", 14),  # Залишаємо класичний
        ("ema_period", 20),  # Середній тренд
        ("bb_period", 20),
        ("bb_devfactor", 2.0),  # Стандартне BB
        ("atr_period", 14),
        ("atr_multiplier_sl", 1.5),  # Більш "справедливий" SL
        ("atr_multiplier_tp", 3.0),  # Класичне співвідношення
        ("rsi_oversold", 35),  # Більш м'яка умова входу
        ("rsi_overbought", 65),  # Раніший вихід
    )

    def __init__(self):
        """Ініціалізуємо індикатори"""
        print("Запуск стратегії RSI + EMA + Bollinger Bands + ATR...")

        # Основні індикатори
        self.rsi = bt.ind.RSI(self.data.close, period=self.params.rsi_period)
        self.ema = bt.ind.EMA(self.data.close, period=self.params.ema_period)
        self.bb = bt.ind.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_devfactor
        )
        self.atr = bt.ind.ATR(self.data, period=self.params.atr_period)

        # Сигнали перетину Bollinger Bands
        self.bb_bottom_cross = bt.ind.CrossUp(self.data.close, self.bb.lines.bot)

        # Змінні для відслідковування
        self.order = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_count = 0

    def next(self):
        """Головна логіка стратегії на кожному барі"""

        # Якщо є активний ордер - чекаємо
        if self.order:
            return

        current_price = self.data.close[0]

        # Якщо є позиція - перевіряємо чи виходити
        if self.position:
            self.check_exit(current_price)
        else:
            # Якщо немає позиції - шукаємо вхід
            self.check_entry(current_price)

    def check_entry(self, price):
        """Перевіряємо умови для входу в позицію"""

        # Потрібно достатньо даних для розрахунку індикаторів
        min_bars = max(self.params.rsi_period, self.params.ema_period, self.params.bb_period)
        if len(self.data) < min_bars + 5:
            return

        # Умови для покупки
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold  # RSI показує перепроданість
        bb_bounce = self.bb_bottom_cross[0]  # Відскок від нижньої лінії BB
        trend_up = price > self.ema[0]  # Ціна вище EMA (бичачий тренд)

        # Додаткова перевірка: ціна не повинна бути занадто високо від EMA
        price_not_too_high = price < self.ema[0] * 1.05  # Не більше 5% вище EMA

        # Якщо всі умови виконані - купуємо
        conditions_met = sum([rsi_oversold, bb_bounce, trend_up]) >= 2
        if conditions_met and price_not_too_high:
            self.buy_signal(price)

    def buy_signal(self, price):
        """Виконуємо покупку"""
        self.order = self.buy()
        self.trade_count += 1

        # Розраховуємо стоп-лос та тейк-профіт на основі ATR
        atr_value = self.atr[0]
        self.stop_loss = price - (atr_value * self.params.atr_multiplier_sl)
        self.take_profit = price + (atr_value * self.params.atr_multiplier_tp)

        print(f"КУПУЄМО #{self.trade_count}: ${price:.2f} | "
              f"SL: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f} | "
              f"RSI: {self.rsi[0]:.1f}")

    def check_exit(self, price):
        """Перевіряємо умови для виходу з позиції"""

        exit_reasons = []

        # Перевіряємо умови виходу
        if price <= self.stop_loss:
            exit_reasons.append("Stop Loss")
        if price >= self.take_profit:
            exit_reasons.append("Take Profit")
        if self.rsi[0] > self.params.rsi_overbought:
            exit_reasons.append(f"RSI > {self.params.rsi_overbought}")
        # Додаткова умова: якщо ціна пробила верхню лінію BB (потенційна перекупленість)
        if price > self.bb.lines.top[0]:
            exit_reasons.append("Price above BB Top")

        if exit_reasons:
            self.order = self.sell()
            reason = ", ".join(exit_reasons)
            print(f"ПРОДАЄМО: ${price:.2f} | Причина: {reason} | RSI: {self.rsi[0]:.1f}")

    def notify_order(self, order):
        """Обробляємо статуси ордерів"""
        if order.status == order.Completed:
            self.order = None  # Ордер виконаний

    def notify_trade(self, trade):
        """Обробляємо закриті угоди"""
        if trade.isclosed:
            profit_loss = trade.pnl
            result = "ПРИБУТОК" if profit_loss > 0 else "ЗБИТОК"

            # Розраховуємо відсоток прибутку/збитку
            if hasattr(trade, 'price') and trade.price > 0:
                pnl_percent = (profit_loss / (trade.price * abs(trade.size))) * 100
                print(f"Угода закрита: {result} ${profit_loss:.2f} ({pnl_percent:+.2f}%)")
            else:
                print(f"Угода закрита: {result} ${profit_loss:.2f}")

            # Скидаємо рівні
            self.stop_loss = None
            self.take_profit = None

    def get_strategy_name(self):
        """Назва стратегії"""
        return "RSI EMA Bollinger Bands ATR Strategy"