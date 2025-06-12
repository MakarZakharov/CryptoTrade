import backtrader as bt


class LOLStrategy(bt.Strategy):
    """
    LOL Strategy - простая забавная стратегия на основе RSI и SMA
    
    Логика:
    - Покупаем когда RSI < 30 и цена выше быстрой SMA
    - Продаем когда RSI > 70 или цена ниже медленной SMA
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('sma_fast', 10),
        ('sma_slow', 20),
        ('position_size', 1.0),  # Изменено с 0.95 на 1.0
        ('stop_loss', 0.05),
        ('take_profit', 0.15),
    )

    def __init__(self):
        # Индикаторы
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.sma_fast = bt.indicators.SMA(period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SMA(period=self.params.sma_slow)
        
        # Состояние
        self.order = None
        self.entry_price = None

    def log(self, txt, dt=None):
        """Логирование с префиксом LOL"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: LOL - {txt}')

    def next(self):
        # Ждем достаточно данных для индикаторов
        if len(self.data) < max(self.params.rsi_period, self.params.sma_slow):
            return
            
        # Проверяем открытые ордера
        if self.order:
            return

        current_price = self.data.close[0]
        rsi_value = self.rsi[0]

        # Защита от некорректных данных
        if not current_price or current_price <= 0:
            return

        # ВХОД В ПОЗИЦИЮ
        if not self.position:
            # Условия для покупки: RSI перепродан + цена выше быстрой SMA
            if (rsi_value < self.params.rsi_oversold and 
                current_price > self.sma_fast[0]):
                
                # Размер позиции
                size = (self.broker.cash * self.params.position_size) / current_price
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.log(f'BUY! 😄 Price: {current_price:.2f}, RSI: {rsi_value:.2f}, Size: {size:.6f}')

        # ВЫХОД ИЗ ПОЗИЦИИ
        elif self.position and self.entry_price:
            try:
                profit_pct = (current_price - self.entry_price) / self.entry_price
                
                # Условия для продажи
                sell_conditions = [
                    # RSI перекуплен
                    rsi_value > self.params.rsi_overbought,
                    # Цена ниже медленной SMA (тренд развернулся)
                    current_price < self.sma_slow[0],
                    # Стоп-лосс
                    profit_pct < -self.params.stop_loss,
                    # Тейк-профит
                    profit_pct > self.params.take_profit
                ]
                
                if any(sell_conditions):
                    self.order = self.close()
                    
                    # Определяем причину продажи
                    if profit_pct < -self.params.stop_loss:
                        reason = f"STOP LOSS 😭"
                    elif profit_pct > self.params.take_profit:
                        reason = f"TAKE PROFIT 🎉"
                    elif rsi_value > self.params.rsi_overbought:
                        reason = f"RSI OVERBOUGHT 📈"
                    else:
                        reason = f"TREND DOWN 📉"
                    
                    self.log(f'SELL! {reason} Price: {current_price:.2f}, '
                            f'Profit: {profit_pct*100:.2f}%, RSI: {rsi_value:.2f}')
                    
                    self.entry_price = None
                    
            except (ZeroDivisionError, TypeError):
                pass

    def notify_order(self, order):
        """Обработка ордеров"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY COMPLETED: Price: {order.executed.price:.2f}, '
                        f'Cost: ${order.executed.value:.2f}')
            else:
                self.log(f'SELL COMPLETED: Price: {order.executed.price:.2f}, '
                        f'Value: ${order.executed.value:.2f}')
                        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Failed: {order.getstatusname()}')
        
        self.order = None

    def notify_trade(self, trade):
        """Обработка закрытых сделок"""
        if not trade.isclosed:
            return
            
        pnl_pct = (trade.pnl / abs(trade.value)) * 100 if trade.value != 0 else 0
        
        if trade.pnl > 0:
            self.log(f'TRADE WIN! 🚀 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')
        else:
            self.log(f'TRADE LOSS 💸 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')

    def stop(self):
        """Финальная статистика"""
        final_value = self.broker.getvalue()
        self.log(f'LOL Strategy finished! Final Value: ${final_value:.2f} 💰')


class LOLScalpingStrategy(bt.Strategy):
    """
    LOL Scalping Strategy - агрессивная скальпинговая версия
    """
    
    params = (
        ('rsi_period', 7),
        ('rsi_oversold', 35),
        ('rsi_overbought', 65),
        ('ema_fast', 5),
        ('ema_slow', 13),
        ('position_size', 1.0),  # Изменено с 0.5 на 1.0
        ('quick_profit', 0.02),  # 2% быстрая прибыль
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.ema_fast = bt.indicators.EMA(period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(period=self.params.ema_slow)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
        
        self.order = None
        self.entry_price = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: LOL-SCALP - {txt}')

    def next(self):
        if len(self.data) < max(self.params.rsi_period, self.params.ema_slow):
            return
            
        if self.order:
            return

        current_price = self.data.close[0]
        rsi_value = self.rsi[0]

        if not self.position:
            # Быстрые сигналы на вход
            if (self.crossover[0] > 0 and  # EMA пересечение вверх
                rsi_value < self.params.rsi_oversold):
                
                size = (self.broker.cash * self.params.position_size) / current_price
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.log(f'SCALP BUY! ⚡ Price: {current_price:.2f}, RSI: {rsi_value:.2f}')

        elif self.position and self.entry_price:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            
            # Быстрый выход
            if (profit_pct > self.params.quick_profit or  # Быстрая прибыль
                self.crossover[0] < 0 or  # EMA пересечение вниз
                rsi_value > self.params.rsi_overbought):
                
                self.order = self.close()
                self.log(f'SCALP SELL! ⚡ Profit: {profit_pct*100:.2f}%')
                self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


# Дополнительная забавная стратегия
class LOLRandomStrategy(bt.Strategy):
    """
    LOL Random Strategy - стратегия с элементом случайности для экспериментов
    """
    
    params = (
        ('trade_probability', 0.1),  # 10% вероятность сделки
        ('position_size', 1.0),  # Изменено с 0.3 на 1.0
        ('hold_days', 5),  # Держим позицию N дней
    )

    def __init__(self):
        import random
        self.random = random
        self.order = None
        self.days_in_position = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: LOL-RANDOM - {txt}')

    def next(self):
        if self.order:
            return

        if not self.position:
            # Случайный вход в позицию
            if self.random.random() < self.params.trade_probability:
                size = (self.broker.cash * self.params.position_size) / self.data.close[0]
                if size > 0:
                    self.order = self.buy(size=size)
                    self.days_in_position = 0
                    self.log(f'RANDOM BUY! 🎲 Price: {self.data.close[0]:.2f}')
        else:
            self.days_in_position += 1
            # Выходим через N дней или случайно
            if (self.days_in_position >= self.params.hold_days or 
                self.random.random() < 0.2):  # 20% шанс выйти каждый день
                
                self.order = self.close()
                self.log(f'RANDOM SELL! 🎲 Days held: {self.days_in_position}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None