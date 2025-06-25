import backtrader as bt
import backtrader.indicators as btind


class RSIStrategy(bt.Strategy):
    """
    RSI стратегия для backtrader
    
    Сигналы:
    - Покупка: RSI < oversold (перепроданность)
    - Продажа: RSI > overbought (перекупленность)
    """
    
    params = dict(
        rsi_period=14,      # Период для расчета RSI
        oversold=30,        # Уровень перепроданности (сигнал покупки)
        overbought=70,      # Уровень перекупленности (сигнал продажи)
        stake=0.95,         # Доля капитала для торговли (95%)
        printlog=True,      # Печатать логи сделок
    )
    
    def __init__(self):
        """Инициализация индикаторов и переменных"""
        # RSI индикатор
        self.rsi = btind.RSI(
            self.data.close,
            period=self.params.rsi_period
        )
        
        # Переменные для отслеживания состояния
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Сигналы
        self.buy_signal = bt.And(
            self.rsi < self.params.oversold,
            self.rsi(-1) >= self.params.oversold  # Пересечение снизу вверх
        )
        
        self.sell_signal = bt.And(
            self.rsi > self.params.overbought,
            self.rsi(-1) <= self.params.overbought  # Пересечение сверху вниз
        )
        
    def log(self, txt, dt=None, doprint=False):
        """Функция логирования"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
    
    def notify_order(self, order):
        """Уведомления об изменении статуса ордера"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'ПОКУПКА ИСПОЛНЕНА, Цена: {order.executed.price:.2f}, '
                    f'Стоимость: {order.executed.value:.2f}, '
                    f'Комиссия: {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    f'ПРОДАЖА ИСПОЛНЕНА, Цена: {order.executed.price:.2f}, '
                    f'Стоимость: {order.executed.value:.2f}, '
                    f'Комиссия: {order.executed.comm:.2f}'
                )
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Ордер отменен/отклонен')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Уведомления о закрытии позиции"""
        if not trade.isclosed:
            return
        
        self.log(
            f'ОПЕРАЦИЯ ПРИБЫЛЬ, Валовая: {trade.pnl:.2f}, '
            f'Чистая: {trade.pnlcomm:.2f}'
        )
    
    def next(self):
        """Основная логика стратегии"""
        # Проверяем, есть ли pending ордер
        if self.order:
            return
        
        current_rsi = self.rsi[0]
        
        # Если нет позиции
        if not self.position:
            # Сигнал покупки: RSI пересек oversold снизу вверх
            if self.buy_signal:
                # Рассчитываем размер позиции
                size = int((self.broker.getcash() * self.params.stake) / self.data.close[0])
                
                self.log(f'СИГНАЛ ПОКУПКИ, RSI: {current_rsi:.2f}, Цена: {self.data.close[0]:.2f}')
                self.order = self.buy(size=size)
        
        # Если есть позиция
        else:
            # Сигнал продажи: RSI пересек overbought сверху вниз
            if self.sell_signal:
                self.log(f'СИГНАЛ ПРОДАЖИ, RSI: {current_rsi:.2f}, Цена: {self.data.close[0]:.2f}')
                self.order = self.sell(size=self.position.size)
    
    def stop(self):
        """Вызывается в конце бэктеста"""
        self.log(
            f'RSI({self.params.rsi_period}), '
            f'Oversold: {self.params.oversold}, '
            f'Overbought: {self.params.overbought}, '
            f'Конечная стоимость: {self.broker.getvalue():.2f}',
            doprint=True
        )


