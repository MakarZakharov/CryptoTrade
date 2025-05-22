import backtrader as bt

class SmaCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 50),
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.slow_period)
        self.cross = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        self.trades = []

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt} {txt}')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'Сделка закрыта: П/У {trade.pnl:.2f} ({trade.pnlcomm:.2f} с учетом комиссии)')
            self.trades.append(trade.pnlcomm)

    def next(self):
        if not self.position:
            if self.cross > 0:
                self.log('Покупка')
                self.buy()
        else:
            if self.cross < 0:
                self.log('Продажа')
                self.sell()
