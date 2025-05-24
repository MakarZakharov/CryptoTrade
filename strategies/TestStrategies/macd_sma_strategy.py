# macd_sma_backtest.py

import backtrader as bt
import pandas as pd

# === Стратегія ===
class MACD_SMA_Strategy(bt.Strategy):
    params = (
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("sma_fast", 10),
        ("sma_slow", 50),
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.sma_slow)

        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.params.macd_fast,
                                       period_me2=self.params.macd_slow)
        self.macd_signal = self.macd.signal
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.macd.macd[0] > self.macd.signal[0] and \
               self.macd.macd[-1] <= self.macd.signal[-1] and \
               self.sma_fast[0] > self.sma_slow[0]:
                self.order = self.buy()
        else:
            if self.macd.macd[0] < self.macd.signal[0] and \
               self.macd.macd[-1] >= self.macd.signal[-1] and \
               self.sma_fast[0] < self.sma_slow[0]:
                self.order = self.sell()

# === Завантаження CSV ===
class PandasData(bt.feeds.PandasData):
    params = (
        ('date', 'timestamp'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )

# === Основний запуск ===
if __name__ == '__main__':
    # Завантаження CSV
    df = pd.read_csv("../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    data = PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MACD_SMA_Strategy)
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.1)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("Початковий баланс:", cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print("Фінальний баланс:", cerebro.broker.getvalue())
    print("Sharpe Ratio:", strat.analyzers.sharpe.get_analysis())
    print("Trade Analysis:", strat.analyzers.trades.get_analysis())

    cerebro.plot()

