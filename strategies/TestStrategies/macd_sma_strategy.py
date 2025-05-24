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

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

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
    df = pd.read_csv("../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    data = PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MACD_SMA_Strategy)
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% комісія

    # Додаємо аналізатори
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    starting_cash = cerebro.broker.getvalue()
    print("Початковий баланс:", starting_cash)

    results = cerebro.run()
    strat = results[0]

    final_cash = cerebro.broker.getvalue()
    print("Фінальний баланс:", final_cash)
    print("Приріст капіталу: {:.2f}%".format((final_cash - starting_cash) / starting_cash * 100))

    trades = strat.analyzers.trades.get_analysis()

    # Безпечний доступ до total.closed
    total_trades = getattr(trades.total, 'closed', 0)

    # Безпечний доступ до won.total і lost.total
    won_trades = getattr(getattr(trades, 'won', {}), 'total', 0)
    lost_trades = getattr(getattr(trades, 'lost', {}), 'total', 0)

    # Обчислення winrate
    winrate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    print("Загальна кількість ордерів:", total_trades)
    print("Кількість виграшних ордерів:", won_trades)
    print("Кількість програшних ордерів:", lost_trades)
    print("Winrate: {:.2f}%".format(winrate))

    # Максимальна просадка
    drawdown = strat.analyzers.drawdown.get_analysis()
    print("Максимальна просадка: {:.2f}%".format(drawdown.max.drawdown))

    # Sharpe Ratio - FIXED
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get('sharperatio', 'N/A')
    print("Sharpe Ratio:", sharpe_ratio)

    cerebro.plot()