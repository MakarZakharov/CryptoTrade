import backtrader as bt
from datetime import datetime

# ======================
#   1) СТРАТЕГИЯ: ТОРГОВЛЯ КАЖДЫЙ БАР
# ======================
class EveryBarFullCapital(bt.Strategy):
    def next(self):
        price = self.data.close[0]
        cash  = self.broker.getcash()
        size  = cash / price  # использовать весь доступный капитал

        if not self.position:
            # открыть длинную позицию на весь капитал
            self.buy(size=size)
        else:
            # закрыть позицию полностью
            self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status == order.Completed:
            dt   = self.data.datetime.date(0).isoformat()
            typ  = "BUY" if order.isbuy() else "SELL"
            price = order.executed.price
            size  = order.executed.size
            print(f"{dt} ▶ {typ} EXECUTED @ {price:.2f}, size={size:.6f}")

    def notify_trade(self, trade):
        if trade.isclosed:
            dt = self.data.datetime.date(0).isoformat()
            print(f"{dt} ▶ TRADE CLOSED  Gross PnL={trade.pnl:.2f}, Net PnL={trade.pnlcomm:.2f}")


# ======================
#   2) БЭКТЕСТ
# ======================
if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EveryBarFullCapital)

    # Загружаем данные из CSV
    data = bt.feeds.GenericCSVData(
        dataname='../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv',
        dtformat='%Y-%m-%dT%H:%M:%S',
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        openinterest=-1,
        timeframe=bt.TimeFrame.Days
    )
    cerebro.adddata(data)

    # Настройки брокера
    cerebro.broker.setcash(100000)                 # стартовый капитал
    cerebro.broker.setcommission(commission=0.001) # комиссия 0.1%
    cerebro.broker.set_slippage_perc(perc=0.0005)   # проскальзывание 0.05%
    cerebro.broker.set_coc(True)                   # CHEAT-ON-CLOSE: ордера исполнять по закрытию бара

    # Анализаторы метрик
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown,     _name="drawdown")

    # Запуск
    start_capital = cerebro.broker.getvalue()
    strat = cerebro.run()[0]
    end_capital   = cerebro.broker.getvalue()
    growth_pct    = (end_capital / start_capital - 1) * 100

    # Сбор статистики
    ta    = strat.analyzers.trades.get_analysis()
    won   = ta.get('won',  {}).get('total', 0)
    lost  = ta.get('lost', {}).get('total', 0)
    total = won + lost
    win_rate = (won / total * 100) if total else 0
    max_dd = strat.analyzers.drawdown.get_analysis().max.drawdown

    # Вывод результатов
    print("\n===== РЕЗУЛЬТАТЫ БЭКТЕСТА =====")
    print(f"Win rate:         {win_rate:.2f}%")
    print(f"Start capital:    ${start_capital:,.2f}")
    print(f"End capital:      ${end_capital:.2f}")
    print(f"Growth:           {growth_pct:.2f}%")
    print(f"Max drawdown:     {max_dd:.2f}%")
    print(f"Number of trades: {total}")
