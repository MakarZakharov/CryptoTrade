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


# === Функція для відображення результатів ===
def print_backtest_results(cerebro, results):
    strat = results[0]

    # Отримання аналізаторів
    trade_analyzer = strat.analyzers.trades.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()

    # Базові метрики
    starting_capital = 100000  # Початковий капітал
    final_capital = cerebro.broker.getvalue()
    capital_gain_pct = ((final_capital - starting_capital) / starting_capital) * 100

    # Торгові метрики
    total_trades = trade_analyzer.total.total if 'total' in trade_analyzer else 0
    won_trades = trade_analyzer.won.total if 'won' in trade_analyzer else 0
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    # Максимальний просідання
    max_drawdown = drawdown_analyzer.max.drawdown if 'max' in drawdown_analyzer else 0

    print("=" * 60)
    print("РЕЗУЛЬТАТИ БЕКТЕСТУВАННЯ")
    print("=" * 60)
    print(f"Початковий капітал:        ${starting_capital:,.2f}")
    print(f"Фінальний капітал:         ${final_capital:,.2f}")
    print(f"Приріст капіталу:          {capital_gain_pct:+.2f}%")
    print(f"Максимальне просідання:    {max_drawdown:.2f}%")
    print(f"Кількість транзакцій:      {total_trades}")
    print(f"Процент виграшних угод:    {win_rate:.2f}%")
    print("=" * 60)

    # Додаткова детальна інформація
    if 'won' in trade_analyzer and 'lost' in trade_analyzer:
        won_total = trade_analyzer.won.total
        lost_total = trade_analyzer.lost.total

        print("ДЕТАЛЬНА СТАТИСТИКА:")
        print(f"Виграшних угод:            {won_total}")
        print(f"Програшних угод:           {lost_total}")

        if won_total > 0:
            avg_win = trade_analyzer.won.pnl.average
            print(f"Середній виграш:           ${avg_win:.2f}")

        if lost_total > 0:
            avg_loss = trade_analyzer.lost.pnl.average
            print(f"Середній програш:          ${avg_loss:.2f}")

        print("=" * 60)


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
    cerebro.broker.setcommission(commission=0.001)

    # Додавання аналізаторів
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Запуск бектестування
    results = cerebro.run()

    # Відображення результатів
    print_backtest_results(cerebro, results)

    # Додаткові метрики (опціонально)
    strat = results[0]
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    if 'sharperatio' in sharpe_analysis:
        print(f"Коефіцієнт Шарпа:          {sharpe_analysis['sharperatio']:.4f}")
        print("=" * 60)

    # Побудова графіків
    cerebro.plot()