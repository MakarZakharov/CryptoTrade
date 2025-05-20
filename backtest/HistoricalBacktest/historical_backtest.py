

from CryptoTrade.strategies.TestStrategies.first_strategy import SmaCrossStrategy

def run_backtest(strategy, datafile, cash=1000, fast_period=10, slow_period=50):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy, fast_period=fast_period, slow_period=slow_period)

    data = bt.feeds.GenericCSVData(
        dataname=datafile,
        dtformat=('%Y-%m-%d'),  # или подходящий формат!
        datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1, header=0
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(cash)
    print(f'Начальный капитал: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'Финальный капитал: {cerebro.broker.getvalue():.2f}')
    cerebro.plot()

if __name__ == "__main__":
    run_backtest(SmaCrossStrategy, '../../data/your_file.csv', fast_period=10, slow_period=50)
