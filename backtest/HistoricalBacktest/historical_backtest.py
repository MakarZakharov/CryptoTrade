import backtrader as bt
import os
from datetime import datetime
from CryptoTrade.strategies.TestStrategies.first_strategy import SmaCrossStrategy

def max_drawdown(values):
    max_val, drawdown = values[0], 0
    for v in values:
        if v > max_val:
            max_val = v
        dd = (max_val - v) / max_val
        if dd > drawdown:
            drawdown = dd
    return drawdown

def run_backtest(
    strategy,
    datafile,
    cash=1000,
    fast_period=10,
    slow_period=50,
    commission=0.00075
):
    print("Текущая рабочая директория:", os.getcwd())
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addstrategy(strategy, fast_period=fast_period, slow_period=slow_period)

    data = bt.feeds.GenericCSVData(
        dataname=datafile,
        dtformat=('%Y-%m-%dT%H:%M:%S'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        header=0
    )
    cerebro.adddata(data)

    print(f"Стартовый капитал: {cash}")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    profit = final_value - cash
    print(f"Конечный капитал: {final_value:.2f}")
    print(f"Общая прибыль: {profit:.2f}")
    print(f"Доходность: {profit/cash*100:.2f}%")

    # Статистика по сделкам
    all_trades = getattr(strat, 'trades', [])
    print(f"Количество сделок: {len(all_trades)}")
    if all_trades:
        print(f"Средняя прибыль на сделку: {sum(all_trades)/len(all_trades):.2f}")
        print(f"Лучшая сделка: {max(all_trades):.2f}")
        print(f"Худшая сделка: {min(all_trades):.2f}")
        # Для расчета просадки
        values = [cash]
        for t in all_trades:
            values.append(values[-1] + t)
        print(f"Максимальная просадка: {max_drawdown(values)*100:.2f}%")

    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    datafile = "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    run_backtest(
        SmaCrossStrategy,
        datafile,
        cash=1000,
        fast_period=10,
        slow_period=50,
        commission=0.00075
    )

