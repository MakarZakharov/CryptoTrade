import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Добавляем путь к стратегиям
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from strategies.TestStrategies.RSI_SMA_Strategy import RSI_SMA_Strategy


class SimpleBacktester:
    """
    Простой бэктестер для RSI_SMA_Strategy
    """
    
    def __init__(self, initial_cash=10000, commission=0.001):
        self.initial_cash = initial_cash
        self.commission = commission

    def load_data(self):
        """Загрузка данных"""
        data_path = os.path.join(os.path.dirname(__file__), "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")

        data = pd.read_csv(data_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        data = data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }).dropna().sort_index()

        return bt.feeds.PandasData(
            dataname=data, datetime=None, open='Open', high='High',
            low='Low', close='Close', volume='Volume', openinterest=None
        )

    def run_backtest(self):
        """Запуск бэктеста"""
        # Параметры стратегии
        params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'sma_fast': 10,
            'sma_slow': 20,
            'position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'log_enabled': False
        }

        # Настройка Cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(RSI_SMA_Strategy, **params)
        cerebro.adddata(self.load_data())
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # Анализаторы
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        print("🚀 Запуск бэктеста...")
        print(f"Начальный капитал: ${self.initial_cash:,}")

        # Запуск
        results = cerebro.run()
        result = results[0]

        # Результаты
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100

        print(f"\n📊 РЕЗУЛЬТАТЫ:")
        print(f"Финальный капитал: ${final_value:,.2f}")
        print(f"Прибыль: ${final_value - self.initial_cash:,.2f}")
        print(f"Доходность: {total_return:.2f}%")

        # Анализ сделок
        trades = result.analyzers.trades.get_analysis()
        if 'total' in trades and trades.total.total > 0:
            total_trades = trades.total.total
            won_trades = trades.won.total
            win_rate = (won_trades / total_trades) * 100
            print(f"Сделок: {total_trades} | Выигрышных: {win_rate:.1f}%")

        # Sharpe и просадка
        sharpe = result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        drawdown = result.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Макс. просадка: {drawdown:.2f}%")

        # График
        print("\n📈 Показ графика...")
        cerebro.plot(figsize=(15, 8), style='candlestick')
        plt.show()


if __name__ == "__main__":
    backtest = SimpleBacktester()
    backtest.run_backtest()