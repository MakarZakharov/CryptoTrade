import os
import sys
import backtrader as bt
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../../strategies/TestStrategies'))
from test_strategy import OptimizedBTCStrategy

class StrategyRunner:
    def __init__(self, initial_cash=100000, commission=0.001):
        self.initial_cash = initial_cash
        self.commission = commission
        self.csv_path = os.path.join(os.path.dirname(__file__), "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")

    def run_strategy(self, strategy_class=OptimizedBTCStrategy, **strategy_params):
        """Запуск стратегії з повним аналізом"""

        # Завантаження даних
        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.dropna()

        # Налаштування Cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **strategy_params)
        cerebro.broker.set_cash(self.initial_cash)
        cerebro.broker.setcommission(self.commission)

        # Аналізатори
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        cerebro.adddata(bt.feeds.PandasData(dataname=df))

        print(f"💰 Стартовий капітал: ${self.initial_cash:,}")
        print(f"🤖 Стратегія: {strategy_class.__name__}")
        print(f"📊 Період: {df.index[0].date()} - {df.index[-1].date()} ({len(df)} днів)")

        # Запуск
        results = cerebro.run()
        final_value = cerebro.broker.get_value()

        # Аналіз
        self._analyze_results(results[0], final_value, df)
        return results, final_value

    def _analyze_results(self, strategy, final_value, df):
        """Повний аналіз результатів"""

        # Основні метрики
        profit = final_value - self.initial_cash
        roi = (profit / self.initial_cash) * 100
        years = len(df) / 365.25
        annual_return = ((final_value / self.initial_cash) ** (1/years) - 1) * 100

        # Торгова статистика
        trades = strategy.analyzers.trades.get_analysis()
        total_trades = getattr(trades.get('total', {}), 'total', 0) or 0
        won_trades = getattr(trades.get('won', {}), 'total', 0) or 0
        win_rate = (won_trades / max(total_trades, 1)) * 100

        won_pnl = getattr(trades.get('won', {}), 'pnl', {}).get('total', 0) or 0
        lost_pnl = abs(getattr(trades.get('lost', {}), 'pnl', {}).get('total', 0) or 0)
        profit_factor = won_pnl / max(lost_pnl, 1)

        # Ризики
        dd = strategy.analyzers.drawdown.get_analysis()
        max_drawdown = dd.get('max', {}).get('drawdown', 0) or 0
        sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

        # Порівняння з HODL
        btc_roi = ((df.iloc[-1]['close'] / df.iloc[0]['close']) - 1) * 100
        alpha = roi - btc_roi

        # Результати
        print(f"\n📈 РЕЗУЛЬТАТИ:")
        print(f"💵 Кінцевий капітал: ${final_value:,.0f}")
        print(f"💰 Прибуток: ${profit:+,.0f}")
        print(f"📊 ROI: {roi:+.1f}%")
        print(f"📅 Річна прибутковість: {annual_return:.1f}%")
        print(f"⚡ Sharpe Ratio: {sharpe:.2f}")

        print(f"\n🎯 ТОРГІВЛЯ:")
        print(f"Угод: {total_trades} | Точність: {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Макс. просадка: {max_drawdown:.1f}%")

        print(f"\n📋 ПОРІВНЯННЯ:")
        print(f"Bitcoin HODL: {btc_roi:+.1f}%")
        print(f"Стратегія: {roi:+.1f}%")
        print(f"Альфа: {alpha:+.1f}%")

        # Оцінка
        if roi >= 1000:
            print(f"\n🎉 ЦІЛЬ ДОСЯГНУТА! ROI {roi:.1f}% ≥ 1000%")
        elif roi > btc_roi * 1.5:
            print(f"\n🔥 Відмінно! Перевершили Bitcoin в 1.5+ рази")
        elif roi > btc_roi:
            print(f"\n✅ Добре! Перевершили Bitcoin")
        else:
            print(f"\n⚠️ Потрібні покращення")

def run_backtest(**params):
    """Швидкий запуск"""
    return StrategyRunner().run_strategy(**params)

if __name__ == '__main__':
    run_backtest()
