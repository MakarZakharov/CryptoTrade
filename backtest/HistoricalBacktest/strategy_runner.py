# backtest_runner.py - Спрощена версія

import backtrader as bt
import pandas as pd
from CryptoTrade.strategies.TestStrategies.test_strategy import RSI_EMA_BBands_ATR_Strategy


class SimpleBacktester:
    """Простий клас для бектестування торгових стратегій"""

    def __init__(self, cash=100000, commission=0.001):
        self.cash = cash
        self.commission = commission
        self.cerebro = bt.Cerebro()

    def load_data(self, csv_path):
        """Завантажуємо дані з CSV файлу"""
        print(f"Завантажуємо дані з {csv_path}...")

        # Читаємо CSV
        df = pd.read_csv(csv_path)

        # Конвертуємо дату
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Створюємо дата-фід для backtrader
        data = bt.feeds.PandasData(dataname=df)
        self.cerebro.adddata(data)

        print(f"Завантажено {len(df)} записів")
        return True

    def setup_strategy(self, strategy_class, **params):
        """Додаємо стратегію з параметрами"""
        if params:
            self.cerebro.addstrategy(strategy_class, **params)
        else:
            self.cerebro.addstrategy(strategy_class)

    def setup_broker(self):
        """Налаштовуємо брокера"""
        self.cerebro.broker.set_cash(self.cash)
        self.cerebro.broker.setcommission(commission=self.commission)

    def add_analyzers(self):
        """Додаємо аналізатори для збору статистики"""
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    def run(self):
        """Запускаємо бектест"""
        print("Початок бектестування...")
        results = self.cerebro.run()
        return results[0]  # Повертаємо першу стратегію

    def print_results(self, strategy):
        """Виводимо результати"""
        # Отримуємо дані з аналізаторів
        trades = strategy.analyzers.trades.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        sharpe = strategy.analyzers.sharpe.get_analysis()

        # Розраховуємо основні метрики
        final_value = self.cerebro.broker.getvalue()
        profit_pct = ((final_value - self.cash) / self.cash) * 100

        # Дані про угоди
        total_trades = trades.total.total if 'total' in trades else 0
        won_trades = trades.won.total if 'won' in trades else 0
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        # Виводимо результати
        print("=" * 50)
        print("РЕЗУЛЬТАТИ БЕКТЕСТУ")
        print("=" * 50)
        print(f"Початковий капітал:    ${self.cash:,.2f}")
        print(f"Фінальний капітал:     ${final_value:,.2f}")
        print(f"Прибуток:              {profit_pct:+.2f}%")
        print(f"Всього угод:           {total_trades}")
        print(f"Виграшних угод:        {won_trades}")
        print(f"Відсоток виграшів:     {win_rate:.1f}%")

        # Максимальне просідання
        if 'max' in drawdown:
            print(f"Макс. просідання:      {drawdown.max.drawdown:.2f}%")

        # Коефіцієнт Шарпа
        if sharpe and 'sharperatio' in sharpe:
            print(f"Коефіцієнт Шарпа:      {sharpe['sharperatio']:.3f}")

        print("=" * 50)


def main():
    """Основна функція"""

    # Налаштування
    DATA_FILE = "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    STARTING_CASH = 100000

    # Параметри нової стратегії
    strategy_params = {
        "rsi_period": 14,
        "ema_period": 20,
        "bb_period": 20,
        "bb_devfactor": 2.0,
        "atr_period": 14,
        "atr_multiplier_sl": 2.0,
        "atr_multiplier_tp": 4.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    }

    try:
        # Створюємо бектестер
        backtester = SimpleBacktester(cash=STARTING_CASH)

        # Завантажуємо дані
        backtester.load_data(DATA_FILE)

        # Налаштовуємо стратегію
        backtester.setup_strategy(RSI_EMA_BBands_ATR_Strategy, **strategy_params)

        # Налаштовуємо брокера і аналізатори
        backtester.setup_broker()
        backtester.add_analyzers()

        # Запускаємо бектест
        result = backtester.run()

        # Виводимо результати
        backtester.print_results(result)

    except Exception as e:
        print(f"Помилка: {e}")


if __name__ == '__main__':
    main()