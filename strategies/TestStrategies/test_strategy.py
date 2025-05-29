import os
import backtrader as bt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class SimpleMovingAverageStrategy(bt.Strategy):
    """Простая стратегия на скользящих средних"""

    params = (
        ('fast_ma', 10),
        ('slow_ma', 20),
        ('position_size', 0.95),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_ma)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_ma)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        if current_price <= 0:
            return

        if not self.position and self.crossover > 0:
            size = int(self.broker.get_cash() * self.params.position_size / current_price)
            if size > 0:
                self.order = self.buy(size=size)
        elif self.position and self.crossover < 0:
            self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class CSVBacktester:
    """Простой бектестер для CSV файлов"""

    def __init__(self, csv_file: str, initial_cash: float = 100000, commission: float = 0.001):
        if not os.path.isabs(csv_file):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
            csv_file = os.path.join(project_root, csv_file)

        self.csv_file = csv_file
        self.initial_cash = initial_cash
        self.commission = commission

    def load_data(self) -> pd.DataFrame:
        """Загрузка данных из CSV"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Файл {self.csv_file} не найден")

        df = pd.read_csv(self.csv_file)

        # Обработка колонок
        df.columns = df.columns.str.lower().str.strip()
        mapping = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        df = df.rename(columns=mapping)

        # Установка индекса
        date_col = next((col for col in df.columns if any(word in col.lower() for word in ['date', 'time'])), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)

        # Проверка и очистка данных
        required = ['open', 'high', 'low', 'close']
        if 'volume' not in df.columns:
            df['volume'] = 1000

        df = df[required + ['volume']].dropna()
        df = df[(df > 0).all(axis=1)]
        df.sort_index(inplace=True)

        print(f"✅ Загружено {len(df)} записей")
        return df

    def run_backtest(self, strategy_class=SimpleMovingAverageStrategy, **strategy_params):
        """Запуск бектестирования"""
        data = self.load_data()

        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=data))
        cerebro.addstrategy(strategy_class, **strategy_params)
        cerebro.broker.set_cash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        final_value = cerebro.broker.get_value()

        # Статистика
        trade_analysis = results[0].analyzers.trades.get_analysis()
        total_trades = getattr(getattr(trade_analysis, 'total', None), 'total', 0)
        won_trades = getattr(getattr(trade_analysis, 'won', None), 'total', 0)
        return_pct = ((final_value - self.initial_cash) / self.initial_cash) * 100

        result = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'profit_loss': final_value - self.initial_cash,
            'return_pct': return_pct,
            'total_trades': total_trades,
            'won_trades': won_trades
        }

        print(f"💰 P&L: ${result['profit_loss']:+,.2f} ({result['return_pct']:+.2f}%)")
        print(f"🔄 Сделок: {total_trades} (выигрышей: {won_trades})")

        return result


def main():
    """Тестирование стратегии"""
    CSV_FILE = "CryptoTrade/data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    print("🚀 ТЕСТИРОВАНИЕ СТРАТЕГИИ СКОЛЬЗЯЩИХ СРЕДНИХ")
    print("=" * 50)

    try:
        backtester = CSVBacktester(csv_file=CSV_FILE, initial_cash=100000, commission=0.001)
        result = backtester.run_backtest(
            strategy_class=SimpleMovingAverageStrategy,
            fast_ma=10, slow_ma=20, position_size=0.95
        )

        win_rate = (result['won_trades'] / max(result['total_trades'], 1) * 100)
        print(f"\n📊 РЕЗУЛЬТАТЫ:")
        print(f"📈 Win Rate: {win_rate:.1f}%")
        print(f"💰 Прибыль: {result['return_pct']:+.2f}%")
        print(f"🔄 Всего сделок: {result['total_trades']}")
        print("\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()