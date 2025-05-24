# macd_sma_backtest_improved.py

import backtrader as bt
import pandas as pd


# === Покращена стратегія (Варіант 1: OR замість AND для виходу) ===
class MACD_SMA_Strategy_v1(bt.Strategy):
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
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Вхід: MACD бичачий кросовер + SMA тренд вгору
            if (self.macd.macd[0] > self.macd.signal[0] and
                    self.macd.macd[-1] <= self.macd.signal[-1] and
                    self.sma_fast[0] > self.sma_slow[0]):
                self.order = self.buy()
        else:
            # Вихід: MACD ведмежий кросовер OR SMA тренд вниз (OR замість AND)
            if ((self.macd.macd[0] < self.macd.signal[0] and
                 self.macd.macd[-1] >= self.macd.signal[-1]) or
                    self.sma_fast[0] < self.sma_slow[0]):
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


# === Покращена стратегія (Варіант 2: Тільки MACD для сигналів) ===
class MACD_Strategy_v2(bt.Strategy):
    params = (
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("sma_period", 50),
        ("rsi_period", 14),
        ("volume_sma", 20),
        ("stop_loss", 0.08),  # 8% стоп-лосс
        ("take_profit", 0.20),  # 20% тейк-профіт
    )

    def __init__(self):
        # Основні індикатори
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.params.macd_fast,
                                       period_me2=self.params.macd_slow,
                                       period_signal=self.params.macd_signal)

        # Обчислюємо MACD гістограму вручну
        self.macd_histogram = self.macd.macd - self.macd.signal

        # Додаткові фільтри
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_sma)

        # Трендові індикатори
        self.ema_fast = bt.indicators.EMA(self.data.close, period=21)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=55)

        self.order = None
        self.buy_price = None

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]

        # Стоп-лосс і тейк-профіт для існуючих позицій
        if self.position:
            if self.buy_price:
                # Стоп-лосс
                if current_price <= self.buy_price * (1 - self.params.stop_loss):
                    self.order = self.sell()
                    return

                # Тейк-профіт
                if current_price >= self.buy_price * (1 + self.params.take_profit):
                    self.order = self.sell()
                    return

            # Сигнал виходу: MACD ведмежий кросовер з додатковими умовами
            macd_bearish_cross = (self.macd.macd[0] < self.macd.signal[0] and
                                  self.macd.macd[-1] >= self.macd.signal[-1])

            # Додаткові умови для більш надійного виходу
            strong_bearish = (macd_bearish_cross and
                              self.macd_histogram[0] < 0 and
                              self.rsi[0] > 70)  # Перекупленість

            weak_trend = self.ema_fast[0] < self.ema_slow[0]  # EMA тренд вниз

            # Падіння гістограми MACD
            histogram_falling = self.macd_histogram[0] < self.macd_histogram[-1]

            if macd_bearish_cross or strong_bearish or (weak_trend and histogram_falling):
                self.order = self.sell()

        else:
            # Вхідні сигнали з множинними фільтрами

            # Основний MACD сигнал
            macd_bullish_cross = (self.macd.macd[0] > self.macd.signal[0] and
                                  self.macd.macd[-1] <= self.macd.signal[-1])

            # Фільтр тренду
            price_above_sma = current_price > self.sma[0]
            ema_trend_up = self.ema_fast[0] > self.ema_slow[0]

            # MACD гістограма зростає
            macd_histogram_growing = self.macd_histogram[0] > self.macd_histogram[-1]
            macd_histogram_positive = self.macd_histogram[0] > 0

            # RSI фільтр (не перекуплено, але є моментум)
            rsi_filter = 30 < self.rsi[0] < 75

            # Фільтр об'єму (підвищена активність)
            volume_filter = self.data.volume[0] > self.volume_sma[0] * 1.1

            # MACD лінія вище нуля (бичачий моментум)
            macd_positive_momentum = self.macd.macd[0] > 0

            # Комбінуємо всі умови
            strong_entry = (macd_bullish_cross and
                            price_above_sma and
                            ema_trend_up and
                            macd_histogram_growing and
                            rsi_filter and
                            volume_filter)

            medium_entry = (macd_bullish_cross and
                            price_above_sma and
                            macd_positive_momentum and
                            rsi_filter)

            # Простий вхід (якщо складні умови не спрацьовують)
            simple_entry = (macd_bullish_cross and
                            price_above_sma and
                            macd_histogram_positive)

            if strong_entry or medium_entry or simple_entry:
                self.order = self.buy()
                self.buy_price = current_price

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
            if order.isbuy():
                self.buy_price = order.executed.price
            else:
                self.buy_price = None


# === Стратегія з коротшими періодами (Варіант 3) ===
class MACD_SMA_Strategy_v3(bt.Strategy):
    params = (
        ("macd_fast", 8),
        ("macd_slow", 21),
        ("sma_fast", 5),
        ("sma_slow", 20),
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.sma_slow)

        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.params.macd_fast,
                                       period_me2=self.params.macd_slow)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if (self.macd.macd[0] > self.macd.signal[0] and
                    self.macd.macd[-1] <= self.macd.signal[-1] and
                    self.sma_fast[0] > self.sma_slow[0]):
                self.order = self.buy()
        else:
            if (self.macd.macd[0] < self.macd.signal[0] and
                    self.macd.macd[-1] >= self.macd.signal[-1]):
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


def run_strategy(strategy_class, strategy_name):
    print(f"\n=== {strategy_name} ===")

    df = pd.read_csv("../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    data = PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class)
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    starting_cash = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    final_cash = cerebro.broker.getvalue()

    print(f"Початковий баланс: {starting_cash}")
    print(f"Фінальний баланс: {final_cash:.2f}")
    print(f"Приріст капіталу: {(final_cash - starting_cash) / starting_cash * 100:.2f}%")

    trades = strat.analyzers.trades.get_analysis()
    total_trades = getattr(trades.total, 'closed', 0)
    won_trades = getattr(getattr(trades, 'won', {}), 'total', 0)
    lost_trades = getattr(getattr(trades, 'lost', {}), 'total', 0)
    winrate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"Загальна кількість ордерів: {total_trades}")
    print(f"Winrate: {winrate:.2f}%")

    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"Максимальна просадка: {drawdown.max.drawdown:.2f}%")

    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get('sharperatio', 'N/A')
    print(f"Sharpe Ratio: {sharpe_ratio}")


# === Основний запуск ===
if __name__ == '__main__':
    # Тестуємо всі варіанти
    run_strategy(MACD_SMA_Strategy_v1, "Варіант 1: OR для виходу")
    run_strategy(MACD_Strategy_v2, "Варіант 2: Тільки MACD сигнали")
    run_strategy(MACD_SMA_Strategy_v3, "Варіант 3: Коротші періоди")

    print("\n" + "=" * 60)
    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ:")
    print("=" * 60)

    # Запускаємо знову для детального порівняння
    results_comparison = []

    strategies = [
        (MACD_SMA_Strategy_v1, "Варіант 1: OR для виходу"),
        (MACD_Strategy_v2, "Варіант 2: Тільки MACD сигнали"),
        (MACD_SMA_Strategy_v3, "Варіант 3: Коротші періоди")
    ]

    for strategy_class, name in strategies:
        df = pd.read_csv("../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        data = PandasData(dataname=df)
        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(strategy_class)
        cerebro.broker.set_cash(100000)
        cerebro.broker.setcommission(commission=0.001)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        starting_cash = cerebro.broker.getvalue()
        results = cerebro.run()
        strat = results[0]
        final_cash = cerebro.broker.getvalue()

        trades = strat.analyzers.trades.get_analysis()
        total_trades = getattr(trades.total, 'closed', 0)
        won_trades = getattr(getattr(trades, 'won', {}), 'total', 0)
        winrate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        profit = (final_cash - starting_cash) / starting_cash * 100

        drawdown = strat.analyzers.drawdown.get_analysis()
        max_dd = drawdown.max.drawdown

        sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', 0)

        results_comparison.append({
            'name': name,
            'profit': profit,
            'trades': total_trades,
            'winrate': winrate,
            'drawdown': max_dd,
            'sharpe': sharpe_ratio
        })

    # Виводимо таблицю порівняння
    print(f"{'Стратегія':<35} {'Прибуток %':<12} {'Угоди':<8} {'Winrate %':<10} {'Просадка %':<12} {'Sharpe':<8}")
    print("-" * 85)

    for result in results_comparison:
        print(
            f"{result['name']:<35} {result['profit']:<12.2f} {result['trades']:<8} {result['winrate']:<10.2f} {result['drawdown']:<12.2f} {result['sharpe']:<8.3f}")

    # Знаходимо найкращу стратегію
    best_strategy = max(results_comparison, key=lambda x: x['sharpe'] if x['sharpe'] != 'N/A' else -999)
    print(f"\nНайкраща стратегія за Sharpe Ratio: {best_strategy['name']}")

    best_profit = max(results_comparison, key=lambda x: x['profit'])
    print(f"Найкраща стратегія за прибутком: {best_profit['name']} ({best_profit['profit']:.2f}%)")

    most_trades = max(results_comparison, key=lambda x: x['trades'])
    print(f"Найактивніша стратегія: {most_trades['name']} ({most_trades['trades']} угод)")