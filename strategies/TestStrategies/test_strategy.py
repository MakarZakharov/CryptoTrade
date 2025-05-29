import os
import backtrader as bt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class ProfitableBTCStrategy(bt.Strategy):
    """Прибуткова BTC стратегія на основі аналізу історичних даних"""

    params = (
        # Оптимізовані параметри для прибутковості
        ('ema_fast', 12),        # Швидка EMA
        ('ema_slow', 26),        # Повільна EMA
        ('rsi_period', 14),      # RSI період
        ('rsi_oversold', 25),    # RSI перепроданість
        ('rsi_overbought', 75),  # RSI перекупленість
        ('atr_period', 14),      # ATR для волатильності
        ('atr_multiplier', 2.0), # ATR множник для стопів
        ('position_size', 0.8),  # 80% капіталу
        ('trend_filter', 200),   # Довгостроковий тренд (200 днів)
        ('min_volume_ratio', 1.2), # Мінімальний обсяг для входу
    )

    def __init__(self):
        # Основні індикатори
        self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.trend_ema = bt.ind.EMA(period=self.p.trend_filter)

        # Сигнали
        self.ema_crossover = bt.ind.CrossOver(self.ema_fast, self.ema_slow)
        self.volume_sma = bt.ind.SMA(self.data.volume, period=20)

        # Стан
        self.entry_price = 0
        self.stop_price = 0
        self.order = None
        self.days_in_position = 0

    def next(self):
        if self.order:
            return

        price = self.data.close[0]
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]

        # Лічильник днів у позиції
        if self.position:
            self.days_in_position += 1
        else:
            self.days_in_position = 0

        # Оновлення трейлінг стопа
        if self.position:
            new_stop = price - (self.atr[0] * self.p.atr_multiplier)
            if new_stop > self.stop_price:
                self.stop_price = new_stop

        # УМОВИ ДЛЯ КУПІВЛІ (тільки при бичачому тренді)
        if (not self.position and
            price > self.trend_ema[0] and  # Над довгостроковим трендом
            self.ema_crossover > 0 and     # EMA crossover вгору
            self.rsi < self.p.rsi_overbought and  # RSI не перекуплений
            current_volume > avg_volume * self.p.min_volume_ratio):  # Високий обсяг

            size = (self.broker.cash * self.p.position_size) / price
            self.order = self.buy(size=size)
            self.entry_price = price
            self.stop_price = price - (self.atr[0] * self.p.atr_multiplier)
            self.days_in_position = 0
            print(f"📈 КУПІВЛЯ: ${price:.2f}, RSI: {self.rsi[0]:.1f}, EMA тренд: ✅")

        # УМОВИ ДЛЯ ПРОДАЖУ
        elif self.position:
            profit_pct = (price - self.entry_price) / self.entry_price

            # Вихід при стоп-лосі (волатильність-базований)
            if price <= self.stop_price:
                self.order = self.close()
                print(f"🛑 СТОП-ЛОС: ${price:.2f}, Прибуток: {profit_pct*100:+.1f}%")

            # Вихід при перекупленості + негативній дивергенції EMA
            elif (self.rsi > self.p.rsi_overbought and
                  self.ema_fast < self.ema_slow and
                  self.days_in_position > 3):  # Мінімум 3 дні тримати
                self.order = self.close()
                print(f"📉 RSI ВИХІД: ${price:.2f}, Прибуток: {profit_pct*100:+.1f}%")

            # Вихід при розвороті тренду (EMA crossover вниз)
            elif (self.ema_crossover < 0 and
                  profit_pct > 0.02 and  # Мінімум 2% прибуток
                  self.days_in_position > 5):  # Мінімум 5 днів тримати
                self.order = self.close()
                print(f"🔄 ТРЕНД РОЗВОРОТ: ${price:.2f}, Прибуток: {profit_pct*100:+.1f}%")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class AdvancedBacktester:
    """Покращений бектестер з кращою аналітикою"""

    def __init__(self, csv_path, cash=100000):
        self.csv_path = csv_path
        self.cash = cash

    def load_data(self):
        if not os.path.isabs(self.csv_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.csv_path = os.path.join(base_dir, "../../../", self.csv_path)

        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df.dropna()

    def test_strategy(self, strategy_class=ProfitableBTCStrategy, **params):
        cerebro = bt.Cerebro()
        data = self.load_data()
        cerebro.adddata(bt.feeds.PandasData(dataname=data))
        cerebro.addstrategy(strategy_class, **params)
        cerebro.broker.set_cash(self.cash)
        cerebro.broker.setcommission(0.001)

        # Розширені аналізатори
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        print(f"🚀 Тестування {strategy_class.__name__}")
        results = cerebro.run()
        final_value = cerebro.broker.get_value()

        # Детальна статистика
        strategy = results[0]
        trades = strategy.analyzers.trades.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()

        total_trades = getattr(trades.get('total', {}), 'total', 0)
        won_trades = getattr(trades.get('won', {}), 'total', 0)

        profit = final_value - self.cash
        roi = (profit / self.cash) * 100
        win_rate = (won_trades / max(total_trades, 1)) * 100

        # Додаткові метрики
        avg_win = getattr(trades.get('won', {}), 'pnl', {}).get('average', 0) or 0
        avg_loss = getattr(trades.get('lost', {}), 'pnl', {}).get('average', 0) or 0
        profit_factor = abs(avg_win * won_trades / max(abs(avg_loss * (total_trades - won_trades)), 1)) if avg_loss != 0 else float('inf')
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0) or 0

        print(f"💰 Прибуток: ${profit:+,.0f} ({roi:+.1f}%)")
        print(f"🎯 Угоди: {total_trades}, Прибуткових: {won_trades} ({win_rate:.1f}%)")
        print(f"📊 Profit Factor: {profit_factor:.2f}")
        print(f"📉 Max Drawdown: {max_drawdown:.1f}%")

        return {
            'profit': profit,
            'roi': roi,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'params': params
        }


def optimize_profitable_strategy():
    """Оптимізація для максимальної прибутковості"""

    backtester = AdvancedBacktester("CryptoTrade/data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")

    best_result = {'roi': -100}

    # Конфігурації орієнтовані на прибуток
    configs = [
        # Консервативна (менше угод, більша точність)
        {'ema_fast': 12, 'ema_slow': 26, 'rsi_overbought': 75, 'atr_multiplier': 2.5, 'min_volume_ratio': 1.5},

        # Збалансована (оптимальний баланс)
        {'ema_fast': 9, 'ema_slow': 21, 'rsi_overbought': 70, 'atr_multiplier': 2.0, 'min_volume_ratio': 1.2},

        # Агресивна (більше угод)
        {'ema_fast': 8, 'ema_slow': 17, 'rsi_overbought': 65, 'atr_multiplier': 1.8, 'min_volume_ratio': 1.0},
    ]

    print("🔍 ОПТИМІЗАЦІЯ ДЛЯ МАКСИМАЛЬНОЇ ПРИБУТКОВОСТІ")
    print("=" * 50)

    for i, config in enumerate(configs, 1):
        print(f"\n📊 Конфігурація {i}/3:")
        result = backtester.test_strategy(**config)

        # Вибираємо за комбінацією ROI та Profit Factor
        score = result['roi'] + (result['profit_factor'] * 10) - (result['max_drawdown'] * 2)
        if score > best_result.get('score', -1000):
            best_result = result
            best_result['score'] = score
            print("⭐ Новий лідер!")

    print(f"\n🏆 НАЙКРАЩА ПРИБУТКОВА СТРАТЕГІЯ:")
    print(f"📈 ROI: {best_result['roi']:+.1f}%")
    print(f"🎯 Win Rate: {best_result['win_rate']:.1f}%")
    print(f"⚡ Profit Factor: {best_result['profit_factor']:.2f}")
    print(f"⚙️ Параметри: {best_result['params']}")

    return best_result


def main():
    """Запуск прибуткової BTC стратегії"""

    print("🚀 ПРИБУТКОВА BTC СТРАТЕГІЯ V2.0")
    print("=" * 50)

    try:
        best_config = optimize_profitable_strategy()

        print(f"\n🎯 ФІНАЛЬНИЙ ТЕСТ ПРИБУТКОВОЇ СТРАТЕГІЇ")
        print("=" * 50)

        backtester = AdvancedBacktester("CryptoTrade/data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")
        final_result = backtester.test_strategy(**best_config['params'])

        if final_result['roi'] > 0:
            print(f"\n🎉 УСПІХ! ПРИБУТКОВА СТРАТЕГІЯ ЗНАЙДЕНА!")
            print(f"💰 Очікуваний прибуток: {final_result['roi']:+.1f}% за період")
            print(f"🎯 Точність: {final_result['win_rate']:.1f}%")
            print(f"⚡ Ефективність: {final_result['profit_factor']:.2f}x")
        else:
            print(f"\n⚠️ Потрібне додаткове налаштування")
            print(f"📊 Поточний результат: {final_result['roi']:+.1f}%")

    except Exception as e:
        print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    main()