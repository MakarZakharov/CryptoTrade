import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Импортируем нашу стратегию
from CryptoTrade.strategies.TestStrategies.RSI_SMA_Strategy import RSI_SMA_Strategy


class BacktestRunner:
    """
    Класс для запуска и анализа бэктестов торговых стратегий
    """

    def __init__(self, initial_cash=10000, commission=0.001):
        """
        Инициализация бэктестера

        Args:
            initial_cash (float): Начальный капитал
            commission (float): Комиссия брокера (0.1% = 0.001)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = None
        self.results = None

    def load_data_from_csv(self, csv_file, datetime_col='datetime',
                           open_col='open', high_col='high', low_col='low',
                           close_col='close', volume_col='volume',
                           datetime_format='%Y-%m-%d %H:%M:%S'):
        """
        Загрузка данных из CSV файла

        Args:
            csv_file (str): Путь к CSV файлу
            datetime_col (str): Название колонки с датой/временем
            open_col (str): Название колонки с ценой открытия
            high_col (str): Название колонки с максимальной ценой
            low_col (str): Название колонки с минимальной ценой
            close_col (str): Название колонки с ценой закрытия
            volume_col (str): Название колонки с объемом
            datetime_format (str): Формат даты/времени
        """
        try:
            print(f"Загрузка данных из CSV файла: {csv_file}")

            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Файл {csv_file} не найден")

            # Загружаем CSV
            data = pd.read_csv(csv_file)

            print(f"Исходные колонки в CSV: {list(data.columns)}")

            # Проверяем наличие необходимых колонок
            required_cols = [datetime_col, open_col, high_col, low_col, close_col]
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

            # Преобразуем datetime колонку
            if datetime_col in data.columns:
                data[datetime_col] = pd.to_datetime(data[datetime_col], format=datetime_format)
                data.set_index(datetime_col, inplace=True)

            # Переименовываем колонки для backtrader
            column_mapping = {
                open_col: 'Open',
                high_col: 'High',
                low_col: 'Low',
                close_col: 'Close'
            }

            if volume_col and volume_col in data.columns:
                column_mapping[volume_col] = 'Volume'

            data = data.rename(columns=column_mapping)

            # Убеждаемся, что данные отсортированы по дате
            data = data.sort_index()

            print(f"Загружено {len(data)} записей с {data.index[0]} по {data.index[-1]}")
            print(f"Финальные колонки: {list(data.columns)}")

            # Преобразуем в формат backtrader
            data_bt = bt.feeds.PandasData(
                dataname=data,
                datetime=None,  # Используем индекс как datetime
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume' if 'Volume' in data.columns else None,
                openinterest=None
            )

            return data_bt

        except Exception as e:
            print(f"Ошибка при загрузке CSV файла: {e}")
            return None

    def load_data_binance_csv(self, csv_file):
        """
        Загрузка данных из CSV файла Binance (стандартный формат)

        Args:
            csv_file (str): Путь к CSV файлу от Binance
        """
        return self.load_data_from_csv(
            csv_file=csv_file,
            datetime_col='timestamp',  # или 'open_time' в зависимости от формата
            open_col='open',
            high_col='high',
            low_col='low',
            close_col='close',
            volume_col='volume',
            datetime_format='%Y-%m-%d %H:%M:%S'
        )

    def setup_cerebro(self, strategy_class=RSI_SMA_Strategy, **strategy_params):
        """
        Настройка движка backtrader

        Args:
            strategy_class: Класс стратегии для тестирования
            **strategy_params: Параметры стратегии
        """
        self.cerebro = bt.Cerebro()

        # Добавляем стратегию
        self.cerebro.addstrategy(strategy_class, **strategy_params)

        # Настраиваем брокера
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)

        # Добавляем анализаторы
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    def run_backtest(self, data_feed):
        """
        Запуск бэктеста

        Args:
            data_feed: Данные для тестирования
        """
        if not self.cerebro:
            raise ValueError("Сначала вызовите setup_cerebro()")

        # Добавляем данные
        self.cerebro.adddata(data_feed)

        print(f'Начальный капитал: {self.cerebro.broker.getvalue():.2f}')

        # Запускаем бэктест
        self.results = self.cerebro.run()

        print(f'Финальный капитал: {self.cerebro.broker.getvalue():.2f}')

        return self.results[0]

    def analyze_results(self, result):
        """
        Анализ результатов бэктеста

        Args:
            result: Результат выполнения стратегии
        """
        print("\n" + "=" * 50)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ БЭКТЕСТА")
        print("=" * 50)

        # Основные метрики
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100

        print(f"Начальный капитал: ${self.initial_cash:,.2f}")
        print(f"Финальный капитал: ${final_value:,.2f}")
        print(f"Общая доходность: {total_return:.2f}%")

        # Анализ сделок
        trade_analyzer = result.analyzers.trades.get_analysis()

        if 'total' in trade_analyzer and trade_analyzer.total.total > 0:
            print(f"\nАНАЛИЗ СДЕЛОК:")
            print(f"Всего сделок: {trade_analyzer.total.total}")
            print(f"Выигрышных: {trade_analyzer.won.total}")
            print(f"Проигрышных: {trade_analyzer.lost.total}")
            print(f"Процент выигрышных: {trade_analyzer.won.total / trade_analyzer.total.total * 100:.1f}%")

            if trade_analyzer.won.total > 0:
                print(f"Средняя прибыль: ${trade_analyzer.won.pnl.average:.2f}")
            if trade_analyzer.lost.total > 0:
                print(f"Средний убыток: ${trade_analyzer.lost.pnl.average:.2f}")

        # Коэффициент Шарпа
        sharpe = result.analyzers.sharpe.get_analysis()
        if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
            print(f"\nКоэффициент Шарпа: {sharpe['sharperatio']:.3f}")

        # Максимальная просадка
        drawdown = result.analyzers.drawdown.get_analysis()
        print(f"Максимальная просадка: {drawdown['max']['drawdown']:.2f}%")

        # SQN (System Quality Number)
        sqn = result.analyzers.sqn.get_analysis()
        if 'sqn' in sqn and sqn['sqn'] is not None:
            print(f"SQN: {sqn['sqn']:.2f}")

    def plot_results(self, figsize=(15, 10)):
        """
        Построение графиков результатов

        Args:
            figsize (tuple): Размер графика
        """
        if not self.cerebro:
            print("Нет данных для построения графика")
            return

        # Стандартный график backtrader
        self.cerebro.plot(figsize=figsize, style='candlestick')
        plt.show()


def main():
    """
    Основная функция для запуска бэктеста с CSV данными
    """
    print("Инициализация бэктестера для высокодоходных стратегий...")

    # Создаем экземпляр бэктестера
    runner = BacktestRunner(
        initial_cash=10000,  # $10,000 начальный капитал
        commission=0.001  # 0.1% комиссия
    )

    # Загрузка данных из CSV файла
    csv_file_path = "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    print(f"Попытка загрузить данные из: {csv_file_path}")

    # Загружаем данные из CSV
    data = runner.load_data_from_csv(
        csv_file=csv_file_path,
        datetime_col='timestamp',
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
        volume_col='volume',
        datetime_format='%Y-%m-%dT%H:%M:%S'
    )

    if data is None:
        print("Не удалось загрузить данные из CSV файла.")
        return

    # Список стратегий для тестирования
    strategies_to_test = [
        {
            'name': 'Агрессивная Momentum',
            'params': {
                'rsi_period': 14,
                'sma_period': 30,
                'sma_fast': 12,
                'sma_slow': 26,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'position_size': 0.98,
                'printlog': False,
                'strategy_type': 'aggressive_momentum',
                'stop_loss_pct': 0.06,
                'take_profit_pct': 0.20,
                'trailing_stop_pct': 0.10,
                'use_leverage': True,
                'leverage_multiplier': 1.8
            }
        },
        {
            'name': 'Мульти-индикаторная',
            'params': {
                'rsi_period': 10,
                'sma_period': 40,
                'sma_fast': 15,
                'sma_slow': 35,
                'rsi_oversold': 35,  # Увеличиваем с 20 до 35
                'rsi_overbought': 70,  # Снижаем с 80 до 70
                'position_size': 0.95,
                'printlog': False,
                'strategy_type': 'multi_indicator',
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.30,
                'use_leverage': True,
                'leverage_multiplier': 1.5
            }
        },
        {
            'name': 'Прорыв трендов',
            'params': {
                'rsi_period': 14,
                'breakout_period': 15,
                'position_size': 0.99,
                'printlog': False,
                'strategy_type': 'trend_breakout',
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.35,
                'trailing_stop_pct': 0.08,
                'use_leverage': True,
                'leverage_multiplier': 2.0
            }
        },
        {
            'name': 'Динамические риски',
            'params': {
                'rsi_period': 12,
                'sma_period': 25,
                'position_size': 0.90,
                'printlog': False,
                'strategy_type': 'dynamic_risk',
                'stop_loss_pct': 0.07,
                'take_profit_pct': 0.25,
                'trailing_stop_pct': 0.12,
                'use_leverage': True,
                'leverage_multiplier': 1.6
            }
        }
    ]

    best_strategy = None
    best_return = -100

    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ ВЫСОКОДОХОДНЫХ СТРАТЕГИЙ")
    print("="*80)

    for strategy_config in strategies_to_test:
        print(f"\n🚀 Тестирование стратегии: {strategy_config['name']}")
        print("-" * 50)

        # Создаем новый экземпляр для каждой стратегии
        strategy_runner = BacktestRunner(
            initial_cash=10000,
            commission=0.001
        )

        # Загружаем данные заново для каждой стратегии
        strategy_data = strategy_runner.load_data_from_csv(
            csv_file=csv_file_path,
            datetime_col='timestamp',
            open_col='open',
            high_col='high',
            low_col='low',
            close_col='close',
            volume_col='volume',
            datetime_format='%Y-%m-%dT%H:%M:%S'
        )

        # Настраиваем cerebro
        strategy_runner.setup_cerebro(RSI_SMA_Strategy, **strategy_config['params'])

        # Запускаем бэктест
        result = strategy_runner.run_backtest(strategy_data)

        # Анализируем результаты
        final_value = strategy_runner.cerebro.broker.getvalue()
        total_return = (final_value - 10000) / 10000 * 100

        print(f"📊 Результаты стратегии '{strategy_config['name']}':")
        print(f"   💰 Финальный капитал: ${final_value:,.2f}")
        print(f"   📈 Доходность: {total_return:.2f}%")

        if total_return > best_return:
            best_return = total_return
            best_strategy = strategy_config['name']

        strategy_runner.analyze_results(result)

    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    print(f"🏆 Лучшая стратегия: {best_strategy}")
    print(f"💎 Максимальная доходность: {best_return:.2f}%")

    if best_return >= 1000:
        print("🎉 ЦЕЛЬ ДОСТИГНУТА! Доходность превышает 1000%!")
    else:
        print("⚠️  Цель не достигнута. Рекомендации:")
        print("   - Попробуйте изменить параметры стратегий")
        print("   - Рассмотрите другие временные рамки")
        print("   - Добавьте дополнительные фильтры")

    # Запускаем лучшую стратегию с графиками
    if best_strategy:
        print(f"\n📊 Запуск лучшей стратегии '{best_strategy}' с графиками...")
        best_config = next(s for s in strategies_to_test if s['name'] == best_strategy)
        best_config['params']['printlog'] = True  # Включаем логирование

        final_runner = BacktestRunner(initial_cash=10000, commission=0.001)
        final_data = final_runner.load_data_from_csv(
            csv_file=csv_file_path,
            datetime_col='timestamp',
            open_col='open',
            high_col='high',
            low_col='low',
            close_col='close',
            volume_col='volume',
            datetime_format='%Y-%m-%dT%H:%M:%S'
        )

        final_runner.setup_cerebro(RSI_SMA_Strategy, **best_config['params'])
        final_result = final_runner.run_backtest(final_data)
        final_runner.analyze_results(final_result)

        try:
            final_runner.plot_results()
        except Exception as e:
            print(f"Ошибка при построении графиков: {e}")

if __name__ == '__main__':
    main()