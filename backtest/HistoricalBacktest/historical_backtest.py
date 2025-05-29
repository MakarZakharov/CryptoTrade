import backtrader as bt
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from typing import Optional, Type, Dict, Any, List

from CryptoTrade.strategies.TestStrategies.RSI_SMA_Strategy import RSI_SMA_Strategy


class BacktestRunner:
    """
    Класс для запуска и анализа бэктестов торговых стратегий
    """

    def __init__(self, initial_cash: float = 10000, commission: float = 0.001) -> None:
        """
        Инициализация бэктестера

        Args:
            initial_cash (float): Начальный капитал
            commission (float): Комиссия брокера (0.1% = 0.001)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro: Optional[bt.Cerebro] = None
        self.results: Optional[Any] = None

    def load_data_from_csv(
        self,
        csv_file: str,
        datetime_col: str = 'datetime',
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: Optional[str] = 'volume',
        datetime_format: str = '%Y-%m-%d %H:%M:%S'
    ) -> Optional[bt.feeds.PandasData]:
        """
        Загрузка данных из CSV файла

        Args:
            csv_file (str): Путь к CSV файлу
            datetime_col (str): Название колонки с датой/временем
            open_col (str): Название колонки с ценой открытия
            high_col (str): Название колонки с максимальной ценой
            low_col (str): Название колонки с минимальной ценой
            close_col (str): Название колонки с ценой закрытия
            volume_col (Optional[str]): Название колонки с объемом
            datetime_format (str): Формат даты/времени

        Returns:
            bt.feeds.PandasData или None при ошибке
        """
        if not os.path.exists(csv_file):
            print(f"Ошибка: файл {csv_file} не найден")
            return None

        try:
            data = pd.read_csv(csv_file)
            print(f"Загружено {len(data)} строк из {csv_file}")

            required_cols = [datetime_col, open_col, high_col, low_col, close_col]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Ошибка: отсутствуют обязательные колонки: {missing_cols}")
                return None

            data[datetime_col] = pd.to_datetime(data[datetime_col], format=datetime_format, errors='coerce')
            data.dropna(subset=[datetime_col], inplace=True)
            data.set_index(datetime_col, inplace=True)

            column_mapping = {
                open_col: 'Open',
                high_col: 'High',
                low_col: 'Low',
                close_col: 'Close'
            }
            if volume_col and volume_col in data.columns:
                column_mapping[volume_col] = 'Volume'

            data.rename(columns=column_mapping, inplace=True)
            data.sort_index(inplace=True)

            # Проверка на наличие Volume
            volume_param = 'Volume' if 'Volume' in data.columns else None

            data_bt = bt.feeds.PandasData(
                dataname=data,
                datetime=None,
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume=volume_param,
                openinterest=None
            )
            return data_bt

        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            return None

    def load_data_binance_csv(self, csv_file: str) -> Optional[bt.feeds.PandasData]:
        """
        Загрузка данных из CSV файла Binance (стандартный формат)

        Args:
            csv_file (str): Путь к CSV файлу от Binance

        Returns:
            bt.feeds.PandasData или None при ошибке
        """
        return self.load_data_from_csv(
            csv_file=csv_file,
            datetime_col='timestamp',
            open_col='open',
            high_col='high',
            low_col='low',
            close_col='close',
            volume_col='volume',
            datetime_format='%Y-%m-%dT%H:%M:%S'
        )

    def setup_cerebro(self, strategy_class: Type[bt.Strategy] = RSI_SMA_Strategy, **strategy_params: Any) -> None:
        """
        Настройка движка backtrader

        Args:
            strategy_class: Класс стратегии для тестирования
            **strategy_params: Параметры стратегии
        """
        self.cerebro = bt.Cerebro()
        self.cerebro.addstrategy(strategy_class, **strategy_params)
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)

        # Добавляем анализаторы
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    def run_backtest(self, data_feed: bt.feeds.PandasData) -> Any:
        """
        Запуск бэктеста

        Args:
            data_feed: Данные для тестирования

        Returns:
            Результат выполнения стратегии
        """
        if self.cerebro is None:
            raise ValueError("Сначала вызовите setup_cerebro()")

        self.cerebro.adddata(data_feed)
        print(f'Начальный капитал: {self.cerebro.broker.getvalue():.2f}')
        self.results = self.cerebro.run()
        print(f'Финальный капитал: {self.cerebro.broker.getvalue():.2f}')
        return self.results[0]

    def analyze_results(self, result: Any) -> None:
        """
        Анализ результатов бэктеста

        Args:
            result: Результат выполнения стратегии
        """
        print("\n" + "=" * 50)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ БЭКТЕСТА")
        print("=" * 50)

        final_value = self.cerebro.broker.getvalue() if self.cerebro else 0
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100 if self.initial_cash else 0

        print(f"Начальный капитал: ${self.initial_cash:,.2f}")
        print(f"Финальный капитал: ${final_value:,.2f}")
        print(f"Общая доходность: {total_return:.2f}%")

        trade_analyzer = result.analyzers.trades.get_analysis()
        if 'total' in trade_analyzer and trade_analyzer.total.total > 0:
            won = trade_analyzer.won.total
            lost = trade_analyzer.lost.total
            total = trade_analyzer.total.total
            print(f"\nАНАЛИЗ СДЕЛОК:")
            print(f"Всего сделок: {total}")
            print(f"Выигрышных: {won}")
            print(f"Проигрышных: {lost}")
            print(f"Процент выигрышных: {won / total * 100:.1f}%")

            if won > 0:
                print(f"Средняя прибыль: ${trade_analyzer.won.pnl.average:.2f}")
            if lost > 0:
                print(f"Средний убыток: ${trade_analyzer.lost.pnl.average:.2f}")

        sharpe = result.analyzers.sharpe.get_analysis()
        sharperatio = sharpe.get('sharperatio', None)
        if sharperatio is not None:
            print(f"\nКоэффициент Шарпа: {sharperatio:.3f}")

        drawdown = result.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.get('max', {}).get('drawdown', None)
        if max_drawdown is not None:
            print(f"Максимальная просадка: {max_drawdown:.2f}%")

        sqn = result.analyzers.sqn.get_analysis()
        sqn_value = sqn.get('sqn', None)
        if sqn_value is not None:
            print(f"SQN: {sqn_value:.2f}")

    def plot_results(self, figsize: tuple = (15, 10)) -> None:
        """
        Построение графиков результатов

        Args:
            figsize (tuple): Размер графика
        """
        if self.cerebro is None:
            print("Нет данных для построения графика")
            return

        try:
            self.cerebro.plot(figsize=figsize, style='candlestick')
            plt.show()
        except Exception as e:
            print(f"Ошибка при построении графиков: {e}")


def main() -> None:
    """
    Основная функция для запуска бэктеста с CSV данными
    """
    print("Инициализация бэктестера для высокодоходных стратегий...")

    csv_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    ))

    strategies_to_test: List[Dict[str, Any]] = [
        {
            'name': 'Агрессивная Momentum',
            'params': {
                'rsi_period': 14,
                'sma_period': 30,
                'sma_fast': 12,
                'sma_slow': 26,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
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
                'rsi_oversold': 35,
                'rsi_overbought': 70,
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
                'rsi_oversold': 40,
                'rsi_overbought': 60,
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
    best_return = float('-inf')

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ВЫСОКОДОХОДНЫХ СТРАТЕГИЙ")
    print("=" * 80)

    for strategy_config in strategies_to_test:
        print(f"\n🚀 Тестирование стратегии: {strategy_config['name']}")
        print("-" * 50)

        runner = BacktestRunner(initial_cash=10000, commission=0.001)
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
            print("Не удалось загрузить данные для стратегии.")
            continue

        runner.setup_cerebro(RSI_SMA_Strategy, **strategy_config['params'])
        result = runner.run_backtest(data)

        final_value = runner.cerebro.broker.getvalue()
        total_return = (final_value - 10000) / 10000 * 100

        print(f"📊 Результаты стратегии '{strategy_config['name']}':")
        print(f"   💰 Финальный капитал: ${final_value:,.2f}")
        print(f"   📈 Доходность: {total_return:.2f}%")

        if total_return > best_return:
            best_return = total_return
            best_strategy = strategy_config['name']

        runner.analyze_results(result)

    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(f"🏆 Лучшая стратегия: {best_strategy}")
    print(f"💎 Максимальная доходность: {best_return:.2f}%")

    if best_return >= 1000:
        print("🎉 ЦЕЛЬ ДОСТИГНУТА! Доходность превышает 1000%!")
    else:
        print("⚠️  Цель не достигнута. Рекомендации:")
        print("   - Попробуйте изменить параметры стратегий")
        print("   - Рассмотрите другие временные рамки")
        print("   - Добавьте дополнительные фильтры")

    if best_strategy:
        print(f"\n📊 Запуск лучшей стратегии '{best_strategy}' с графиками...")
        best_config = next(s for s in strategies_to_test if s['name'] == best_strategy)
        best_config['params']['printlog'] = True

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

        if final_data is not None:
            final_runner.setup_cerebro(RSI_SMA_Strategy, **best_config['params'])
            final_result = final_runner.run_backtest(final_data)
            final_runner.analyze_results(final_result)
            final_runner.plot_results()
        else:
            print("Не удалось загрузить данные для финального запуска.")


if __name__ == '__main__':
    main()
