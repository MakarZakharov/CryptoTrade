import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import datetime
from typing import Dict, List, Tuple, Any, Type, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
from functools import partial

# Импортируем компоненты из существующего проекта
from CryptoTrade.backtest.HistoricalBacktest.historical_backtest import BacktestRunner
from CryptoTrade.strategies.TestStrategies.RSI_SMA_Strategy import RSI_SMA_Strategy


class StrategyOptimizer:
    """
    Класс для оптимизации параметров торговых стратегий
    """

    def __init__(self, csv_file: str, initial_cash: float = 10000, commission: float = 0.001,
                 max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Инициализация оптимизатора

        Args:
            csv_file (str): Путь к файлу с историческими данными
            initial_cash (float): Начальный капитал
            commission (float): Комиссия брокера
            max_workers (int, optional): Количество потоков/процессов
            use_processes (bool): Использовать процессы вместо потоков
        """
        self.csv_file = csv_file
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = []
        self.best_params = None
        self.best_result = None
        # Увеличиваем количество потоков для лучшей производительности
        self.max_workers = max_workers or min(cpu_count() * 2, 16)
        self.use_processes = use_processes
        self._lock = threading.Lock()

        # Предзагружаем данные один раз
        self._data_cache = None
        self._load_data_once()

        # Проверяем существование файла данных
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Файл данных не найден: {csv_file}")

        print(f"Инициализирован оптимизатор с {self.max_workers} {'процессами' if use_processes else 'потоками'}")

    def _load_data_once(self):
        """Предзагрузка данных для повторного использования"""
        try:
            temp_backtest = BacktestRunner(initial_cash=self.initial_cash, commission=self.commission)
            self._data_cache = temp_backtest.load_data_binance_csv(self.csv_file)
            if self._data_cache is None:
                raise ValueError(f"Не удалось загрузить данные из {self.csv_file}")
            print("Данные предзагружены в кэш")
        except Exception as e:
            print(f"Ошибка при предзагрузке данных: {e}")
            self._data_cache = None

    def optimize_parameters(
        self,
        strategy_class: Type[bt.Strategy],
        param_ranges: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        strategy_type: str = 'simple_rsi'
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Оптимизация параметров стратегии с многопоточностью
        """
        self.results = []

        # Создаем все возможные комбинации параметров
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(itertools.product(*param_values))

        print(f"Начинаем оптимизацию. Всего комбинаций: {len(combinations)}")
        print(f"Используем {self.max_workers} {'процессов' if self.use_processes else 'потоков'}")

        # Создаем частичную функцию для передачи в пул
        backtest_func = partial(
            self._run_single_backtest,
            strategy_class=strategy_class,
            strategy_type=strategy_type,
            param_names=param_names
        )

        # Выбираем тип исполнителя
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        # Запускаем параллельное выполнение с батчами для лучшей производительности
        with executor_class(max_workers=self.max_workers) as executor:
            # Отправляем задачи в пул батчами
            batch_size = min(50, len(combinations))
            futures = []

            for i in range(0, len(combinations), batch_size):
                batch = combinations[i:i + batch_size]
                for j, combination in enumerate(batch):
                    future = executor.submit(backtest_func, combination, i + j)
                    futures.append(future)

            # Собираем результаты по мере выполнения
            completed = 0
            for future in as_completed(futures):
                completed += 1

                try:
                    result = future.result(timeout=30)  # Добавляем таймаут
                    if result:
                        with self._lock:
                            self.results.append(result)

                    # Показываем прогресс чаще
                    if completed % max(1, len(combinations) // 50) == 0 or completed == len(combinations):
                        progress = (completed / len(combinations)) * 100
                        print(f"Прогресс: {completed}/{len(combinations)} ({progress:.1f}%)")

                except Exception as e:
                    pass  # Игнорируем ошибки для избежания конфликтов в многопоточности

        print(f"Завершено выполнение всех {len(combinations)} комбинаций")
        print(f"Получено {len(self.results)} валидных результатов")

        # Фильтруем результаты с валидными метриками
        def is_valid_metric(value):
            return value is not None and isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value)

        # Определяем ключ метрики
        metric_key_map = {
            'sharpe_ratio': 'sharpe_ratio',
            'returns': 'returns',
            'sqn': 'sqn',
            'drawdown': 'max_drawdown',
            'total_trades': 'total_trades'
        }

        metric_key = metric_key_map.get(metric, 'returns')  # По умолчанию используем доходность

        # Фильтруем результаты с валидными значениями метрики
        valid_results = []
        for result in self.results:
            metric_value = result['metrics'].get(metric_key)
            total_trades = result['metrics'].get('total_trades', 0)

            # Проверяем, что есть хотя бы одна сделка и валидная метрика
            if total_trades > 0 and is_valid_metric(metric_value):
                valid_results.append(result)

        if not valid_results:
            print(f"\nВНИМАНИЕ: Нет валидных результатов для метрики '{metric}'.")
            print("Возможные причины:")
            print("- Недостаточно данных для расчета метрик")
            print("- Стратегия не совершает сделок")
            print("- Проблемы с параметрами стратегии")
            print("- Слишком строгие условия входа")

            # Попробуем найти результаты хотя бы с одной сделкой
            results_with_trades = [r for r in self.results if r['metrics'].get('total_trades', 0) > 0]
            if results_with_trades:
                print(f"Найдено {len(results_with_trades)} результатов с торговыми сделками")
                # Используем доходность как запасную метрику
                self.best_result = max(results_with_trades, key=lambda x: x['metrics'].get('returns', -999))
                self.best_params = self.best_result['params']
                print(f"Лучшие параметры (по доходности): {self.best_params}")
                print(f"Результаты: {self.best_result['metrics']}")
                return self.best_params, self.best_result['metrics']
            else:
                self.best_params = {}
                self.best_result = {'params': {}, 'metrics': {}}
                return {}, {}

        print(f"Найдено {len(valid_results)} валидных результатов из {len(self.results)} общих")

        # Находим лучший результат среди валидных
        if metric == 'drawdown':
            # Для просадки ищем минимальное значение (меньше = лучше)
            self.best_result = min(valid_results, key=lambda x: abs(x['metrics'].get(metric_key, 999)))
        else:
            # Для остальных метрик ищем максимальное значение (больше = лучше)
            self.best_result = max(valid_results, key=lambda x: x['metrics'].get(metric_key, -999))

        self.best_params = self.best_result['params']

        print(f"\nЛучшие параметры: {self.best_params}")
        print(f"Результаты: {self.best_result['metrics']}")

        return self.best_params, self.best_result['metrics']

    def _run_single_backtest(self, combination: Tuple, index: int, strategy_class: Type[bt.Strategy],
                           strategy_type: str, param_names: List[str]) -> Optional[Dict[str, Any]]:
        """
        Выполнение одного бэктеста (для использования в пуле потоков/процессов)
        """
        try:
            # Создаем словарь параметров
            params = dict(zip(param_names, combination))
            params['strategy_type'] = strategy_type
            params['printlog'] = False  # Отключаем логирование для многопоточности

            # Запуск бэктеста с текущими параметрами
            metrics = self._run_backtest(strategy_class, params)

            return {
                'params': params,
                'metrics': metrics
            }

        except Exception as e:
            return None

    def _run_backtest(self, strategy_class: Type[bt.Strategy], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизированный запуск бэктеста с заданными параметрами
        """
        try:
            # Используем предзагруженные данные вместо повторной загрузки
            if self._data_cache is None:
                raise ValueError("Данные не загружены в кэш")

            # Создаем новый экземпляр cerebro для каждого теста
            cerebro = bt.Cerebro()

            # Безопасное клонирование данных
            try:
                data_feed = self._data_cache.clone()
            except AttributeError:
                # Если clone не поддерживается, создаем новый экземпляр
                temp_backtest = BacktestRunner(initial_cash=self.initial_cash, commission=self.commission)
                data_feed = temp_backtest.load_data_binance_csv(self.csv_file)

            cerebro.adddata(data_feed)

            # Настраиваем cerebro
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Добавляем стратегию с параметрами
            cerebro.addstrategy(strategy_class, **params)

            # Добавляем только необходимые анализаторы для скорости
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

            # Запускаем бэктест
            results = cerebro.run()
            result = results[0]

            # Быстрое извлечение метрик
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash

            metrics = {
                'final_value': final_value,
                'returns': total_return,
                'sharpe_ratio': self._safe_get_metric(result.analyzers.sharpe.get_analysis(), 'sharperatio', 0),
                'max_drawdown': self._safe_get_metric(result.analyzers.drawdown.get_analysis(), 'max.drawdown', 0),
                'sqn': self._safe_get_metric(result.analyzers.sqn.get_analysis(), 'sqn', 0),
            }

            # Анализ сделок
            trades_analysis = result.analyzers.trades.get_analysis()
            total_trades = trades_analysis.get('total', {}).get('total', 0)
            won_trades = trades_analysis.get('won', {}).get('total', 0)

            metrics.update({
                'total_trades': total_trades,
                'won_trades': won_trades,
                'win_rate': won_trades / total_trades if total_trades > 0 else 0
            })

            return metrics

        except Exception as e:
            print(f"Ошибка в бэктесте: {e}")
            # Возвращаем базовые метрики при ошибке
            return {
                'final_value': self.initial_cash,
                'returns': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'sqn': 0,
                'total_trades': 0,
                'won_trades': 0,
                'win_rate': 0
            }

    def _safe_get_metric(self, analysis_dict: dict, key_path: str, default_value: Any) -> Any:
        """Безопасное извлечение метрики из вложенного словаря"""
        try:
            keys = key_path.split('.')
            value = analysis_dict
            for key in keys:
                value = value[key]

            if value is None or np.isnan(value) or np.isinf(value):
                return default_value
            return value
        except:
            return default_value

    def plot_optimization_results(self, param1: str, param2: str, metric: str = 'returns'):
        """
        Визуализация результатов оптимизации для двух параметров

        Args:
            param1: Первый параметр для визуализации
            param2: Второй параметр для визуализации
            metric: Метрика для визуализации ('sharpe_ratio', 'returns', 'drawdown', 'sqn')
        """
        if not self.results:
            print("Нет результатов для визуализации. Сначала запустите оптимизацию.")
            return

        # Извлекаем уникальные значения параметров
        param1_values = sorted(list(set(result['params'].get(param1, 0) for result in self.results)))
        param2_values = sorted(list(set(result['params'].get(param2, 0) for result in self.results)))

        # Создаем сетку для тепловой карты
        heatmap_data = np.zeros((len(param1_values), len(param2_values)))

        # Заполняем сетку значениями метрики
        for result in self.results:
            if param1 in result['params'] and param2 in result['params']:
                p1_idx = param1_values.index(result['params'][param1])
                p2_idx = param2_values.index(result['params'][param2])

                if metric == 'sharpe_ratio':
                    value = result['metrics'].get('sharpe_ratio', 0)
                elif metric == 'returns':
                    value = result['metrics'].get('returns', 0)
                elif metric == 'sqn':
                    value = result['metrics'].get('sqn', 0)
                elif metric == 'drawdown':
                    # Для просадки инвертируем значение (меньше = лучше)
                    value = -result['metrics'].get('max_drawdown', 0)

                heatmap_data[p1_idx, p2_idx] = value

        # Создаем тепловую карту
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label=metric.replace('_', ' ').title())

        # Подписи осей
        plt.xticks(range(len(param2_values)), param2_values, rotation=90)
        plt.yticks(range(len(param1_values)), param1_values)
        plt.xlabel(param2)
        plt.ylabel(param1)

        # Добавляем значения в ячейки
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                value = heatmap_data[i, j]
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="w")

        plt.title(f"Оптимизация {metric.replace('_', ' ').title()} по параметрам {param1} и {param2}")
        plt.tight_layout()

        # Сохраняем график
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_{param1}_{param2}_{metric}_{timestamp}.png"
        save_path = os.path.join(os.path.dirname(self.csv_file), filename)
        plt.savefig(save_path)

        print(f"График сохранен: {save_path}")
        plt.show()

    def save_results(self, filename: Optional[str] = None):
        """
        Сохранение результатов оптимизации в CSV файл

        Args:
            filename: Имя файла для сохранения (если None, генерируется автоматически)
        """
        if not self.results:
            print("Нет результатов для сохранения. Сначала запустите оптимизацию.")
            return

        # Создаем DataFrame с результатами
        rows = []
        for result in self.results:
            row = {}
            # Добавляем параметры
            for param_name, param_value in result['params'].items():
                row[f"param_{param_name}"] = param_value

            # Добавляем метрики
            for metric_name, metric_value in result['metrics'].items():
                row[f"metric_{metric_name}"] = metric_value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Создаем имя файла, если не указано
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_type = self.results[0]['params'].get('strategy_type', 'unknown')
            filename = f"optimization_results_{strategy_type}_{timestamp}.csv"

        # Сохраняем в CSV
        save_path = os.path.join(os.path.dirname(self.csv_file), filename)
        df.to_csv(save_path, index=False)

        print(f"Результаты сохранены в: {save_path}")

        return save_path


def main():
    """
    Пример использования оптимизатора
    """
    # Путь к данным
    data_file = "C:\\Users\\Макар\\PycharmProjects\\trading\\CryptoTrade\\data\\binance\\BTCUSDT\\4h\\2022_12_15-2025_01_01.csv"

    # Создаем оптимизатор
    optimizer = StrategyOptimizer(
        csv_file=data_file,
        initial_cash=10000,
        commission=0.001
    )

    # Определяем диапазоны параметров для оптимизации (максимально мягкие условия)
    param_ranges = {
        'rsi_period': [10, 14, 21],  # RSI период
        'rsi_oversold': [15, 20, 25],  # Очень мягкие условия RSI перепроданности
        'rsi_overbought': [75, 80, 85],  # Очень мягкие условия RSI перекупленности
        'sma_period': [20, 30, 50],  # SMA период
        'position_size': [0.8, 0.9, 0.95],  # Размер позиции
        'use_leverage': [False],  # Отключаем леверидж для стабильности
        'leverage_multiplier': [1.0]  # Без левериджа
    }

    print("Тестируем разные стратегии...")

    strategies_to_test = ['simple_rsi', 'aggressive_momentum', 'multi_indicator']

    best_overall_params = None
    best_overall_metrics = None
    best_overall_return = -999

    for strategy_type in strategies_to_test:
        print(f"\n{'='*60}")
        print(f"ОПТИМИЗАЦИЯ СТРАТЕГИИ: {strategy_type}")
        print(f"{'='*60}")

        # Запускаем оптимизацию
        best_params, best_metrics = optimizer.optimize_parameters(
            strategy_class=RSI_SMA_Strategy,
            param_ranges=param_ranges,
            metric='returns',  # Оптимизируем по доходности
            strategy_type=strategy_type
        )

        if best_metrics and best_metrics.get('returns', 0) > best_overall_return:
            best_overall_return = best_metrics.get('returns', 0)
            best_overall_params = best_params
            best_overall_metrics = best_metrics

        # Сохраняем результаты для каждой стратегии
        optimizer.save_results(f"optimization_results_{strategy_type}.csv")

    print(f"\n{'='*60}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print(f"{'='*60}")

    if best_overall_params:
        print(f"Лучшая стратегия: {best_overall_params.get('strategy_type', 'unknown')}")
        print(f"Лучшие параметры: {best_overall_params}")
        print(f"Лучшие метрики:")
        for metric, value in best_overall_metrics.items():
            print(f"  {metric}: {value}")

        # Визуализируем результаты лучшей стратегии
        if len(optimizer.results) > 0:
            optimizer.plot_optimization_results(
                param1='rsi_period',
                param2='rsi_oversold',
                metric='returns'
            )
    else:
        print("Не удалось найти оптимальные параметры. Попробуйте:")
        print("1. Изменить диапазоны параметров")
        print("2. Использовать другие временные рамки")
        print("3. Проверить качество данных")

    print("\nОптимизация завершена!")


# Запуск оптимизации, если файл запускается напрямую
if __name__ == "__main__":
    main()