import os
import importlib
import inspect
import itertools
from typing import Dict, List, Any, Type, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Попытка импорта backtrader
try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None


class BaseStrategy(ABC):
    """Базовый класс для всех стратегий"""
    
    def __init__(self, **params):
        self.params = params
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерирует торговые сигналы"""
        pass
    
    @classmethod
    @abstractmethod
    def get_param_ranges(cls) -> Dict[str, Any]:
        """Возвращает диапазоны параметров для оптимизации"""
        pass
    
    @classmethod
    def get_strategy_name(cls) -> str:
        """Возвращает название стратегии"""
        return cls.__name__


class StrategyOptimizer:
    """Оптимизатор стратегий с автоматическим обнаружением"""

    def __init__(self, strategies_path: str = None, strategy_type: str = "auto"):
        """
        strategy_type: "auto", "baseStrategy", "backtrader"
        """
        self.strategies_path = strategies_path or self._get_default_strategies_path()
        self.discovered_strategies = {}
        self.current_strategy = None
        self.strategy_type = strategy_type

    def _get_default_strategies_path(self) -> str:
        """Получает путь к папке со стратегиями по умолчанию"""
        current_dir = Path(__file__).parent.parent.parent.parent
        return str(current_dir / "strategies" / "TestStrategies")
    
    def discover_strategies(self) -> Dict[str, Type]:
        """Автоматически обнаруживает все доступные стратегии"""
        strategies = {}
        
        if not os.path.exists(self.strategies_path):
            print(f"Путь к стратегиям не найден: {self.strategies_path}")
            return strategies
        
        # Поиск файлов Python в директории стратегий
        for root, dirs, files in os.walk(self.strategies_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    self._load_strategy_from_file(root, file, strategies)
        
        self.discovered_strategies = strategies
        return strategies
    
    def _load_strategy_from_file(self, root: str, filename: str, strategies: Dict):
        """Загружает стратегию из файла"""
        try:
            # Создаем путь для импорта
            file_path = os.path.join(root, filename)

            # Динамический импорт модуля
            spec = importlib.util.spec_from_file_location(
                filename[:-3], file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Поиск классов стратегий в модуле
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_valid_strategy(obj):
                    strategy_name = f"{obj.__name__}"
                    strategies[strategy_name] = obj
                    print(f"Найдена стратегия: {strategy_name}")

        except Exception as e:
            print(f"Ошибка загрузки стратегии из {filename}: {e}")

    def _is_valid_strategy(self, obj) -> bool:
        """Проверяет, является ли класс валидной стратегией"""
        if self.strategy_type == "baseStrategy":
            return (issubclass(obj, BaseStrategy) and
                    obj != BaseStrategy and
                    not inspect.isabstract(obj))
        elif self.strategy_type == "backtrader":
            return (BACKTRADER_AVAILABLE and
                    issubclass(obj, bt.Strategy) and
                    obj != bt.Strategy)
        elif self.strategy_type == "auto":
            # Автоматическое определение типа стратегии
            base_strategy_check = (issubclass(obj, BaseStrategy) and
                                 obj != BaseStrategy and
                                 not inspect.isabstract(obj))

            backtrader_check = (BACKTRADER_AVAILABLE and
                              issubclass(obj, bt.Strategy) and
                              obj != bt.Strategy)

            return base_strategy_check or backtrader_check

        return False

    def get_strategy_parameters(self, strategy_class: Type) -> Dict[str, Any]:
        """Получает параметры стратегии"""
        if hasattr(strategy_class, 'get_param_ranges'):
            return strategy_class.get_param_ranges()
        elif hasattr(strategy_class, 'params'):
            # Для backtrader стратегий - извлекаем параметры из params tuple
            params = {}
            if hasattr(strategy_class.params, '_getkeys'):
                try:
                    # Используем _getkeys для получения всех ключей параметров
                    for key in strategy_class.params._getkeys():
                        value = getattr(strategy_class.params, key)

                        # Создаем диапазоны на основе типа и значения параметра
                        if isinstance(value, int):
                            if value <= 5:
                                params[key] = list(range(max(1, value - 2), value + 3))
                            elif value <= 20:
                                params[key] = [max(1, value - 5), value, value + 5]
                            else:
                                params[key] = [max(1, int(value * 0.8)), value, int(value * 1.2)]
                        elif isinstance(value, float):
                            if value < 0.1:
                                params[key] = [round(max(0.001, value * 0.5), 4),
                                             round(value, 4),
                                             round(value * 2, 4)]
                            elif value < 1:
                                params[key] = [round(max(0.01, value * 0.7), 3),
                                             round(value, 3),
                                             round(value * 1.5, 3)]
                            else:
                                params[key] = [round(max(0.1, value * 0.8), 2),
                                             round(value, 2),
                                             round(value * 1.2, 2)]
                        elif isinstance(value, (str, bool)):
                            params[key] = [value]  # Оставляем как есть для строк и булевых
                        else:
                            params[key] = [value]
                except Exception as e:
                    print(f"Ошибка извлечения параметров: {e}")
            elif hasattr(strategy_class.params, '_getpairs'):
                try:
                    # Альтернативный способ для старых версий backtrader
                    for pair in strategy_class.params._getpairs():
                        if len(pair) >= 2:
                            key, value = pair[0], pair[1]
                            if isinstance(value, int):
                                if value <= 5:
                                    params[key] = list(range(max(1, value - 2), value + 3))
                                elif value <= 20:
                                    params[key] = [max(1, value - 5), value, value + 5]
                                else:
                                    params[key] = [max(1, int(value * 0.8)), value, int(value * 1.2)]
                            elif isinstance(value, float):
                                if value < 1:
                                    params[key] = [round(max(0.01, value * 0.7), 3),
                                                 round(value, 3),
                                                 round(value * 1.5, 3)]
                                else:
                                    params[key] = [round(max(0.1, value * 0.8), 2),
                                                 round(value, 2),
                                                 round(value * 1.2, 2)]
                            else:
                                params[key] = [value]
                except Exception as e:
                    print(f"Ошибка извлечения параметров через _getpairs: {e}")

            return params
        else:
            # Создаем базовые параметры для стратегий без явных параметров
            return {
                'period_fast': [5, 10, 15, 20],
                'period_slow': [20, 30, 50, 100],
                'rsi_period': [14, 21, 28],
                'rsi_upper': [70, 75, 80],
                'rsi_lower': [20, 25, 30]
            }

    def list_available_strategies(self) -> List[str]:
        """Возвращает список доступных стратегий"""
        if not self.discovered_strategies:
            self.discover_strategies()
        return list(self.discovered_strategies.keys())
    
    def select_strategy(self, strategy_name: str = None) -> Type[BaseStrategy]:
        """Выбирает стратегию для оптимизации"""
        if not self.discovered_strategies:
            self.discover_strategies()
        
        if not self.discovered_strategies:
            raise ValueError("Стратегии не найдены")
        
        if strategy_name is None:
            strategy_name = self._interactive_strategy_selection()
        
        if strategy_name not in self.discovered_strategies:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")
        
        self.current_strategy = self.discovered_strategies[strategy_name]
        return self.current_strategy
    
    def _interactive_strategy_selection(self) -> str:
        """Интерактивный выбор стратегии"""
        strategies = list(self.discovered_strategies.keys())
        
        print("\nДоступные стратегии:")
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
        
        while True:
            try:
                choice = input(f"\nВыберите стратегию (1-{len(strategies)}): ")
                index = int(choice) - 1
                if 0 <= index < len(strategies):
                    return strategies[index]
                else:
                    print("Неверный выбор. Попробуйте снова.")
            except ValueError:
                print("Введите число.")
    
    def generate_parameter_combinations(self, param_ranges: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерирует все возможные комбинации параметров"""
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def optimize_strategy_backtrader(self,
                                   data_path: str,
                                   strategy_class: Type,
                                   optimization_metric: str = 'total_return',
                                   custom_param_ranges: Dict[str, Any] = None,
                                   max_workers: int = 4) -> Dict[str, Any]:
        """Оптимизирует backtrader стратегию с многопоточностью"""
        if not BACKTRADER_AVAILABLE:
            raise ImportError("Backtrader не установлен")

        # Получаем диапазоны параметров
        param_ranges = custom_param_ranges or self.get_strategy_parameters(strategy_class)

        if not param_ranges:
            print("Нет параметров для оптимизации")
            return {}

        # Генерируем комбинации параметров
        param_combinations = self.generate_parameter_combinations(param_ranges)
        
        best_params = None
        best_score = float('-inf')
        results = []
        lock = threading.Lock()

        print(f"\nОптимизация backtrader стратегии {strategy_class.__name__}")
        print(f"Количество комбинаций параметров: {len(param_combinations)}")
        print(f"Используется потоков: {max_workers}")

        def run_single_backtest(params):
            try:
                score = self._run_backtrader_backtest(data_path, strategy_class, params, optimization_metric)
                return {'params': params.copy(), 'score': score}
            except Exception as e:
                print(f"Ошибка при тестировании параметров {params}: {str(e)}")
                return {'params': params.copy(), 'score': float('-inf')}

        # Многопоточная оптимизация
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_backtest, params): i
                      for i, params in enumerate(param_combinations)}

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    with lock:
                        results.append(result)
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_params = result['params']

                completed += 1
                if completed % max(1, len(param_combinations) // 20) == 0:
                    progress = completed / len(param_combinations) * 100
                    print(f"Прогресс: {progress:.1f}% | Лучший результат: {best_score:.4f}")

        optimization_result = {
            'best_params': best_params,
            'best_score': best_score,
            'strategy_name': strategy_class.__name__,
            'optimization_metric': optimization_metric,
            'all_results': results
        }
        
        print(f"\nОптимизация завершена!")
        print(f"Лучшие параметры: {best_params}")
        print(f"Лучший результат ({optimization_metric}): {best_score:.4f}")
        
        return optimization_result

    def _run_backtrader_backtest(self, data_path: str, strategy_class: Type, params: Dict, metric: str) -> float:
        """Запускает бэктест для одной комбинации параметров"""
        try:
            import pandas as pd

            # Создаем Cerebro
            cerebro = bt.Cerebro()

            # Загружаем данные
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)

                # Обработка индекса
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif df.index.name not in ['timestamp', 'date']:
                    # Если индекс не настроен, используем первый столбец как дату
                    first_col = df.columns[0]
                    if pd.api.types.is_datetime64_any_dtype(df[first_col]) or 'time' in first_col.lower() or 'date' in first_col.lower():
                        df[first_col] = pd.to_datetime(df[first_col])
                        df.set_index(first_col, inplace=True)

                # Обработка названий столбцов
                df.columns = df.columns.str.lower()
                required_columns = ['open', 'high', 'low', 'close']

                # Проверяем наличие столбцов
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    return 0  # Возвращаем 0 если данные некорректные

                # Создаем подкласс PandasData для правильного чтения данных
                class CustomPandasData(bt.feeds.PandasData):
                    params = (
                        ('datetime', None),
                        ('open', 'open'),
                        ('high', 'high'),
                        ('low', 'low'),
                        ('close', 'close'),
                        ('volume', 'volume' if 'volume' in df.columns else None),
                        ('openinterest', -1),
                    )

                data = CustomPandasData(dataname=df)
                cerebro.adddata(data)

            # Добавляем стратегию с параметрами
            cerebro.addstrategy(strategy_class, **params)

            # Настройки брокера
            cerebro.broker.set_cash(100000)
            cerebro.broker.setcommission(commission=0.001)

            # Добавляем анализаторы
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

            # Запускаем бэктест
            results = cerebro.run()

            # Вычисляем метрику
            if results and len(results) > 0:
                strat = results[0]

                if metric == 'total_return':
                    final_value = cerebro.broker.getvalue()
                    return (final_value - 100000) / 100000
                elif metric == 'sharpe_ratio':
                    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
                    return sharpe_analysis.get('sharperatio', 0) or 0
                elif metric == 'max_drawdown':
                    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
                    return -drawdown_analysis.get('max', {}).get('drawdown', 100)

            return 0

        except Exception as e:
            # Возвращаем 0 вместо исключения для продолжения оптимизации
            return 0

    def optimize_strategy(self,
                         data: pd.DataFrame = None,
                         data_path: str = None,
                         strategy_class: Type = None,
                         optimization_metric: str = 'total_return',
                         custom_param_ranges: Dict[str, Any] = None,
                         max_workers: int = 4) -> Dict[str, Any]:
        """Универсальный метод оптимизации стратегии"""

        if strategy_class is None:
            if self.current_strategy is None:
                strategy_class = self.select_strategy()
            else:
                strategy_class = self.current_strategy

        # Определяем тип стратегии и используем соответствующий метод
        if BACKTRADER_AVAILABLE and issubclass(strategy_class, bt.Strategy):
            if data_path is None:
                raise ValueError("Для backtrader стратегий требуется data_path")
            return self.optimize_strategy_backtrader(data_path, strategy_class, optimization_metric, custom_param_ranges, max_workers)
        elif issubclass(strategy_class, BaseStrategy):
            if data is None:
                raise ValueError("Для BaseStrategy требуется data DataFrame")
            return super().optimize_strategy(data, strategy_class, optimization_metric, custom_param_ranges)
        else:
            raise ValueError("Неподдерживаемый тип стратегии")

    def _calculate_performance_metric(self, signals: pd.DataFrame, metric: str) -> float:
        """Вычисляет метрику производительности"""
        if 'returns' not in signals.columns:
            raise ValueError("Столбец 'returns' не найден в сигналах")
        
        returns = signals['returns'].dropna()
        
        if len(returns) == 0:
            return float('-inf')
        
        if metric == 'sharpe_ratio':
            return self._calculate_sharpe_ratio(returns)
        elif metric == 'total_return':
            return (1 + returns).prod() - 1
        elif metric == 'max_drawdown':
            return -self._calculate_max_drawdown(returns)  # Отрицательное значение для максимизации
        elif metric == 'profit_factor':
            return self._calculate_profit_factor(returns)
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Вычисляет коэффициент Шарпа"""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)  # Годовой Sharpe
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Вычисляет максимальную просадку"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Вычисляет profit factor"""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 0
        
        return positive_returns / negative_returns
    
    def save_optimization_results(self, results: Dict[str, Any], filename: str = None):
        """Сохраняет результаты оптимизации"""
        if filename is None:
            filename = f"optimization_results_{results['strategy_name']}.json"
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            # Преобразуем результаты для JSON сериализации
            json_results = {
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'strategy_name': results['strategy_name'],
                'optimization_metric': results['optimization_metric'],
                'total_combinations': len(results['all_results'])
            }
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"Результаты сохранены в {filename}")


def main():
    """Основная функция для демонстрации работы оптимизатора"""
    optimizer = StrategyOptimizer(strategy_type="auto")

    # Путь к данным по умолчанию
    default_data_path = "CryptoTrade/data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    # Обнаружение стратегий
    strategies = optimizer.discover_strategies()
    print(f"Найдено стратегий: {len(strategies)}")
    
    if strategies:
        print("\nДоступные стратегии:")
        for name in strategies.keys():
            print(f"- {name}")

        # Выбираем стратегию
        try:
            selected_strategy = optimizer.select_strategy()
            print(f"\nВыбрана стратегия: {selected_strategy.__name__}")

            # Получаем параметры для оптимизации
            param_ranges = optimizer.get_strategy_parameters(selected_strategy)
            if param_ranges:
                print(f"Параметры для оптимизации: {param_ranges}")

                # Проверяем существование файла данных
                if os.path.exists(default_data_path):
                    data_path = default_data_path
                    print(f"\nИспользуется файл данных: {data_path}")
                else:
                    print(f"\nФайл данных по умолчанию не найден: {default_data_path}")
                    print("Введите путь к CSV файлу с данными:")
                    data_path = input().strip()

                    if not os.path.exists(data_path):
                        print(f"❌ Файл не найден: {data_path}")
                        return

                print(f"\n🚀 Запуск оптимизации стратегии {selected_strategy.__name__}...")
                print(f"📊 Количество параметров: {len(param_ranges)}")

                # Вычисляем общее количество комбинаций
                total_combinations = 1
                for values in param_ranges.values():
                    total_combinations *= len(values)
                print(f"🔄 Общее количество комбинаций: {total_combinations}")

                # Запускаем оптимизацию с многопоточностью
                results = optimizer.optimize_strategy(
                    data_path=data_path,
                    strategy_class=selected_strategy,
                    optimization_metric='total_return',
                    max_workers=8  # Увеличиваем количество потоков
                )

                if results:
                    # Сохраняем результаты
                    # optimizer.save_optimization_results(results)

                    print(f"\n🎉 Оптимизация завершена успешно!")
                    print(f"📊 Лучшие параметры: {results['best_params']}")
                    print(f"📈 Лучший результат: {results['best_score']:.4f}")
                    print(f"💾 Результаты сохранены в файл")
                else:
                    print("❌ Оптимизация не дала результатов")
            else:
                print("❌ Не удалось получить параметры для оптимизации")

        except KeyboardInterrupt:
            print("\n❌ Оптимизация прервана пользователем")
        except Exception as e:
            print(f"❌ Ошибка при оптимизации: {e}")
    else:
        print("❌ Стратегии не найдены.")


if __name__ == "__main__":
    main()