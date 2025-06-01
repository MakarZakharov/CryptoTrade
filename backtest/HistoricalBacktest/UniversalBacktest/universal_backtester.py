import os
import sys
import importlib.util
import inspect
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
import glob
from collections import defaultdict
import numpy as np

warnings.filterwarnings('ignore')


class AdvancedSizer(bt.Sizer):
    """Продвинутый сайзер с учетом риск-менеджмента"""
    
    params = (
        ('position_size', 0.95),
        ('max_risk_per_trade', 0.02),  # 2% риска на сделку
        ('use_fixed_size', False),
    )
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        if self.params.use_fixed_size:
            # Фиксированный размер позиции
            size = (cash * self.params.position_size) / data.close[0]
        else:
            # Размер на основе риска
            size = (cash * self.params.max_risk_per_trade) / data.close[0]
        
        return int(size) if size > 0 else 0


class EnhancedCommissionInfo(bt.CommInfoBase):
    """Улучшенная комиссионная схема с учетом спреда и проскальзывания"""
    
    params = (
        ('commission', 0.001),  # 0.1% комиссия
        ('spread', 0.0005),     # 0.05% спред
        ('slippage', 0.0002),   # 0.02% проскальзывание
        ('margin', None),
        ('mult', 1.0),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        """Расчет комиссии с учетом спреда и проскальзывания"""
        # Базовая комиссия
        commission = abs(size) * price * self.p.commission
        
        # Добавляем спред (на каждую сделку)
        spread_cost = abs(size) * price * self.p.spread
        
        # Добавляем проскальзывание
        slippage_cost = abs(size) * price * self.p.slippage
        
        total_cost = commission + spread_cost + slippage_cost
        
        return total_cost


class SilentStrategyWrapper:
    """Wrapper для подавления ошибок стратегии"""

    @classmethod
    def wrap_strategy(cls, strategy_class):
        """Создает wrapper для стратегии с подавлением ошибок"""
        
        class WrappedStrategy(strategy_class):
            error_count = 0
            max_errors_to_show = 5

            def notify_order(self, order):
                try:
                    super().notify_order(order)
                except Exception:
                    pass

            def next(self):
                try:
                    super().next()
                except (IndexError, TypeError, ZeroDivisionError, KeyError):
                    self.__class__.error_count += 1
                except Exception:
                    pass

        return WrappedStrategy


class DataManager:
    """Менеджер данных для автоматического поиска и загрузки данных"""
    
    def __init__(self, data_root_path: str = None):
        self.data_root_path = data_root_path or self._find_data_root()
        self.available_data = self._scan_available_data()
    
    def _find_data_root(self) -> str:
        """Автоматический поиск корневой папки данных"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        possible_paths = [
            os.path.join(current_dir, '../../../data'),
            os.path.join(current_dir, '../../data'),
            os.path.join(current_dir, '../data'),
            os.path.join(current_dir, 'data'),
            os.path.join(current_dir.split('CryptoTrade')[0], 'CryptoTrade', 'data'),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        raise FileNotFoundError("Папка с данными не найдена")
    
    def _scan_available_data(self) -> Dict[str, Dict[str, List[str]]]:
        """Сканирование доступных данных"""
        data_structure = defaultdict(lambda: defaultdict(list))
        
        if not os.path.exists(self.data_root_path):
            return dict(data_structure)
        
        for exchange in os.listdir(self.data_root_path):
            exchange_path = os.path.join(self.data_root_path, exchange)
            if not os.path.isdir(exchange_path):
                continue
                
            for symbol in os.listdir(exchange_path):
                symbol_path = os.path.join(exchange_path, symbol)
                if not os.path.isdir(symbol_path):
                    continue
                    
                for timeframe in os.listdir(symbol_path):
                    timeframe_path = os.path.join(symbol_path, timeframe)
                    if not os.path.isdir(timeframe_path):
                        continue
                    
                    # Поиск CSV файлов
                    csv_files = glob.glob(os.path.join(timeframe_path, "*.csv"))
                    if csv_files:
                        key = f"{exchange}_{symbol}_{timeframe}"
                        data_structure[exchange][symbol].extend([
                            {
                                'timeframe': timeframe,
                                'files': csv_files,
                                'key': key
                            }
                        ])
        
        return dict(data_structure)
    
    def list_available_data(self):
        """Вывод списка доступных данных"""
        print("\n📊 ДОСТУПНЫЕ ДАННЫЕ:")
        print("=" * 80)
        
        total_datasets = 0
        for exchange, symbols in self.available_data.items():
            print(f"\n📈 Биржа: {exchange.upper()}")
            print("-" * 40)
            
            for symbol, timeframe_data in symbols.items():
                print(f"  💰 {symbol}:")
                for tf_info in timeframe_data:
                    file_count = len(tf_info['files'])
                    print(f"    ⏰ {tf_info['timeframe']} ({file_count} файл(ов))")
                    total_datasets += file_count
        
        print(f"\n📊 Всего наборов данных: {total_datasets}")
        print("=" * 80)
    
    def get_data_path(self, exchange: str, symbol: str, timeframe: str, 
                     start_date: str = None, end_date: str = None) -> str:
        """Получение пути к данным"""
        if exchange not in self.available_data:
            raise ValueError(f"Биржа {exchange} не найдена")
        
        if symbol not in self.available_data[exchange]:
            raise ValueError(f"Символ {symbol} не найден для биржи {exchange}")
        
        # Поиск нужного таймфрейма
        for tf_info in self.available_data[exchange][symbol]:
            if tf_info['timeframe'] == timeframe:
                # Если указаны даты, ищем подходящий файл
                if start_date or end_date:
                    return self._find_file_by_date_range(tf_info['files'], start_date, end_date)
                else:
                    # Возвращаем первый доступный файл
                    return tf_info['files'][0]
        
        raise ValueError(f"Таймфрейм {timeframe} не найден для {exchange}:{symbol}")
    
    def _find_file_by_date_range(self, files: List[str], start_date: str, end_date: str) -> str:
        """Поиск файла по диапазону дат"""
        # Простая реализация - возвращаем первый файл
        # В реальности здесь можно анализировать имена файлов
        if files:
            return files[0]
        raise FileNotFoundError("Файлы данных не найдены")
    
    def load_data(self, exchange: str, symbol: str, timeframe: str,
                 start_date: str = None, end_date: str = None) -> bt.feeds.PandasData:
        """Загрузка данных в формате BackTrader"""
        file_path = self.get_data_path(exchange, symbol, timeframe, start_date, end_date)
        
        # Загрузка CSV
        df = pd.read_csv(file_path)
        
        # Обработка данных
        df = self._process_dataframe(df, start_date, end_date)

        # Создание feed для BackTrader - используем index как datetime
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Используем индекс как datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        return data_feed
    
    def _process_dataframe(self, df: pd.DataFrame, start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """Обработка DataFrame"""
        # Конвертация времени
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Фильтрация по датам
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        # Установка индекса
        df.set_index('timestamp', inplace=True)
        
        # Проверка и очистка данных
        df = df.dropna()
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        
        # Добавление volume если отсутствует
        if 'volume' not in df.columns:
            df['volume'] = 1000
        
        # Сортировка по индексу (времени)
        df.sort_index(inplace=True)

        return df


class UniversalBacktester:
    """Универсальный бэктестер с расширенным функционалом"""

    def __init__(self, 
                 initial_cash: float = 100000,
                 commission: float = 0.001,
                 spread: float = 0.0005,
                 slippage: float = 0.0002,
                 data_root_path: str = None):
        
        self.initial_cash = initial_cash
        self.commission = commission
        self.spread = spread
        self.slippage = slippage
        
        # Менеджеры
        self.data_manager = DataManager(data_root_path)
        self.strategies_registry = {}
        
        print("🔍 Инициализация универсального бэктестера...")
        self._discover_strategies()

    def _discover_strategies(self):
        """Автоматическое обнаружение стратегий"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        possible_paths = [
            os.path.join(current_dir, '../../../strategies/TestStrategies/'),
            os.path.join(current_dir, '../../strategies/TestStrategies/'),
            os.path.join(current_dir, '../strategies/TestStrategies/'),
            os.path.join(current_dir, 'strategies/TestStrategies/'),
            os.path.join(current_dir.split('CryptoTrade')[0], 'CryptoTrade', 'strategies', 'TestStrategies'),
        ]

        strategies_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isdir(abs_path):
                strategies_path = abs_path
                break

        if not strategies_path:
            print(f"⚠️ Папка стратегий не найдена")
            return

        print(f"📁 Сканирую стратегии: {strategies_path}")
        
        if strategies_path not in sys.path:
            sys.path.insert(0, strategies_path)

        strategies_found = 0
        for filename in os.listdir(strategies_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                found_count = self._load_strategies_from_module(module_name, strategies_path)
                strategies_found += found_count

        print(f"✅ Загружено стратегий: {strategies_found}")

    def _load_strategies_from_module(self, module_name: str, module_path: str) -> int:
        """Загрузка стратегий из модуля"""
        strategies_loaded = 0
        
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(module_path, f"{module_name}.py"))
            
            if spec is None or spec.loader is None:
                return 0
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if self._is_strategy_class(obj):
                    try:
                        default_params = self._extract_strategy_params(obj)
                        
                        unique_key = f"{name}_{module_name}" if name in self.strategies_registry else name
                        
                        self.strategies_registry[unique_key] = {
                            'class': obj,
                            'module': module_name,
                            'file': f"{module_name}.py",
                            'default_params': default_params,
                            'description': self._clean_docstring(obj.__doc__) or f"Стратегия {name}",
                            'original_name': name
                        }
                        strategies_loaded += 1
                        print(f"✅ {name} (параметров: {len(default_params)})")

                    except Exception as e:
                        print(f"⚠️ Ошибка загрузки {name}: {e}")

        except Exception as e:
            print(f"❌ Ошибка модуля {module_name}: {e}")

        return strategies_loaded

    def _is_strategy_class(self, obj) -> bool:
        """Проверка является ли объект классом стратегии"""
        return (
            inspect.isclass(obj) and
            issubclass(obj, bt.Strategy) and
            obj != bt.Strategy and
            not obj.__name__.startswith('_') and
            hasattr(obj, '__module__')
        )

    def _extract_strategy_params(self, strategy_class) -> Dict[str, Any]:
        """Извлечение параметров стратегии"""
        default_params = {}
        
        if not hasattr(strategy_class, 'params'):
            return default_params

        params_attr = getattr(strategy_class, 'params')
        if params_attr is None:
            return default_params

        try:
            if isinstance(params_attr, (tuple, list)):
                for param in params_attr:
                    if isinstance(param, tuple) and len(param) >= 2:
                        name, value = param[0], param[1]
                        if self._is_valid_param(name, value):
                            default_params[name] = value
            
            elif isinstance(params_attr, dict):
                for name, value in params_attr.items():
                    if self._is_valid_param(name, value):
                        default_params[name] = value
            
            elif hasattr(params_attr, '__dict__'):
                for name in dir(params_attr):
                    if not name.startswith('_'):
                        value = getattr(params_attr, name)
                        if self._is_valid_param(name, value):
                            default_params[name] = value
        
        except Exception:
            pass

        return default_params

    def _is_valid_param(self, name: str, value: Any) -> bool:
        """Проверка валидности параметра"""
        return (
            not callable(value) and 
            not name.startswith('_') and
            name not in ['isdefault', 'notdefault'] and
            not inspect.isclass(value)
        )

    def _clean_docstring(self, docstring: str) -> str:
        """Очистка и форматирование docstring"""
        if not docstring:
            return ""
        
        lines = [line.strip() for line in docstring.strip().split('\n')]
        cleaned = ' '.join(line for line in lines if line)
        
        if len(cleaned) > 100:
            return cleaned[:97] + "..."
        return cleaned

    def list_available_options(self):
        """Вывод всех доступных опций"""
        print("\n🔍 УНИВЕРСАЛЬНЫЙ БЭКТЕСТЕР")
        print("=" * 80)
        
        # Доступные данные
        self.data_manager.list_available_data()
        
        # Доступные стратегии
        self.list_strategies()

    def list_strategies(self):
        """Вывод доступных стратегий"""
        print("\n🎯 ДОСТУПНЫЕ СТРАТЕГИИ:")
        print("=" * 80)

        if not self.strategies_registry:
            print("❌ Стратегии не найдены!")
            return

        strategies_by_file = defaultdict(list)
        for key, info in self.strategies_registry.items():
            strategies_by_file[info['file']].append((key, info))

        for file_name, strategies in strategies_by_file.items():
            print(f"\n📄 Файл: {file_name}")
            print("-" * 60)

            for i, (key, info) in enumerate(strategies, 1):
                name = info['original_name']
                print(f"   {i}. 🎯 {name}")
                print(f"      📝 {info['description']}")

                if info['default_params']:
                    param_count = len(info['default_params'])
                    print(f"      ⚙️ Параметры ({param_count}):")
                    
                    for param_name, param_value in list(info['default_params'].items())[:5]:
                        print(f"         • {param_name}: {param_value}")

                    if param_count > 5:
                        print(f"         ... и еще {param_count - 5} параметров")
                print()

        print(f"📊 Всего стратегий: {len(self.strategies_registry)}")
        print("=" * 80)

    def run_single_backtest(self,
                           strategy_name: str,
                           exchange: str = "binance",
                           symbol: str = "BTCUSDT", 
                           timeframe: str = "1d",
                           start_date: str = None,
                           end_date: str = None,
                           strategy_params: Dict[str, Any] = None,
                           show_plot: bool = True,
                           verbose: bool = True,
                           suppress_strategy_errors: bool = True) -> Dict[str, Any]:
        """Запуск одиночного бэктеста"""
        
        if strategy_name not in self.strategies_registry:
            available = list(self.strategies_registry.keys())
            raise ValueError(f"Стратегия '{strategy_name}' не найдена. Доступные: {available}")

        strategy_info = self.strategies_registry[strategy_name]
        strategy_class = strategy_info['class']

        # Объединение параметров
        final_params = strategy_info['default_params'].copy()
        if strategy_params:
            final_params.update(strategy_params)

        if verbose:
            self._print_backtest_header(strategy_name, exchange, symbol, timeframe, 
                                      start_date, end_date, final_params)

        try:
            # Создание Cerebro
            cerebro = bt.Cerebro()

            # Добавление стратегии
            if suppress_strategy_errors:
                wrapped_class = SilentStrategyWrapper.wrap_strategy(strategy_class)
                cerebro.addstrategy(wrapped_class, **final_params)
            else:
                cerebro.addstrategy(strategy_class, **final_params)

            # Загрузка данных
            data_feed = self.data_manager.load_data(exchange, symbol, timeframe, start_date, end_date)
            cerebro.adddata(data_feed)

            # Настройка брокера
            cerebro.broker.setcash(self.initial_cash)
            
            # Добавление улучшенной комиссионной схемы
            comminfo = EnhancedCommissionInfo(
                commission=self.commission,
                spread=self.spread,
                slippage=self.slippage
            )
            cerebro.broker.addcommissioninfo(comminfo)

            # Добавление продвинутого сайзера
            cerebro.addsizer(AdvancedSizer)

            # Добавление анализаторов
            self._add_analyzers(cerebro)

            # Запуск бэктеста
            results = cerebro.run()
            if not results:
                raise RuntimeError("Стратегия не вернула результатов")
            
            result = results[0]

            # Обработка результатов
            analysis_result = self._process_results(result, strategy_name, final_params, 
                                                  exchange, symbol, timeframe)

            if verbose:
                self._print_results(analysis_result)
            
            if show_plot:
                self._plot_results(cerebro, strategy_name, exchange, symbol, timeframe)

            return analysis_result

        except Exception as e:
            if verbose:
                print(f"❌ Ошибка выполнения: {str(e)}")
            raise

    def run_multi_data_backtest(self,
                               strategy_name: str,
                               data_configs: List[Dict[str, str]],
                               strategy_params: Dict[str, Any] = None,
                               show_individual_plots: bool = False,
                               verbose: bool = True) -> pd.DataFrame:
        """Запуск бэктеста на множественных данных"""
        
        print(f"\n🔍 МУЛЬТИ-ТЕСТ СТРАТЕГИИ: {strategy_name}")
        print("=" * 80)
        print(f"📊 Наборов данных: {len(data_configs)}")
        print()

        results = []
        failed_tests = []

        for i, config in enumerate(data_configs, 1):
            exchange = config.get('exchange', 'binance')
            symbol = config.get('symbol', 'BTCUSDT')
            timeframe = config.get('timeframe', '1d')
            start_date = config.get('start_date')
            end_date = config.get('end_date')

            test_name = f"{exchange}_{symbol}_{timeframe}"
            print(f"⏳ [{i}/{len(data_configs)}] Тестирование на {test_name}")

            try:
                result = self.run_single_backtest(
                    strategy_name=strategy_name,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_params=strategy_params,
                    show_plot=show_individual_plots,
                    verbose=False,
                    suppress_strategy_errors=True
                )
                
                result['test_name'] = test_name
                result['exchange'] = exchange
                result['symbol'] = symbol
                result['timeframe'] = timeframe
                
                results.append(result)
                print(f"✅ {test_name}: {result['total_return']:+.2f}% | {result.get('total_trades', 0)} сделок")

            except Exception as e:
                error_msg = str(e)
                print(f"❌ {test_name}: {error_msg}")
                failed_tests.append((test_name, error_msg))

        if failed_tests:
            print(f"\n⚠️ Неудачные тесты ({len(failed_tests)}):")
            for test_name, error in failed_tests:
                print(f"   • {test_name}: {error}")

        if not results:
            print("❌ Нет успешных результатов")
            return pd.DataFrame()

        # Создание сводной таблицы
        comparison_df = pd.DataFrame(results).sort_values('total_return', ascending=False)
        
        key_metrics = [
            'test_name', 'exchange', 'symbol', 'timeframe', 'total_return', 
            'profit_loss', 'total_trades', 'win_rate', 'profit_factor', 
            'sharpe_ratio', 'max_drawdown'
        ]
        available_metrics = [col for col in key_metrics if col in comparison_df.columns]
        display_df = comparison_df[available_metrics].copy()

        print(f"\n🏆 РЕЗУЛЬТАТЫ МУЛЬТИ-ТЕСТА:")
        print("=" * 120)
        print(display_df.to_string(index=False, float_format='%.2f'))

        # Статистика
        if len(results) > 1:
            avg_return = comparison_df['total_return'].mean()
            std_return = comparison_df['total_return'].std()
            best_test = comparison_df.iloc[0]
            worst_test = comparison_df.iloc[-1]

            print(f"\n📊 СТАТИСТИКА:")
            print(f"   Средняя доходность: {avg_return:.2f}%")
            print(f"   Стандартное отклонение: {std_return:.2f}%")
            print(f"   🥇 Лучший тест: {best_test['test_name']} ({best_test['total_return']:+.2f}%)")
            print(f"   🥉 Худший тест: {worst_test['test_name']} ({worst_test['total_return']:+.2f}%)")

        print("=" * 120)
        return display_df

    def compare_strategies(self,
                          strategy_names: List[str] = None,
                          exchange: str = "binance",
                          symbol: str = "BTCUSDT",
                          timeframe: str = "1d",
                          start_date: str = None,
                          end_date: str = None,
                          custom_params: Dict[str, Dict[str, Any]] = None,
                          skip_errors: bool = True) -> pd.DataFrame:
        """Сравнение множественных стратегий"""
        
        if strategy_names is None:
            strategy_names = list(self.strategies_registry.keys())
        
        if custom_params is None:
            custom_params = {}

        print(f"\n🔍 СРАВНЕНИЕ СТРАТЕГИЙ")
        print("=" * 80)
        print(f"📊 Стратегий: {len(strategy_names)}")
        print(f"📈 Данные: {exchange}:{symbol} ({timeframe})")
        if start_date or end_date:
            print(f"📅 Период: {start_date or 'начало'} - {end_date or 'конец'}")
        print()

        results = []
        failed_strategies = []

        for i, strategy_name in enumerate(strategy_names, 1):
            if strategy_name not in self.strategies_registry:
                print(f"❌ Стратегия '{strategy_name}' не найдена")
                continue

            print(f"⏳ [{i}/{len(strategy_names)}] Тестирование: {strategy_name}")

            try:
                params = custom_params.get(strategy_name, {})
                result = self.run_single_backtest(
                    strategy_name=strategy_name,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_params=params,
                    show_plot=False,
                    verbose=False,
                    suppress_strategy_errors=True
                )
                results.append(result)
                print(f"✅ {result['total_return']:+.2f}% | {result.get('total_trades', 0)} сделок")

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Ошибка: {error_msg}")
                failed_strategies.append(strategy_name)
                
                if not skip_errors:
                    raise e

        if failed_strategies:
            print(f"\n⚠️ Стратегии с ошибками ({len(failed_strategies)}):")
            for strategy in failed_strategies:
                print(f"   • {strategy}")

        if not results:
            print("❌ Нет успешных результатов")
            return pd.DataFrame()

        # Создание таблицы сравнения
        comparison_df = pd.DataFrame(results).sort_values('total_return', ascending=False)
        
        key_metrics = [
            'strategy_name', 'total_return', 'profit_loss', 'total_trades',
            'win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown'
        ]
        available_metrics = [col for col in key_metrics if col in comparison_df.columns]
        display_df = comparison_df[available_metrics].copy()

        print(f"\n🏆 РЕЙТИНГ СТРАТЕГИЙ:")
        print("=" * 100)
        print(display_df.to_string(index=False, float_format='%.2f'))

        if len(results) > 0:
            best_strategy = comparison_df.iloc[0]
            print(f"\n🥇 ЛУЧШАЯ СТРАТЕГИЯ: {best_strategy['strategy_name']}")
            print(f"   📈 Доходность: {best_strategy['total_return']:+.2f}%")
            print(f"   💰 Прибыль: ${best_strategy['profit_loss']:+,.2f}")
            print(f"   🎯 Винрейт: {best_strategy.get('win_rate', 0):.1f}%")

        print("=" * 100)
        return display_df

    def optimize_strategy(self,
                         strategy_name: str,
                         optimization_params: Dict[str, tuple],
                         exchange: str = "binance",
                         symbol: str = "BTCUSDT",
                         timeframe: str = "1d",
                         start_date: str = None,
                         end_date: str = None,
                         max_iterations: int = 100) -> pd.DataFrame:
        """Оптимизация параметров стратегии"""
        
        print(f"\n🔧 ОПТИМИЗАЦИЯ СТРАТЕГИИ: {strategy_name}")
        print("=" * 60)
        print(f"📈 Данные: {exchange}:{symbol} ({timeframe})")
        print(f"⚙️ Параметры оптимизации: {list(optimization_params.keys())}")

        if strategy_name not in self.strategies_registry:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")

        strategy_info = self.strategies_registry[strategy_name]
        strategy_class = strategy_info['class']

        # Настройка Cerebro для оптимизации
        cerebro = bt.Cerebro(optreturn=False)
        
        # Загрузка данных
        data_feed = self.data_manager.load_data(exchange, symbol, timeframe, start_date, end_date)
        cerebro.adddata(data_feed)
        
        # Настройка брокера
        cerebro.broker.setcash(self.initial_cash)
        comminfo = EnhancedCommissionInfo(
            commission=self.commission,
            spread=self.spread,
            slippage=self.slippage
        )
        cerebro.broker.addcommissioninfo(comminfo)
        
        # Добавление анализаторов
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Настройка параметров оптимизации
        opt_params = {}
        for param_name, (min_val, max_val, step) in optimization_params.items():
            if isinstance(min_val, float):
                # Для float параметров
                values = np.arange(min_val, max_val + step, step)
                opt_params[param_name] = [round(v, 4) for v in values]
            else:
                # Для int параметров
                opt_params[param_name] = range(int(min_val), int(max_val) + 1, int(step))

        cerebro.optstrategy(strategy_class, **opt_params)
        print(f"🚀 Запуск оптимизации...")

        # Запуск оптимизации
        optimization_results = cerebro.run(maxcpus=1)

        # Обработка результатов
        results_list = []
        for result in optimization_results:
            strategy_result = result[0]
            params = dict(strategy_result.params._getitems())

            final_value = strategy_result.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100

            # Получение метрик
            sharpe_ratio = 0
            total_trades = 0
            win_rate = 0
            
            try:
                sharpe_analysis = strategy_result.analyzers.sharpe.get_analysis()
                sharpe_ratio = sharpe_analysis.get('sharperatio', 0) or 0
            except:
                pass

            try:
                trades_analysis = strategy_result.analyzers.trades.get_analysis()
                total_dict = trades_analysis.get('total', {})
                won_dict = trades_analysis.get('won', {})
                
                total_trades = total_dict.get('total', 0)
                won_trades = won_dict.get('total', 0)
                win_rate = (won_trades / max(total_trades, 1)) * 100
            except:
                pass

            result_data = {
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'win_rate': win_rate,
                **{k: v for k, v in params.items() if k in optimization_params}
            }
            results_list.append(result_data)

        # Создание DataFrame с результатами
        results_df = pd.DataFrame(results_list).sort_values('total_return', ascending=False)

        print(f"\n🏆 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
        print("=" * 80)
        print(results_df.head(10).to_string(index=False, float_format='%.2f'))

        if not results_df.empty:
            best_result = results_df.iloc[0]
            print(f"\n🥇 ЛУЧШИЕ ПАРАМЕТРЫ:")
            for param in optimization_params.keys():
                print(f"   • {param}: {best_result[param]}")
            print(f"   📈 Доходность: {best_result['total_return']:+.2f}%")
            print(f"   📊 Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
            print(f"   🎯 Винрейт: {best_result['win_rate']:.1f}%")

        return results_df

    def _print_backtest_header(self, strategy_name: str, exchange: str, symbol: str, 
                              timeframe: str, start_date: str, end_date: str, params: Dict):
        """Вывод заголовка бэктеста"""
        print(f"\n🚀 ЗАПУСК БЭКТЕСТА: {strategy_name}")
        print("=" * 60)
        print(f"📈 Данные: {exchange}:{symbol} ({timeframe})")
        if start_date or end_date:
            print(f"📅 Период: {start_date or 'начало'} - {end_date or 'конец'}")
        print(f"💰 Начальный капитал: ${self.initial_cash:,}")
        print(f"💸 Комиссия: {self.commission:.3f} | Спред: {self.spread:.4f} | Проскальзывание: {self.slippage:.4f}")
        
        if params:
            print(f"⚙️ Параметры:")
            for param, value in params.items():
                print(f"   • {param}: {value}")
        print()

    def _add_analyzers(self, cerebro):
        """Добавление анализаторов"""
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    def _process_results(self, result, strategy_name: str, params: Dict, 
                        exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Обработка результатов бэктеста"""
        final_value = result.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100

        analysis_result = {
            'strategy_name': strategy_name,
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'profit_loss': final_value - self.initial_cash,
            'parameters': params
        }

        # Анализ сделок
        try:
            trades = result.analyzers.trades.get_analysis()
            analysis_result.update(self._analyze_trades(trades) if trades else self._empty_trades())
        except Exception:
            analysis_result.update(self._empty_trades())

        # Дополнительные метрики
        try:
            analysis_result.update(self._analyze_metrics(result))
        except Exception:
            analysis_result.update({
                'sharpe_ratio': 0, 'max_drawdown': 0, 'max_drawdown_period': 0, 'sqn': 0
            })

        return analysis_result

    def _analyze_trades(self, trades: Dict) -> Dict[str, Any]:
        """Анализ торговых операций"""
        total = trades.get('total', {})
        won = trades.get('won', {})
        lost = trades.get('lost', {})

        result = {
            'total_trades': total.get('total', 0),
            'won_trades': won.get('total', 0),
            'lost_trades': lost.get('total', 0),
            'won_pnl_total': won.get('pnl', {}).get('total', 0),
            'lost_pnl_total': lost.get('pnl', {}).get('total', 0)
        }

        # Производные метрики
        total_trades = result['total_trades']
        won_trades = result['won_trades']
        result['win_rate'] = (won_trades / max(total_trades, 1)) * 100

        gross_profit = abs(result['won_pnl_total'])
        gross_loss = abs(result['lost_pnl_total'])
        result['profit_factor'] = gross_profit / max(gross_loss, 1)

        return result

    def _empty_trades(self) -> Dict[str, Any]:
        """Пустой анализ сделок"""
        return {
            'total_trades': 0, 'won_trades': 0, 'lost_trades': 0,
            'win_rate': 0, 'profit_factor': 0, 'won_pnl_total': 0, 'lost_pnl_total': 0
        }

    def _analyze_metrics(self, result) -> Dict[str, Any]:
        """Анализ дополнительных метрик"""
        analysis = {}

        # Sharpe Ratio
        try:
            sharpe = result.analyzers.sharpe.get_analysis()
            analysis['sharpe_ratio'] = sharpe.get('sharperatio', 0) or 0
        except:
            analysis['sharpe_ratio'] = 0

        # Drawdown
        try:
            drawdown = result.analyzers.drawdown.get_analysis()
            analysis['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0) or 0
            analysis['max_drawdown_period'] = drawdown.get('max', {}).get('len', 0) or 0
        except:
            analysis['max_drawdown'] = 0
            analysis['max_drawdown_period'] = 0

        # SQN
        try:
            sqn = result.analyzers.sqn.get_analysis()
            analysis['sqn'] = sqn.get('sqn', 0) or 0
        except:
            analysis['sqn'] = 0

        return analysis

    def _print_results(self, results: Dict[str, Any]):
        """Вывод отформатированных результатов"""
        print("\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТИРОВАНИЯ")
        print("=" * 60)
        
        # Основные метрики
        print(f"💰 Начальный капитал:     ${results['initial_value']:,.2f}")
        print(f"💰 Финальный капитал:     ${results['final_value']:,.2f}")
        print(f"📈 Общая доходность:      {results['total_return']:+.2f}%")
        print(f"💵 Прибыль/Убыток:        ${results['profit_loss']:+,.2f}")

        # Торговые метрики
        if 'total_trades' in results:
            print(f"\n🔄 Всего сделок:          {results['total_trades']}")
            print(f"✅ Выигрышных сделок:     {results.get('won_trades', 0)}")
            print(f"❌ Проигрышных сделок:    {results.get('lost_trades', 0)}")
            print(f"🎯 Винрейт:               {results.get('win_rate', 0):.1f}%")
            print(f"⚖️ Profit Factor:         {results.get('profit_factor', 0):.2f}")

        # Дополнительные метрики
        if 'sharpe_ratio' in results:
            print(f"\n📊 Коэффициент Шарпа:     {results['sharpe_ratio']:.3f}")
            print(f"📉 Макс. просадка:        {results['max_drawdown']:.2f}%")
            print(f"🎖️ SQN:                   {results.get('sqn', 0):.2f}")

        print("=" * 60)

    def _plot_results(self, cerebro, strategy_name: str, exchange: str, symbol: str, timeframe: str):
        """Построение графика результатов"""
        try:
            print(f"\n📈 Построение графика...")
            cerebro.plot(figsize=(15, 8), style='candlestick', volume=False)
            plt.suptitle(f'{strategy_name} | {exchange}:{symbol} ({timeframe})', fontsize=16)
            plt.show()
        except Exception as e:
            print(f"⚠️ Ошибка построения графика: {e}")


# Пример использования
if __name__ == "__main__":
    # Создание бэктестера
    backtester = UniversalBacktester(
        initial_cash=100000,
        commission=0.001,  # 0.1%
        spread=0.0005,     # 0.05%
        slippage=0.0002    # 0.02%
    )
    
    # Просмотр доступных опций
    backtester.list_available_options()
    
    # Пример: запуск одной стратегии
    backtester.run_single_backtest(
        strategy_name="SafeProfitableBTCStrategy",
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1d"
    )
    
    # Пример: тестирование на множественных данных
    data_configs = [
        {"exchange": "binance", "symbol": "BTCUSDT", "timeframe": "1d"},
        {"exchange": "binance", "symbol": "ETHUSDT", "timeframe": "1d"},
        {"exchange": "binance", "symbol": "BTCUSDT", "timeframe": "4h"},
    ]
    backtester.run_multi_data_backtest("SafeProfitableBTCStrategy", data_configs)

    # Пример: сравнение стратегий
    backtester.compare_strategies()

    # Пример: оптимизация параметров
    optimization_params = {
        'ema_fast': (10, 20, 2),
        'ema_slow': (20, 30, 5),
        'rsi_period': (10, 20, 2)
    }
    backtester.optimize_strategy("SafeProfitableBTCStrategy", optimization_params)