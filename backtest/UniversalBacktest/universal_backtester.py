import os
import sys
import importlib.util
import inspect
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore')

# Добавляем пути к стратегиям
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


class SilentStrategyWrapper:
    """Обертка для подавления ошибок в стратегиях"""

    @classmethod
    def wrap_strategy(cls, strategy_class):
        """Создает обертку для стратегии с подавлением ошибок"""

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
                except (IndexError, TypeError, ZeroDivisionError):
                    self.__class__.error_count += 1
                except Exception:
                    pass

        return WrappedStrategy


class UniversalBacktester:
    """Универсальный бэктестер с автоматическим определением параметров стратегий"""

    def __init__(self, initial_cash: float = 100000, commission: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission
        self.strategies_registry = {}
        self.data_cache = {}

        print("🔍 Инициализация универсального бэктестера...")
        self._discover_strategies()
    
    def _discover_strategies(self):
        """Автоматическое обнаружение всех стратегий"""
        strategies_path = os.path.join(os.path.dirname(__file__), '../../strategies/TestStrategies/')
        
        if not os.path.exists(strategies_path):
            print(f"⚠️ Папка стратегий не найдена: {strategies_path}")
            return

        print(f"📁 Сканирую папку: {strategies_path}")

        for filename in os.listdir(strategies_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                self._load_strategies_from_module(filename[:-3], strategies_path)

    def _is_strategy_class(self, obj) -> bool:
        """Проверка класса стратегии"""
        return (inspect.isclass(obj) and issubclass(obj, bt.Strategy) and
                obj != bt.Strategy and not obj.__name__.startswith('_'))

    def _extract_strategy_params(self, strategy_class) -> Dict[str, Any]:
        """Извлечение параметров стратегии"""
        default_params = {}
        if not hasattr(strategy_class, 'params'):
            return default_params

        params_attr = getattr(strategy_class, 'params')
        if params_attr is None:
            return default_params

        # Универсальная обработка разных форматов params
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
        return default_params

    def _is_valid_param(self, name: str, value: Any) -> bool:
        """Проверка валидности параметра"""
        return (not callable(value) and not name.startswith('_') and
                name not in ['isdefault', 'notdefault'])

    def _load_strategies_from_module(self, module_name: str, module_path: str):
        """Загрузка стратегий из модуля"""
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(module_path, f"{module_name}.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            strategies_found = 0
            for name, obj in inspect.getmembers(module):
                if self._is_strategy_class(obj):
                    try:
                        default_params = self._extract_strategy_params(obj)

                        self.strategies_registry[name] = {
                            'class': obj, 'module': module_name, 'file': f"{module_name}.py",
                            'default_params': default_params,
                            'description': obj.__doc__ or f"Стратегия {name}"
                        }
                        strategies_found += 1
                        print(f"✅ Найдена стратегия: {name} (файл: {module_name}.py, параметров: {len(default_params)})")

                    except Exception as e:
                        print(f"⚠️ Ошибка обработки стратегии {name}: {e}")
            if strategies_found == 0:
                print(f"⚠️ В файле {module_name}.py стратегии не найдены")

        except Exception as e:
            print(f"❌ Ошибка загрузки модуля {module_name}: {e}")

    def list_strategies(self):
        """Показать все доступные стратегии"""
        print("\n📋 ДОСТУПНЫЕ СТРАТЕГИИ:")
        print("=" * 80)

        if not self.strategies_registry:
            print("❌ Стратегии не найдены!")
            return

        strategies_by_file = {}
        for name, info in self.strategies_registry.items():
            file_name = info['file']
            if file_name not in strategies_by_file:
                strategies_by_file[file_name] = []
            strategies_by_file[file_name].append((name, info))

        for file_name, strategies in strategies_by_file.items():
            print(f"\n📄 Файл: {file_name}")
            print("-" * 60)

            for i, (name, info) in enumerate(strategies, 1):
                print(f"   {i}. 🎯 {name}")
                print(f"      📝 Описание: {info['description'][:80]}...")

                if info['default_params']:
                    print(f"      ⚙️ Параметры ({len(info['default_params'])}):")
                    for param_name, param_value in list(info['default_params'].items())[:5]:
                        print(f"         • {param_name}: {param_value}")

                    if len(info['default_params']) > 5:
                        print(f"         ... и еще {len(info['default_params']) - 5} параметров")
                else:
                    print(f"      ⚙️ Параметры: Нет настраиваемых параметров")
                print()

        print(f"📊 Всего найдено: {len(self.strategies_registry)} стратегий в {len(strategies_by_file)} файлах")
        print("=" * 80)

    def load_data(self, data_path: str = None, timeframe: str = "1d") -> bt.feeds.PandasData:
        """Загрузка данных"""
        if data_path is None:
            data_path = f"../../data/binance/BTCUSDT/{timeframe}/2018_01_01-2025_01_01.csv"

        cache_key = f"{data_path}_{timeframe}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        full_path = os.path.join(os.path.dirname(__file__), data_path) if not os.path.isabs(data_path) else data_path
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Файл данных не найден: {full_path}")

        df = pd.read_csv(full_path)

        # Автоопределение колонок
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(x in col_lower for x in ['timestamp', 'date', 'time']):
                column_mapping[col] = 'datetime'
            elif col_lower in ['o', 'open']: column_mapping[col] = 'open'
            elif col_lower in ['h', 'high']: column_mapping[col] = 'high'
            elif col_lower in ['l', 'low']: column_mapping[col] = 'low'
            elif col_lower in ['c', 'close']: column_mapping[col] = 'close'
            elif col_lower in ['v', 'volume', 'vol']: column_mapping[col] = 'volume'

        df = df.rename(columns=column_mapping)

        # Обработка времени
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        if 'volume' not in df.columns:
            df['volume'] = 1000

        df = df[required_cols + ['volume']].dropna()
        df = df[(df[required_cols] > 0).all(axis=1)]
        df.sort_index(inplace=True)
        
        print(f"✅ Загружено {len(df)} записей из {os.path.basename(full_path)}")
        data_feed = bt.feeds.PandasData(dataname=df)
        self.data_cache[cache_key] = data_feed
        return data_feed

    def run_backtest(self, strategy_name: str, strategy_params: Dict[str, Any] = None,
                    data_path: str = None, timeframe: str = "1d", show_plot: bool = True,
                    verbose: bool = True, suppress_strategy_errors: bool = False) -> Dict[str, Any]:
        """Запуск бэктестирования"""
        if strategy_name not in self.strategies_registry:
            available = list(self.strategies_registry.keys())
            raise ValueError(f"Стратегия '{strategy_name}' не найдена. Доступные: {available}")
        
        strategy_info = self.strategies_registry[strategy_name]
        strategy_class = strategy_info['class']
        final_params = strategy_info['default_params'].copy()
        if strategy_params:
            final_params.update(strategy_params)

        if verbose:
            print(f"\n🚀 ЗАПУСК БЭКТЕСТА: {strategy_name}")
            print("=" * 60)
            print(f"💰 Начальный капитал: ${self.initial_cash:,}")
            print(f"💸 Комиссия: {self.commission:.3f}")
            print(f"📊 Таймфрейм: {timeframe}")
            if final_params:
                print(f"⚙️ Параметры:")
                for param, value in final_params.items():
                    print(f"   • {param}: {value}")
            print()

        try:
            cerebro = bt.Cerebro()

            # Создание стратегии
            if suppress_strategy_errors:
                wrapped_class = SilentStrategyWrapper.wrap_strategy(strategy_class)
                cerebro.addstrategy(wrapped_class, **final_params)
            else:
                cerebro.addstrategy(strategy_class, **final_params)

            cerebro.adddata(self.load_data(data_path, timeframe))
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Анализаторы
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

            try:
                results = cerebro.run()
                if not results:
                    raise RuntimeError("Стратегия не вернула результатов")
                result = results[0]

                if (suppress_strategy_errors and hasattr(result, 'error_count') and
                    result.error_count > 0 and verbose):
                    print(f"⚠️ Обнаружено {result.error_count} ошибок индексации (подавлено)")

            except Exception as e:
                if verbose:
                    print(f"❌ Ошибка в стратегии {strategy_name}: {str(e)}")
                raise RuntimeError(f"Ошибка выполнения стратегии: {str(e)}")

            # Подготовка результатов
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100

            analysis_result = {
                'strategy_name': strategy_name, 'initial_value': self.initial_cash,
                'final_value': final_value, 'total_return': total_return,
                'profit_loss': final_value - self.initial_cash, 'parameters': final_params
            }

            # Анализ сделок и метрик
            try:
                trades = result.analyzers.trades.get_analysis()
                analysis_result.update(self._analyze_trades(trades) if trades else self._empty_trades())
                analysis_result.update(self._detailed_analysis(result))
            except Exception as e:
                if verbose:
                    print(f"⚠️ Ошибка анализа: {e}")
                analysis_result.update(self._empty_trades())
                analysis_result.update({'sharpe_ratio': 0, 'max_drawdown': 0})

            if verbose:
                self._print_results(analysis_result)
            if show_plot:
                self._plot_results(cerebro, strategy_name)

            return analysis_result

        except Exception as e:
            if verbose:
                print(f"❌ Ошибка выполнения стратегии {strategy_name}: {str(e)}")
            raise e

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

        total_trades = result['total_trades']
        won_trades = result['won_trades']
        result['win_rate'] = (won_trades / max(total_trades, 1)) * 100

        gross_profit = abs(result['won_pnl_total'])
        gross_loss = abs(result['lost_pnl_total'])
        result['profit_factor'] = gross_profit / max(gross_loss, 1)
        
        return result

    def _empty_trades(self) -> Dict[str, Any]:
        """Пустые торговые метрики"""
        return {'total_trades': 0, 'won_trades': 0, 'lost_trades': 0,
                'win_rate': 0, 'profit_factor': 0}

    def _detailed_analysis(self, result) -> Dict[str, Any]:
        """Детальный анализ результатов"""
        analysis = {}
        try:
            sharpe = result.analyzers.sharpe.get_analysis()
            analysis['sharpe_ratio'] = sharpe.get('sharperatio', 0) or 0
        except:
            analysis['sharpe_ratio'] = 0

        try:
            drawdown = result.analyzers.drawdown.get_analysis()
            analysis['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0) or 0
            analysis['max_drawdown_period'] = drawdown.get('max', {}).get('len', 0) or 0
        except:
            analysis['max_drawdown'] = 0
            analysis['max_drawdown_period'] = 0

        try:
            returns = result.analyzers.returns.get_analysis()
            analysis['total_returns'] = (returns.get('rtot', 0) or 0) * 100
            analysis['average_returns'] = (returns.get('ravg', 0) or 0) * 100
        except:
            analysis['total_returns'] = 0
            analysis['average_returns'] = 0

        return analysis

    def _print_results(self, results: Dict[str, Any]):
        """Вывод результатов"""
        print("\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТИРОВАНИЯ")
        print("=" * 60)
        print(f"💰 Начальный капитал:     ${results['initial_value']:,.2f}")
        print(f"💰 Финальный капитал:     ${results['final_value']:,.2f}")
        print(f"📈 Общая доходность:      {results['total_return']:+.2f}%")
        print(f"💵 Прибыль/Убыток:        ${results['profit_loss']:+,.2f}")

        if 'total_trades' in results:
            print(f"\n🔄 Всего сделок:          {results['total_trades']}")
            print(f"✅ Выигрышных сделок:     {results.get('won_trades', 0)}")
            print(f"❌ Проигрышных сделок:    {results.get('lost_trades', 0)}")
            print(f"🎯 Винрейт:               {results.get('win_rate', 0):.1f}%")
            print(f"⚖️ Profit Factor:         {results.get('profit_factor', 0):.2f}")

        if 'sharpe_ratio' in results:
            print(f"\n📊 Коэффициент Шарпа:     {results['sharpe_ratio']:.3f}")
            print(f"📉 Макс. просадка:        {results['max_drawdown']:.2f}%")
        print("=" * 60)

    def _plot_results(self, cerebro, strategy_name: str):
        """Построение графиков"""
        try:
            print(f"\n📈 Построение графика для {strategy_name}...")
            cerebro.plot(figsize=(15, 8), style='candlestick', volume=False)
            plt.suptitle(f'Backtest Results: {strategy_name}', fontsize=16)
            plt.show()
        except Exception as e:
            print(f"⚠️ Ошибка построения графика: {e}")

    def compare_strategies(self, strategy_names: List[str] = None,
                          custom_params: Dict[str, Dict[str, Any]] = None,
                          data_path: str = None, timeframe: str = "1d",
                          skip_errors: bool = True, suppress_strategy_errors: bool = True) -> pd.DataFrame:
        """Сравнение стратегий"""
        if strategy_names is None:
            strategy_names = list(self.strategies_registry.keys())
        if custom_params is None:
            custom_params = {}

        print(f"\n🔍 СРАВНЕНИЕ СТРАТЕГИЙ")
        print("=" * 80)
        print(f"📊 Стратегий к тестированию: {len(strategy_names)}")
        print(f"⏱️ Таймфрейм: {timeframe}")
        if suppress_strategy_errors:
            print("🔇 Режим: Ошибки стратегий подавлены")
        print()

        results = []
        failed_strategies = []

        for i, strategy_name in enumerate(strategy_names, 1):
            if strategy_name not in self.strategies_registry:
                print(f"❌ Стратегия '{strategy_name}' не найдена, пропускаю...")
                continue

            print(f"⏳ [{i}/{len(strategy_names)}] Тестирование: {strategy_name}")
            try:
                params = custom_params.get(strategy_name, {})
                result = self.run_backtest(
                    strategy_name=strategy_name, strategy_params=params,
                    data_path=data_path, timeframe=timeframe, show_plot=False,
                    verbose=False, suppress_strategy_errors=suppress_strategy_errors
                )
                results.append(result)
                print(f"✅ Завершено: {result['total_return']:+.2f}% | {result.get('total_trades', 0)} сделок")
            except Exception as e:
                error_msg = str(e)
                if "array index out of range" in error_msg:
                    print(f"❌ Ошибка в {strategy_name}: Ошибка выполнения стратегии")
                else:
                    print(f"❌ Ошибка в {strategy_name}: {error_msg}")
                failed_strategies.append(strategy_name)
                if not skip_errors:
                    raise e

        if failed_strategies:
            print(f"\n⚠️ Стратегии с ошибками ({len(failed_strategies)}):")
            for strategy in failed_strategies:
                print(f"   • {strategy}")

        if not results:
            print("❌ Нет успешных результатов для сравнения")
            return pd.DataFrame()

        comparison_df = pd.DataFrame(results).sort_values('total_return', ascending=False)
        key_metrics = ['strategy_name', 'total_return', 'profit_loss', 'total_trades',
                      'win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown']
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

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Получить информацию о стратегии"""
        if strategy_name not in self.strategies_registry:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")
        return self.strategies_registry[strategy_name]

    def optimize_strategy(self, strategy_name: str, optimization_params: Dict[str, tuple],
                         data_path: str = None, timeframe: str = "1d", max_iterations: int = 100) -> pd.DataFrame:
        """Оптимизация параметров стратегии"""
        print(f"\n🔧 ОПТИМИЗАЦИЯ СТРАТЕГИИ: {strategy_name}")
        print("=" * 60)

        if strategy_name not in self.strategies_registry:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")

        strategy_info = self.strategies_registry[strategy_name]
        strategy_class = strategy_info['class']

        cerebro = bt.Cerebro(optreturn=False)
        data_feed = self.load_data(data_path, timeframe)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        opt_params = {}
        for param_name, (min_val, max_val, step) in optimization_params.items():
            opt_params[param_name] = range(int(min_val), int(max_val), int(step))

        cerebro.optstrategy(strategy_class, **opt_params)
        print(f"🚀 Запуск оптимизации с параметрами: {optimization_params}")

        optimization_results = cerebro.run(maxcpus=1)

        results_list = []
        for result in optimization_results:
            strategy_result = result[0]
            params = strategy_result.params._getitems()

            final_value = strategy_result.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100

            sharpe_ratio = 0
            try:
                sharpe_analysis = strategy_result.analyzers.sharpe.get_analysis()
                sharpe_ratio = sharpe_analysis.get('sharperatio', 0) or 0
            except:
                pass

            result_data = {
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                **{k: v for k, v in params if k in optimization_params}
            }
            results_list.append(result_data)

        results_df = pd.DataFrame(results_list).sort_values('total_return', ascending=False)

        print(f"\n🏆 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
        print("=" * 80)
        print(results_df.head(10).to_string(index=False, float_format='%.2f'))

        best_result = results_df.iloc[0]
        print(f"\n🥇 ЛУЧШИЕ ПАРАМЕТРЫ:")
        for param in optimization_params.keys():
            print(f"   • {param}: {best_result[param]}")
        print(f"   📈 Доходность: {best_result['total_return']:+.2f}%")
        print(f"   📊 Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")

        return results_df


# Пример использования
if __name__ == "__main__":
    backtester = UniversalBacktester(initial_cash=100000, commission=0.001)
    backtester.list_strategies()
    # backtester.run_backtest("SafeProfitableBTCStrategy")
    # backtester.compare_strategies()
if __name__ == "__main__":
    backtester = UniversalBacktester(initial_cash=100000, commission=0.001)
    backtester.list_strategies()
    # backtester.run_backtest("SafeProfitableBTCStrategy")
    # backtester.compare_strategies()

    def _detailed_analysis(self, result) -> Dict[str, Any]:
        """Детальный анализ результатов"""
        analysis = {}
        
        # Sharpe Ratio
        try:
            sharpe = result.analyzers.sharpe.get_analysis()
            analysis['sharpe_ratio'] = sharpe.get('sharperatio', 0) or 0
        except:
            analysis['sharpe_ratio'] = 0

        # DrawDown
        try:
            drawdown = result.analyzers.drawdown.get_analysis()
            analysis['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0) or 0
            analysis['max_drawdown_period'] = drawdown.get('max', {}).get('len', 0) or 0
        except:
            analysis['max_drawdown'] = 0
            analysis['max_drawdown_period'] = 0

        # Returns
        try:
            returns = result.analyzers.returns.get_analysis()
            analysis['total_returns'] = (returns.get('rtot', 0) or 0) * 100
            analysis['average_returns'] = (returns.get('ravg', 0) or 0) * 100
        except:
            analysis['total_returns'] = 0
            analysis['average_returns'] = 0

        return analysis
    
    def _print_results(self, results: Dict[str, Any]):
        """Вывод результатов на консоль"""
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

        print("=" * 60)

    def _plot_results(self, cerebro, strategy_name: str):
        """Построение графиков"""
        try:
            print(f"\n📈 Построение графика для {strategy_name}...")
            cerebro.plot(figsize=(15, 8), style='candlestick', volume=False)
            plt.suptitle(f'Backtest Results: {strategy_name}', fontsize=16)
            plt.show()
        except Exception as e:
            print(f"⚠️ Ошибка построения графика: {e}")

    def compare_strategies(self,
                          strategy_names: List[str] = None,
                          custom_params: Dict[str, Dict[str, Any]] = None,
                          data_path: str = None,
                          timeframe: str = "1d",
                          skip_errors: bool = True) -> pd.DataFrame:
        """
        Сравнение стратегий

        Args:
            strategy_names: список имен стратегий для сравнения (если None - все стратегии)
            custom_params: словарь кастомных параметров для стратегий
            data_path: путь к данным
            timeframe: таймфрейм данных
            skip_errors: пропускать стратегии с ошибками
        """

        if strategy_names is None:
            strategy_names = list(self.strategies_registry.keys())

        if custom_params is None:
            custom_params = {}

        print(f"\n🔍 СРАВНЕНИЕ СТРАТЕГИЙ")
        print("=" * 80)
        print(f"📊 Стратегий к тестированию: {len(strategy_names)}")
        print(f"⏱️ Таймфрейм: {timeframe}")
        print()

        results = []
        failed_strategies = []

        for i, strategy_name in enumerate(strategy_names, 1):
            if strategy_name not in self.strategies_registry:
                print(f"❌ Стратегия '{strategy_name}' не найдена, пропускаю...")
                continue

            print(f"⏳ [{i}/{len(strategy_names)}] Тестирование: {strategy_name}")

            try:
                # Используем кастомные параметры если есть
                params = custom_params.get(strategy_name, {})

                result = self.run_backtest(
                    strategy_name=strategy_name,
                    strategy_params=params,
                    data_path=data_path,
                    timeframe=timeframe,
                    show_plot=False,
                    verbose=False
                )
                results.append(result)
                print(f"✅ Завершено: {result['total_return']:+.2f}% | {result.get('total_trades', 0)} сделок")

            except Exception as e:
                error_msg = str(e)
                if "array index out of range" in error_msg:
                    print(f"❌ Ошибка в {strategy_name}: Ошибка выполнения стратегии: {error_msg}")
                else:
                    print(f"❌ Ошибка в {strategy_name}: {error_msg}")

                failed_strategies.append(strategy_name)

                if not skip_errors:
                    raise e
                continue

        if failed_strategies:
            print(f"\n⚠️ Стратегии с ошибками ({len(failed_strategies)}):")
            for strategy in failed_strategies:
                print(f"   • {strategy}")

        if not results:
            print("❌ Нет успешных результатов для сравнения")
            return pd.DataFrame()

        # Создаем DataFrame для сравнения
        comparison_df = pd.DataFrame(results)

        # Сортируем по доходности
        comparison_df = comparison_df.sort_values('total_return', ascending=False)

        # Выбираем ключевые метрики для отображения
        key_metrics = [
            'strategy_name', 'total_return', 'profit_loss', 'total_trades',
            'win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown'
        ]

        available_metrics = [col for col in key_metrics if col in comparison_df.columns]
        display_df = comparison_df[available_metrics].copy()

        print(f"\n🏆 РЕЙТИНГ СТРАТЕГИЙ:")
        print("=" * 100)
        print(display_df.to_string(index=False, float_format='%.2f'))

        # Лучшая стратегия
        if len(results) > 0:
            best_strategy = comparison_df.iloc[0]
            print(f"\n🥇 ЛУЧШАЯ СТРАТЕГИЯ: {best_strategy['strategy_name']}")
            print(f"   📈 Доходность: {best_strategy['total_return']:+.2f}%")
            print(f"   💰 Прибыль: ${best_strategy['profit_loss']:+,.2f}")
            print(f"   🎯 Винрейт: {best_strategy.get('win_rate', 0):.1f}%")

        print("=" * 100)

        return display_df

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Получить подробную информацию о стратегии"""
        if strategy_name not in self.strategies_registry:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")

        return self.strategies_registry[strategy_name]

    def optimize_strategy(self,
                         strategy_name: str,
                         optimization_params: Dict[str, tuple],
                         data_path: str = None,
                         timeframe: str = "1d",
                         max_iterations: int = 100) -> pd.DataFrame:
        """
        Оптимизация параметров стратегии

        Args:
            strategy_name: имя стратегии для оптимизации
            optimization_params: словарь параметров для оптимизации в формате {param_name: (min, max, step)}
            data_path: путь к данным
            timeframe: таймфрейм
            max_iterations: максимальное количество итераций
        """
        print(f"\n🔧 ОПТИМИЗАЦИЯ СТРАТЕГИИ: {strategy_name}")
        print("=" * 60)

        if strategy_name not in self.strategies_registry:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")

        strategy_info = self.strategies_registry[strategy_name]
        strategy_class = strategy_info['class']

        # Настройка Cerebro для оптимизации
        cerebro = bt.Cerebro(optreturn=False)

        # Добавляем данные
        data_feed = self.load_data(data_path, timeframe)
        cerebro.adddata(data_feed)

        # Настройки брокера
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # Добавляем анализаторы
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        # Добавляем стратегию с параметрами для оптимизации
        opt_params = {}
        for param_name, (min_val, max_val, step) in optimization_params.items():
            opt_params[param_name] = range(int(min_val), int(max_val), int(step))

        cerebro.optstrategy(strategy_class, **opt_params)

        print(f"🚀 Запуск оптимизации с параметрами: {optimization_params}")

        # Запуск оптимизации
        optimization_results = cerebro.run(maxcpus=1)

        # Обработка результатов
        results_list = []
        for result in optimization_results:
            strategy_result = result[0]
            params = strategy_result.params._getitems()

            final_value = strategy_result.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100

            sharpe_ratio = 0
            try:
                sharpe_analysis = strategy_result.analyzers.sharpe.get_analysis()
                sharpe_ratio = sharpe_analysis.get('sharperatio', 0) or 0
            except:
                pass

            result_data = {
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                **{k: v for k, v in params if k in optimization_params}
            }
            results_list.append(result_data)

        # Создаем DataFrame и сортируем по доходности
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('total_return', ascending=False)

        print(f"\n🏆 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
        print("=" * 80)
        print(results_df.head(10).to_string(index=False, float_format='%.2f'))

        best_result = results_df.iloc[0]
        print(f"\n🥇 ЛУЧШИЕ ПАРАМЕТРЫ:")
        for param in optimization_params.keys():
            print(f"   • {param}: {best_result[param]}")
        print(f"   📈 Доходность: {best_result['total_return']:+.2f}%")
        print(f"   📊 Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")

        return results_df


# Пример использования
if __name__ == "__main__":
    # Создаем бэктестер
    backtester = UniversalBacktester(initial_cash=100000, commission=0.001)

    # Показываем доступные стратегии
    backtester.list_strategies()

    # Можно тестировать любую стратегию
    # backtester.run_backtest("SafeProfitableBTCStrategy")

    # Или сравнить все стратегии
    # backtester.compare_strategies()

    # Или оптимизировать параметры стратегии
    # optimization_params = {
    #     'ema_fast': (10, 20, 2),
    #     'ema_slow': (20, 30, 5),
    #     'rsi_period': (10, 20, 2)
    # }
    # backtester.optimize_strategy("SafeProfitableBTCStrategy", optimization_params)


class UniversalBacktester:
    """
    Универсальный бэктестер с автоматическим определением параметров стратегий
    """

    def __init__(self, initial_cash: float = 100000, commission: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission
        self.strategies_registry = {}
        self.data_cache = {}

        print("🔍 Инициализация универсального бэктестера...")
        # Автоматически загружаем все доступные стратегии
        self._discover_strategies()

    def _discover_strategies(self):
        """Автоматическое обнаружение всех стратегий в проекте"""
        strategies_path = os.path.join(os.path.dirname(__file__), '../../strategies/TestStrategies/')

        if not os.path.exists(strategies_path):
            print(f"⚠️ Папка стратегий не найдена: {strategies_path}")
            return

        print(f"📁 Сканирую папку: {strategies_path}")

        for filename in os.listdir(strategies_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # убираем .py
                self._load_strategies_from_module(module_name, strategies_path)

    def _load_strategies_from_module(self, module_name: str, module_path: str):
        """Загрузка стратегий из модуля с извлечением параметров"""
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                os.path.join(module_path, f"{module_name}.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            strategies_found = 0

            # Находим все классы стратегий в модуле
            for name, obj in inspect.getmembers(module):
                if self._is_strategy_class(obj):
                    try:
                        # Извлекаем параметры стратегии
                        default_params = self._extract_strategy_params(obj)

                        self.strategies_registry[name] = {
                            'class': obj,
                            'module': module_name,
                            'file': f"{module_name}.py",
                            'default_params': default_params,
                            'description': obj.__doc__ or f"Стратегия {name}"
                        }
                        strategies_found += 1
                        print(f"✅ Найдена стратегия: {name} (файл: {module_name}.py, параметров: {len(default_params)})")

                    except Exception as e:
                        print(f"⚠️ Ошибка обработки стратегии {name}: {e}")
                        continue

            if strategies_found == 0:
                print(f"⚠️ В файле {module_name}.py стратегии не найдены")

        except Exception as e:
            print(f"❌ Ошибка загрузки модуля {module_name}: {e}")

    def _is_strategy_class(self, obj) -> bool:
        """Проверяем, является ли объект классом стратегии"""
        return (
            inspect.isclass(obj) and
            issubclass(obj, bt.Strategy) and
            obj != bt.Strategy and
            not obj.__name__.startswith('_') and
            hasattr(obj, '__module__')  # Убеждаемся что это не встроенный класс
        )

    def _extract_strategy_params(self, strategy_class) -> Dict[str, Any]:
        """Извлекаем параметры по умолчанию из стратегии"""
        default_params = {}

        try:
            # Получаем параметры из атрибута params
            if hasattr(strategy_class, 'params'):
                params_attr = getattr(strategy_class, 'params')

                # Проверяем различные форматы params
                if params_attr is None:
                    return default_params

                # Если params это кортеж кортежей
                if isinstance(params_attr, tuple):
                    for param in params_attr:
                        if isinstance(param, tuple) and len(param) >= 2:
                            param_name = param[0]
                            param_value = param[1]
                            # Фильтруем внутренние функции backtrader
                            if not callable(param_value) and not param_name.startswith('_') and param_name not in ['isdefault', 'notdefault']:
                                default_params[param_name] = param_value

                # Если params это список
                elif isinstance(params_attr, list):
                    for param in params_attr:
                        if isinstance(param, tuple) and len(param) >= 2:
                            param_name = param[0]
                            param_value = param[1]
                            # Фильтруем внутренние функции backtrader
                            if not callable(param_value) and not param_name.startswith('_') and param_name not in ['isdefault', 'notdefault']:
                                default_params[param_name] = param_value

                # Если params это словарь
                elif isinstance(params_attr, dict):
                    for param_name, param_value in params_attr.items():
                        # Фильтруем внутренние функции backtrader
                        if not callable(param_value) and not param_name.startswith('_') and param_name not in ['isdefault', 'notdefault']:
                            default_params[param_name] = param_value

                # Если params это класс параметров (backtrader стиль)
                elif hasattr(params_attr, '__dict__'):
                    for attr_name in dir(params_attr):
                        if not attr_name.startswith('_') and attr_name not in ['isdefault', 'notdefault']:
                            attr_value = getattr(params_attr, attr_name)
                            # Фильтруем внутренние функции backtrader
                            if not callable(attr_value):
                                default_params[attr_name] = attr_value

        except Exception as e:
            print(f"⚠️ Ошибка извлечения параметров: {e}")

        return default_params

    def _load_strategies_from_module(self, module_name: str, module_path: str):
        """Загрузка стратегий из модуля с извлечением параметров"""
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                os.path.join(module_path, f"{module_name}.py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Добавляем модуль в sys.modules
            spec.loader.exec_module(module)

            strategies_found = 0

            # Находим все классы стратегий в модуле
            for name, obj in inspect.getmembers(module):
                if self._is_strategy_class(obj):
                    try:
                        # Извлекаем параметры стратегии
                        default_params = self._extract_strategy_params(obj)

                        self.strategies_registry[name] = {
                            'class': obj,
                            'module': module_name,
                            'module_obj': module,  # Сохраняем ссылку на модуль
                            'file': f"{module_name}.py",
                            'default_params': default_params,
                            'description': obj.__doc__ or f"Стратегия {name}"
                        }
                        strategies_found += 1
                        print(f"✅ Найдена стратегия: {name} (файл: {module_name}.py, параметров: {len(default_params)})")

                    except Exception as e:
                        print(f"⚠️ Ошибка обработки стратегии {name}: {e}")
                        continue

            if strategies_found == 0:
                print(f"⚠️ В файле {module_name}.py стратегии не найдены")

        except Exception as e:
            print(f"❌ Ошибка загрузки модуля {module_name}: {e}")

    def list_strategies(self):
        """Показать все доступные стратегии с их параметрами"""
        print("\n📋 ДОСТУПНЫЕ СТРАТЕГИИ:")
        print("=" * 80)

        if not self.strategies_registry:
            print("❌ Стратегии не найдены!")
            print("💡 Убедитесь что в папке strategies/TestStrategies/ есть .py файлы со стратегиями")
            return

        # Группируем стратегии по файлам
        strategies_by_file = {}
        for name, info in self.strategies_registry.items():
            file_name = info['file']
            if file_name not in strategies_by_file:
                strategies_by_file[file_name] = []
            strategies_by_file[file_name].append((name, info))

        for file_name, strategies in strategies_by_file.items():
            print(f"\n📄 Файл: {file_name}")
            print("-" * 60)

            for i, (name, info) in enumerate(strategies, 1):
                print(f"   {i}. 🎯 {name}")
                print(f"      📝 Описание: {info['description'][:80]}...")

                if info['default_params']:
                    print(f"      ⚙️ Параметры ({len(info['default_params'])}):")
                    for param_name, param_value in list(info['default_params'].items())[:5]:  # Показываем первые 5
                        print(f"         • {param_name}: {param_value}")

                    if len(info['default_params']) > 5:
                        print(f"         ... и еще {len(info['default_params']) - 5} параметров")
                else:
                    print(f"      ⚙️ Параметры: Нет настраиваемых параметров")
                print()

        print(f"📊 Всего найдено: {len(self.strategies_registry)} стратегий в {len(strategies_by_file)} файлах")
        print("=" * 80)

    def load_data(self, data_path: str = None, timeframe: str = "1d") -> bt.feeds.PandasData:
        """Загрузка данных с автоматическим определением формата"""
        if data_path is None:
            data_path = f"../../data/binance/BTCUSDT/{timeframe}/2018_01_01-2025_01_01.csv"

        # Проверяем кэш
        cache_key = f"{data_path}_{timeframe}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Формируем полный путь
        if not os.path.isabs(data_path):
            full_path = os.path.join(os.path.dirname(__file__), data_path)
        else:
            full_path = data_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Файл данных не найден: {full_path}")

        # Загружаем данные
        df = pd.read_csv(full_path)

        # Автоматическое определение колонок
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'timestamp' in col_lower or 'date' in col_lower or 'time' in col_lower:
                column_mapping[col] = 'datetime'
            elif col_lower in ['o', 'open']:
                column_mapping[col] = 'open'
            elif col_lower in ['h', 'high']:
                column_mapping[col] = 'high'
            elif col_lower in ['l', 'low']:
                column_mapping[col] = 'low'
            elif col_lower in ['c', 'close']:
                column_mapping[col] = 'close'
            elif col_lower in ['v', 'volume', 'vol']:
                column_mapping[col] = 'volume'

        # Переименовываем колонки
        df = df.rename(columns=column_mapping)

        # Обработка временных меток
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # Проверяем обязательные колонки
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        # Добавляем volume если отсутствует
        if 'volume' not in df.columns:
            df['volume'] = 1000  # Значение по умолчанию

        # Очистка данных
        df = df[required_cols + ['volume']].dropna()
        df = df[(df[required_cols] > 0).all(axis=1)]  # Убираем отрицательные цены
        df.sort_index(inplace=True)

        print(f"✅ Загружено {len(df)} записей из {os.path.basename(full_path)}")

        # Создаем объект данных для backtrader
        data_feed = bt.feeds.PandasData(dataname=df)

        # Кэшируем
        self.data_cache[cache_key] = data_feed
        return data_feed

    def run_backtest(self,
                    strategy_name: str,
                    strategy_params: Dict[str, Any] = None,
                    data_path: str = None,
                    timeframe: str = "1d",
                    show_plot: bool = True,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Запуск бэктестирования для выбранной стратегии
        """

        if strategy_name not in self.strategies_registry:
            available = list(self.strategies_registry.keys())
            raise ValueError(f"Стратегия '{strategy_name}' не найдена. Доступные: {available}")

        strategy_info = self.strategies_registry[strategy_name]
        strategy_class = strategy_info['class']

        # Объединяем параметры: по умолчанию + пользовательские
        final_params = strategy_info['default_params'].copy()
        if strategy_params:
            final_params.update(strategy_params)

        if verbose:
            print(f"\n🚀 ЗАПУСК БЭКТЕСТА: {strategy_name}")
            print("=" * 60)
            print(f"💰 Начальный капитал: ${self.initial_cash:,}")
            print(f"💸 Комиссия: {self.commission:.3f}")
            print(f"📊 Таймфрейм: {timeframe}")
            if final_params:
                print(f"⚙️ Параметры:")
                for param, value in final_params.items():
                    print(f"   • {param}: {value}")
            print()

        try:
            # Настройка Cerebro
            cerebro = bt.Cerebro()

            # Добавляем стратегию с параметрами
            cerebro.addstrategy(strategy_class, **final_params)

            # Добавляем данные
            data_feed = self.load_data(data_path, timeframe)
            cerebro.adddata(data_feed)

            # Настройки брокера
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Добавляем анализаторы
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

            # Запуск с защитой от ошибок
            try:
                results = cerebro.run()
                if not results:
                    raise RuntimeError("Стратегия не вернула результатов")
                result = results[0]
            except IndexError as e:
                if verbose:
                    print(f"❌ Ошибка индекса в стратегии {strategy_name}: {str(e)}")
                    print("💡 Возможные причины:")
                    print("   - Недостаточно данных для расчета индикаторов")
                    print("   - Обращение к данным за пределами массива")
                    print("   - Неправильная обработка первых/последних периодов")
                    print("   - Попытка доступа к данным до их инициализации")
                raise RuntimeError(f"Ошибка выполнения стратегии: {str(e)}")
            except Exception as e:
                if verbose:
                    print(f"❌ Общая ошибка в стратегии {strategy_name}: {str(e)}")
                    print("💡 Рекомендации:")
                    print("   - Проверьте логику стратегии на корректность")
                    print("   - Убедитесь в правильной обработке граничных случаев")
                    print("   - Добавьте проверки на существование данных")
                raise RuntimeError(f"Ошибка выполнения стратегии: {str(e)}")

            # Подготовка результатов
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100

            analysis_result = {
                'strategy_name': strategy_name,
                'initial_value': self.initial_cash,
                'final_value': final_value,
                'total_return': total_return,
                'profit_loss': final_value - self.initial_cash,
                'parameters': final_params
            }

            # Анализ сделок
            try:
                trades = result.analyzers.trades.get_analysis()
                if trades:
                    analysis_result.update(self._analyze_trades(trades))
                else:
                    # Добавляем пустые значения если сделок нет
                    analysis_result.update({
                        'total_trades': 0,
                        'won_trades': 0,
                        'lost_trades': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'won_pnl_total': 0,
                        'lost_pnl_total': 0
                    })
            except Exception as e:
                if verbose:
                    print(f"⚠️ Ошибка анализа сделок: {e}")
                analysis_result.update({
                    'total_trades': 0,
                    'won_trades': 0,
                    'lost_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0
                })

            # Дополнительные метрики
            try:
                analysis_result.update(self._detailed_analysis(result))
            except Exception as e:
                if verbose:
                    print(f"⚠️ Ошибка детального анализа: {e}")
                analysis_result.update({
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'max_drawdown_period': 0
                })

            # Вывод результатов
            if verbose:
                self._print_results(analysis_result)

            # График
            if show_plot:
                self._plot_results(cerebro, strategy_name)

            return analysis_result

        except Exception as e:
            if verbose:
                print(f"❌ Ошибка выполнения стратегии {strategy_name}: {str(e)}")
                import traceback
                traceback.print_exc()
            raise e

    def _analyze_trades(self, trades: Dict) -> Dict[str, Any]:
        """Анализ торговых операций"""
        result = {}

        # Общие сделки
        if 'total' in trades:
            total = trades['total']
            result['total_trades'] = total.get('total', 0)
            result['open_trades'] = total.get('open', 0)
            result['closed_trades'] = total.get('closed', 0)
        else:
            result['total_trades'] = 0
            result['open_trades'] = 0
            result['closed_trades'] = 0

        # Выигрышные сделки
        if 'won' in trades:
            won = trades['won']
            result['won_trades'] = won.get('total', 0)
            result['won_pnl_total'] = won.get('pnl', {}).get('total', 0)
            result['won_pnl_average'] = won.get('pnl', {}).get('average', 0)
        else:
            result['won_trades'] = 0
            result['won_pnl_total'] = 0
            result['won_pnl_average'] = 0

        # Проигрышные сделки
        if 'lost' in trades:
            lost = trades['lost']
            result['lost_trades'] = lost.get('total', 0)
            result['lost_pnl_total'] = lost.get('pnl', {}).get('total', 0)
            result['lost_pnl_average'] = lost.get('pnl', {}).get('average', 0)
        else:
            result['lost_trades'] = 0
            result['lost_pnl_total'] = 0
            result['lost_pnl_average'] = 0

        # Вычисляем производные метрики
        total_trades = result.get('total_trades', 0)
        won_trades = result.get('won_trades', 0)
        result['win_rate'] = (won_trades / max(total_trades, 1)) * 100

        # Profit Factor
        gross_profit = abs(result.get('won_pnl_total', 0))
        gross_loss = abs(result.get('lost_pnl_total', 0))
        result['profit_factor'] = gross_profit / max(gross_loss, 1)

        return result

    def _detailed_analysis(self, result) -> Dict[str, Any]:
        """Детальный анализ результатов"""
        analysis = {}

        # Sharpe Ratio
        try:
            sharpe = result.analyzers.sharpe.get_analysis()
            analysis['sharpe_ratio'] = sharpe.get('sharperatio', 0) or 0
        except:
            analysis['sharpe_ratio'] = 0

        # DrawDown
        try:
            drawdown = result.analyzers.drawdown.get_analysis()
            analysis['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0) or 0
            analysis['max_drawdown_period'] = drawdown.get('max', {}).get('len', 0) or 0
        except:
            analysis['max_drawdown'] = 0
            analysis['max_drawdown_period'] = 0

        # Returns
        try:
            returns = result.analyzers.returns.get_analysis()
            analysis['total_returns'] = (returns.get('rtot', 0) or 0) * 100
            analysis['average_returns'] = (returns.get('ravg', 0) or 0) * 100
        except:
            analysis['total_returns'] = 0
            analysis['average_returns'] = 0

        return analysis

    def _print_results(self, results: Dict[str, Any]):
        """Вывод результатов на консоль"""
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

        print("=" * 60)

    def _plot_results(self, cerebro, strategy_name: str):
        """Построение графиков"""
        try:
            print(f"\n📈 Построение графика для {strategy_name}...")
            cerebro.plot(figsize=(15, 8), style='candlestick', volume=False)
            plt.suptitle(f'Backtest Results: {strategy_name}', fontsize=16)
            plt.show()
        except Exception as e:
            print(f"⚠️ Ошибка построения графика: {e}")

    def compare_strategies(self,
                          strategy_names: List[str] = None,
                          custom_params: Dict[str, Dict[str, Any]] = None,
                          data_path: str = None,
                          timeframe: str = "1d",
                          skip_errors: bool = True) -> pd.DataFrame:
        """
        Сравнение стратегий

        Args:
            strategy_names: список имен стратегий для сравнения (если None - все стратегии)
            custom_params: словарь кастомных параметров для стратегий
            data_path: путь к данным
            timeframe: таймфрейм данных
            skip_errors: пропускать стратегии с ошибками
        """

        if strategy_names is None:
            strategy_names = list(self.strategies_registry.keys())

        if custom_params is None:
            custom_params = {}

        print(f"\n🔍 СРАВНЕНИЕ СТРАТЕГИЙ")
        print("=" * 80)
        print(f"📊 Стратегий к тестированию: {len(strategy_names)}")
        print(f"⏱️ Таймфрейм: {timeframe}")
        print()

        results = []
        failed_strategies = []

        for i, strategy_name in enumerate(strategy_names, 1):
            if strategy_name not in self.strategies_registry:
                print(f"❌ Стратегия '{strategy_name}' не найдена, пропускаю...")
                continue

            print(f"⏳ [{i}/{len(strategy_names)}] Тестирование: {strategy_name}")

            try:
                # Используем кастомные параметры если есть
                params = custom_params.get(strategy_name, {})

                result = self.run_backtest(
                    strategy_name=strategy_name,
                    strategy_params=params,
                    data_path=data_path,
                    timeframe=timeframe,
                    show_plot=False,
                    verbose=False
                )
                results.append(result)
                print(f"✅ Завершено: {result['total_return']:+.2f}% | {result.get('total_trades', 0)} сделок")

            except Exception as e:
                error_msg = str(e)
                if "array index out of range" in error_msg:
                    print(f"❌ Ошибка в {strategy_name}: Ошибка выполнения стратегии: {error_msg}")
                else:
                    print(f"❌ Ошибка в {strategy_name}: {error_msg}")

                failed_strategies.append(strategy_name)

                if not skip_errors:
                    raise e
                continue

        if failed_strategies:
            print(f"\n⚠️ Стратегии с ошибками ({len(failed_strategies)}):")
            for strategy in failed_strategies:
                print(f"   • {strategy}")

        if not results:
            print("❌ Нет успешных результатов для сравнения")
            return pd.DataFrame()

        # Создаем DataFrame для сравнения
        comparison_df = pd.DataFrame(results)

        # Сортируем по доходности
        comparison_df = comparison_df.sort_values('total_return', ascending=False)

        # Выбираем ключевые метрики для отображения
        key_metrics = [
            'strategy_name', 'total_return', 'profit_loss', 'total_trades',
            'win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown'
        ]

        available_metrics = [col for col in key_metrics if col in comparison_df.columns]
        display_df = comparison_df[available_metrics].copy()

        print(f"\n🏆 РЕЙТИНГ СТРАТЕГИЙ:")
        print("=" * 100)
        print(display_df.to_string(index=False, float_format='%.2f'))

        # Лучшая стратегия
        if len(results) > 0:
            best_strategy = comparison_df.iloc[0]
            print(f"\n🥇 ЛУЧШАЯ СТРАТЕГИЯ: {best_strategy['strategy_name']}")
            print(f"   📈 Доходность: {best_strategy['total_return']:+.2f}%")
            print(f"   💰 Прибыль: ${best_strategy['profit_loss']:+,.2f}")
            print(f"   🎯 Винрейт: {best_strategy.get('win_rate', 0):.1f}%")

        print("=" * 100)

        return display_df

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Получить подробную информацию о стратегии"""
        if strategy_name not in self.strategies_registry:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")

        return self.strategies_registry[strategy_name]


# Пример использования
if __name__ == "__main__":
    # Создаем бэктестер
    backtester = UniversalBacktester(initial_cash=100000, commission=0.001)

    # Показываем доступные стратегии
    backtester.list_strategies()

    # Можно тестировать любую стратегию
    # backtester.run_backtest("ProfitableBTCStrategy")

    # Или сравнить все стратегии
    backtester.compare_strategies()
