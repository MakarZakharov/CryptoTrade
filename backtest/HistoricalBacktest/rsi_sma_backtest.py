import backtrader as bt
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from tqdm import tqdm
import pickle
import json

from datetime import datetime

# Добавляем путь к стратегиям
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from strategies.TestStrategies.RSI_SMA_Strategy import RSI_SMA_Strategy


class RSI_SMA_BacktestRunner:
    """
    Специализированный бэктестер для RSI_SMA_Strategy
    """
    
    def __init__(self, initial_cash=10000, commission=0.001):
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = None
        
    def load_data_from_csv(self, csv_file):
        """
        Загрузка данных из CSV файла Binance
        """
        if not os.path.exists(csv_file):
            print(f"Ошибка: файл {csv_file} не найден")
            return None
            
        try:
            # Загружаем данные
            data = pd.read_csv(csv_file)
            print(f"Загружено {len(data)} строк из {csv_file}")
            print(f"Колонки: {list(data.columns)}")
            
            # Проверяем структуру данных
            if 'timestamp' in data.columns:
                # Формат Binance
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'datetime' in data.columns:
                # Альтернативный формат
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
            else:
                print("Ошибка: не найдена колонка с временем")
                return None
                
            # Переименовываем колонки для backtrader
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Убираем NaN значения
            data = data.dropna()
            data = data.sort_index()
            
            print(f"Период данных: с {data.index[0]} по {data.index[-1]}")
            print(f"Первые 3 строки:")
            print(data.head(3))
            
            # Создаем data feed для backtrader
            data_bt = bt.feeds.PandasData(
                dataname=data,
                datetime=None,
                open='Open',
                high='High',
                low='Low', 
                close='Close',
                volume='Volume' if 'Volume' in data.columns else None,
                openinterest=None
            )
            
            return data_bt
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None
    
    def setup_cerebro(self, **strategy_params):
        """
        Настройка Cerebro с RSI_SMA_Strategy
        """
        self.cerebro = bt.Cerebro()
        
        # Добавляем стратегию с параметрами
        self.cerebro.addstrategy(RSI_SMA_Strategy, **strategy_params)
        
        # Настройки брокера
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
        """
        if self.cerebro is None:
            raise ValueError("Сначала вызовите setup_cerebro()")
            
        self.cerebro.adddata(data_feed)
        
        print(f'\n{"="*50}')
        print("ЗАПУСК БЭКТЕСТА RSI_SMA_Strategy")
        print(f'{"="*50}')
        print(f'Начальный капитал: ${self.cerebro.broker.getvalue():,.2f}')
        
        # Запускаем бэктест
        results = self.cerebro.run()
        result = results[0]
        
        final_value = self.cerebro.broker.getvalue()
        print(f'Финальный капитал: ${final_value:,.2f}')
        
        return result
    
    def analyze_results(self, result):
        """
        Детальный анализ результатов
        """
        print(f'\n{"="*50}')
        print("АНАЛИЗ РЕЗУЛЬТАТОВ")
        print(f'{"="*50}')
        
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        print(f"💰 Финансовые результаты:")
        print(f"   Начальный капитал: ${self.initial_cash:,.2f}")
        print(f"   Финальный капитал: ${final_value:,.2f}")
        print(f"   Абсолютная прибыль: ${final_value - self.initial_cash:,.2f}")
        print(f"   Общая доходность: {total_return:.2f}%")
        
        # Анализ сделок
        trade_analyzer = result.analyzers.trades.get_analysis()
        
        if 'total' in trade_analyzer and trade_analyzer.total.total > 0:
            total_trades = trade_analyzer.total.total
            won_trades = trade_analyzer.won.total
            lost_trades = trade_analyzer.lost.total
            win_rate = (won_trades / total_trades) * 100
            
            print(f"\n📊 Анализ сделок:")
            print(f"   Всего сделок: {total_trades}")
            print(f"   Выигрышных: {won_trades}")
            print(f"   Проигрышных: {lost_trades}")
            print(f"   Процент выигрышных: {win_rate:.1f}%")
            
            if won_trades > 0:
                avg_win = trade_analyzer.won.pnl.average
                print(f"   Средняя прибыль: ${avg_win:.2f}")
                
            if lost_trades > 0:
                avg_loss = trade_analyzer.lost.pnl.average
                print(f"   Средний убыток: ${avg_loss:.2f}")
                
                if won_trades > 0:
                    profit_factor = abs(avg_win * won_trades / (avg_loss * lost_trades))
                    print(f"   Profit Factor: {profit_factor:.2f}")
        else:
            print("\n⚠️  Сделки не были совершены")
            
        # Коэффициент Шарпа
        sharpe = result.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', None)
        if sharpe_ratio is not None:
            print(f"\n📈 Коэффициент Шарпа: {sharpe_ratio:.3f}")
            if sharpe_ratio > 1:
                print("   ✅ Отличный результат (>1)")
            elif sharpe_ratio > 0.5:
                print("   ✅ Хороший результат (>0.5)")
            else:
                print("   ⚠️  Низкий результат (<0.5)")
                
        # Максимальная просадка
        drawdown = result.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.get('max', {}).get('drawdown', None)
        if max_drawdown is not None:
            print(f"\n📉 Максимальная просадка: {max_drawdown:.2f}%")
            if max_drawdown < 10:
                print("   ✅ Низкий риск (<10%)")
            elif max_drawdown < 20:
                print("   ⚠️  Средний риск (10-20%)")
            else:
                print("   🔴 Высокий риск (>20%)")
                
        # SQN (System Quality Number)
        sqn = result.analyzers.sqn.get_analysis()
        sqn_value = sqn.get('sqn', None)
        if sqn_value is not None:
            print(f"\n🎯 SQN (качество системы): {sqn_value:.2f}")
            if sqn_value > 3:
                print("   ✅ Превосходное качество (>3)")
            elif sqn_value > 2:
                print("   ✅ Хорошее качество (2-3)")
            elif sqn_value > 1:
                print("   ⚠️  Среднее качество (1-2)")
            else:
                print("   🔴 Низкое качество (<1)")
    
    def optimize_strategy(self, data_feed, param_ranges):
        """
        Оптимизация параметров стратегии
        """
        print(f'\n{"="*50}')
        print("ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
        print(f'{"="*50}')
        
        # Создаем новый Cerebro для оптимизации
        cerebro_opt = bt.Cerebro(optreturn=False)
        cerebro_opt.optstrategy(RSI_SMA_Strategy, **param_ranges)
        cerebro_opt.adddata(data_feed)
        cerebro_opt.broker.setcash(self.initial_cash)
        cerebro_opt.broker.setcommission(commission=self.commission)
        cerebro_opt.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        print("Запуск оптимизации... (это может занять время)")
        opt_results = cerebro_opt.run()
        
        print(f"Протестировано {len(opt_results)} комбинаций параметров")
        
        # Находим лучшие параметры
        best_sharpe = float('-inf')
        best_params = None
        
        for run in opt_results:
            for strategy in run:
                sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', None)
                if sharpe_ratio is not None and sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_params = strategy.params._getitems()
                    
        print(f"\n🏆 Лучшие параметры (Sharpe Ratio: {best_sharpe:.3f}):")
        for param, value in best_params:
            print(f"   {param}: {value}")
            
        return best_params
    
    def plot_results(self, figsize=(15, 10)):
        """
        Построение графиков
        """
        if self.cerebro is None:
            print("Нет данных для построения графика")
            return
            
        try:
            print("\n📊 Построение графиков...")
            self.cerebro.plot(figsize=figsize, style='candlestick', barup='green', bardown='red')
            plt.show()
        except Exception as e:
            print(f"Ошибка при построении графиков: {e}")


def run_daily_backtest():
    """
    Запуск бэктеста на дневных данных
    """
    print("🚀 БЭКТЕСТ RSI_SMA_Strategy НА ДНЕВНЫХ ДАННЫХ")
    print("="*60)
    
    # Путь к дневным данным
    data_path = os.path.join(
        os.path.dirname(__file__), 
        "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    )
    
    # Создаем бэктестер
    runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
    
    # Загружаем данные
    data = runner.load_data_from_csv(data_path)
    if data is None:
        print("❌ Не удалось загрузить данные")
        return
    
    # Параметры стратегии для частой торговли
    strategy_params = {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'rsi_exit_overbought': 75,
        'rsi_exit_oversold': 25,
        'sma_fast': 10,
        'sma_slow': 20,
        'position_size': 0.1,
        'stop_loss': 0.02,
        'take_profit': 0.03
    }
    
    print(f"\n📋 Параметры стратегии:")
    for param, value in strategy_params.items():
        print(f"   {param}: {value}")
    
    # Настраиваем и запускаем бэктест
    runner.setup_cerebro(**strategy_params)
    result = runner.run_backtest(data)
    
    # Анализируем результаты
    runner.analyze_results(result)
    
    # Строим графики
    runner.plot_results()
    
    return runner, result


def run_4h_backtest():
    """
    Запуск бэктеста на 4-часовых данных
    """
    print("\n🚀 БЭКТЕСТ RSI_SMA_Strategy НА 4-ЧАСОВЫХ ДАННЫХ")
    print("="*60)
    
    # Путь к 4-часовым данным
    data_path = os.path.join(
        os.path.dirname(__file__), 
        "../../data/binance/BTCUSDT/4h/2022_12_15-2025_01_01.csv"
    )
    
    # Создаем бэктестер
    runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
    
    # Загружаем данные
    data = runner.load_data_from_csv(data_path)
    if data is None:
        print("❌ Не удалось загрузить данные")
        return
    
    # Параметры для более частой торговли на 4h
    strategy_params = {
        'rsi_period': 10,  # Более короткий период для 4h
        'rsi_overbought': 65,  # Более агрессивные уровни
        'rsi_oversold': 35,
        'rsi_exit_overbought': 70,
        'rsi_exit_oversold': 30,
        'sma_fast': 8,    # Более быстрые скользящие
        'sma_slow': 16,
        'position_size': 0.15,  # Больший размер позиции
        'stop_loss': 0.025,     # Чуть больший стоп-лосс
        'take_profit': 0.04     # Больший тейк-профит
    }
    
    print(f"\n📋 Параметры стратегии для 4h:")
    for param, value in strategy_params.items():
        print(f"   {param}: {value}")
    
    # Настраиваем и запускаем бэктест
    runner.setup_cerebro(**strategy_params)
    result = runner.run_backtest(data)
    
    # Анализируем результаты
    runner.analyze_results(result)
    
    # Строим графики
    runner.plot_results()
    
    return runner, result


def run_optimization():
    """
    Запуск оптимизации параметров
    """
    print("\n🔧 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ RSI_SMA_Strategy")
    print("="*60)
    
    # Путь к данным
    data_path = os.path.join(
        os.path.dirname(__file__), 
        "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    )
    
    runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
    data = runner.load_data_from_csv(data_path)
    
    if data is None:
        print("❌ Не удалось загрузить данные")
        return
    
    # Диапазоны параметров для оптимизации
    param_ranges = {
        'rsi_period': range(10, 21, 2),        # 10, 12, 14, 16, 18, 20
        'rsi_oversold': range(25, 36, 5),      # 25, 30, 35
        'rsi_overbought': range(65, 76, 5),    # 65, 70, 75
        'sma_fast': range(8, 13, 2),           # 8, 10, 12
        'sma_slow': range(18, 23, 2),          # 18, 20, 22
        'position_size': [0.05, 0.1, 0.15],   # Разные размеры позиций
        'stop_loss': [0.015, 0.02, 0.025],    # Разные стоп-лоссы
        'take_profit': [0.025, 0.03, 0.035]   # Разные тейк-профиты
    }
    
    print("📊 Диапазоны оптимизации:")
    for param, values in param_ranges.items():
        if isinstance(values, range):
            print(f"   {param}: {list(values)}")
        else:
            print(f"   {param}: {values}")
    
    # Запускаем оптимизацию
    best_params = runner.optimize_strategy(data, param_ranges)
    
    if best_params:
        print("\n🎯 Тестируем лучшие параметры:")
        
        # Создаем новый бэктестер с лучшими параметрами
        best_runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
        best_data = best_runner.load_data_from_csv(data_path)
        
        # Конвертируем параметры в словарь
        best_params_dict = dict(best_params)
        
        best_runner.setup_cerebro(**best_params_dict)
        best_result = best_runner.run_backtest(best_data)
        best_runner.analyze_results(best_result)
        best_runner.plot_results()


def run_optimization_with_results():
    """
    Enhanced optimization with detailed results
    """
    print("\n🔧 ENHANCED OPTIMIZATION - RSI_SMA_Strategy")
    print("="*60)

    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    )

    runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
    data = runner.load_data_from_csv(data_path)

    if data is None:
        print("❌ Не удалось загрузить данные")
        return

    # Focused parameter ranges for optimization
    param_ranges = {
        'rsi_period': [10, 12, 14, 16],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'sma_fast': [8, 10, 12],
        'sma_slow': [18, 20, 22],
        'position_size': [0.05, 0.1, 0.15],
        'stop_loss': [0.015, 0.02, 0.025],
        'take_profit': [0.025, 0.03, 0.035],
        'log_enabled': [False]  # Disable logging for optimization
    }

    print("📊 Запуск оптимизации параметров...")

    # Create optimization cerebro
    cerebro_opt = bt.Cerebro(optreturn=False)
    cerebro_opt.optstrategy(RSI_SMA_Strategy, **param_ranges)
    cerebro_opt.adddata(data)
    cerebro_opt.broker.setcash(10000)
    cerebro_opt.broker.setcommission(commission=0.001)
    cerebro_opt.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro_opt.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro_opt.addanalyzer(bt.analyzers.Returns, _name='returns')

    opt_results = cerebro_opt.run()

    print(f"✅ Оптимизация завершена! Протестировано {len(opt_results)} комбинаций")

    # Analyze results
    results_analysis = []

    for i, run in enumerate(opt_results):
        for strategy in run:
            # Get analyzers
            trades = strategy.analyzers.trades.get_analysis()
            sharpe = strategy.analyzers.sharpe.get_analysis()
            returns = strategy.analyzers.returns.get_analysis()

            # Get final value
            final_value = strategy.broker.getvalue()
            total_return = (final_value - 10000) / 10000 * 100

            # Get parameters
            params = dict(strategy.params._getitems())

            # Calculate trade statistics
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

            results_analysis.append({
                'params': params,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe.get('sharperatio', 0),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'score': total_return * 0.7 + (sharpe.get('sharperatio', 0) or 0) * 30  # Combined score
            })

    # Sort by combined score
    results_analysis.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n🏆 ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ:")
    print("="*80)

    for i, result in enumerate(results_analysis[:10], 1):
        print(f"\n#{i} | Доходность: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.3f}")
        print(f"    Сделок: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}%")
        print(f"    Параметры:")
        for param, value in result['params'].items():
            if param != 'log_enabled':
                print(f"      {param}: {value}")

    # Test best parameters
    best_params = results_analysis[0]['params']
    print(f"\n🎯 ТЕСТИРОВАНИЕ ЛУЧШИХ ПАРАМЕТРОВ:")
    print("="*50)

    best_runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
    best_data = best_runner.load_data_from_csv(data_path)
    best_params['log_enabled'] = True  # Enable logging for final test

    best_runner.setup_cerebro(**best_params)
    best_result = best_runner.run_backtest(best_data)
    best_runner.analyze_results(best_result)

    return results_analysis[0]


def run_single_optimization(params_combination, data_path, initial_cash=10000, commission=0.001):
    """
    Запуск одной комбинации параметров (для мультипроцессинга)
    """
    try:
        # Создаем отдельный бэктестер для каждого процесса
        runner = RSI_SMA_BacktestRunner(initial_cash=initial_cash, commission=commission)
        data = runner.load_data_from_csv(data_path)

        if data is None:
            return None

        # Настраиваем cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(RSI_SMA_Strategy, **params_combination)
        cerebro.adddata(data)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        # Запускаем бэктест
        results = cerebro.run()
        result = results[0]

        # Извлекаем метрики
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100

        trades = result.analyzers.trades.get_analysis()
        sharpe = result.analyzers.sharpe.get_analysis()
        drawdown = result.analyzers.drawdown.get_analysis()

        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
        sharpe_ratio = sharpe.get('sharperatio', 0) or 0

        # Комбинированный скор с учетом просадки
        score = (
            total_return * 0.5 +           # 50% - доходность
            sharpe_ratio * 20 +            # 20% - Sharpe ratio
            win_rate * 0.2 +               # 20% - процент выигрышных сделок
            max(-max_drawdown, -50) * 0.1  # 10% - штраф за просадку
        )

        return {
            'params': params_combination,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'score': score
        }

    except Exception as e:
        print(f"Ошибка в оптимизации: {e}")
        return None


def show_best_result_summary(best_result, title="ЛУЧШИЙ РЕЗУЛЬТАТ ОПТИМИЗАЦИИ"):
    """
    Отображение краткой сводки лучшего результата с параметрами
    """
    print(f"\n{'='*80}")
    print(f"🏆 {title}")
    print(f"{'='*80}")

    print(f"💰 ФИНАНСОВЫЕ ПОКАЗАТЕЛИ:")
    print(f"   📈 Доходность: {best_result['total_return']:.2f}%")
    print(f"   💵 Финальная стоимость: ${best_result['final_value']:,.2f}")
    print(f"   📊 Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
    print(f"   📉 Макс. просадка: {best_result['max_drawdown']:.2f}%")

    print(f"\n📋 ТОРГОВЫЕ ПОКАЗАТЕЛИ:")
    print(f"   🔄 Всего сделок: {best_result['total_trades']}")
    print(f"   ✅ Процент выигрышных: {best_result['win_rate']:.1f}%")
    print(f"   🎯 Общий скор: {best_result['score']:.1f}")

    print(f"\n⚙️  ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
    print(f"   {'='*50}")
    for param, value in best_result['params'].items():
        if param != 'log_enabled':
            if isinstance(value, float):
                print(f"   📌 {param}: {value:.3f}")
            else:
                print(f"   📌 {param}: {value}")
    print(f"   {'='*50}")


def run_fast_optimization():
    """
    БЫСТРАЯ оптимизация с мультипроцессингом и умными диапазонами
    """
    print("\n🚀 FAST OPTIMIZATION - RSI_SMA_Strategy")
    print("="*60)

    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    )

    # СОКРАЩЕННЫЕ диапазоны для быстрой оптимизации
    param_ranges = {
        'rsi_period': [10, 14, 18],              # 3 значения вместо 6
        'rsi_oversold': [25, 30, 35],            # 3 значения
        'rsi_overbought': [65, 70, 75],          # 3 значения
        'sma_fast': [8, 10, 12],                 # 3 значения
        'sma_slow': [18, 20, 22],                # 3 значения
        'position_size': [0.08, 0.12],          # 2 значения (лучшие из предыдущих тестов)
        'stop_loss': [0.02, 0.025],             # 2 значения
        'take_profit': [0.03, 0.035],           # 2 значения
        'log_enabled': [False]                   # Отключаем логи
    }

    # Генерируем все комбинации
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(itertools.product(*param_values))

    total_combinations = len(combinations)
    print(f"📊 Всего комбинаций для тестирования: {total_combinations}")
    print(f"⚡ Используем {mp.cpu_count()} процессов")

    # Создаем список параметров для каждой комбинации
    param_combinations = []
    for combination in combinations:
        params = dict(zip(param_names, combination))
        param_combinations.append(params)

    # Запускаем оптимизацию с мультипроцессингом
    results = []
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Отправляем задачи
        future_to_params = {
            executor.submit(run_single_optimization, params, data_path): params
            for params in param_combinations
        }

        # Собираем результаты с прогресс-баром
        with tqdm(total=total_combinations, desc="Оптимизация") as pbar:
            for future in as_completed(future_to_params):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)

    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()

    print(f"\n✅ Оптимизация завершена за {optimization_time:.1f} секунд!")
    print(f"📈 Успешно протестировано {len(results)} комбинаций")

    if not results:
        print("❌ Нет успешных результатов")
        return None

    # Сортируем по скору
    results.sort(key=lambda x: x['score'], reverse=True)

    # Сохраняем результаты
    save_optimization_results(results, "fast_optimization_results.json")

    # Показываем ЛУЧШИЙ результат отдельно
    best_result = results[0]
    show_best_result_summary(best_result, "🏆 ЛУЧШИЙ РЕЗУЛЬТАТ БЫСТРОЙ ОПТИМИЗАЦИИ")

    # Показываем топ результаты
    print(f"\n📊 ТОП-5 ЛУЧШИХ КОМБИНАЦИЙ:")
    print("="*80)

    for i, result in enumerate(results[:5], 1):
        print(f"\n#{i} | Доходность: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.3f} | Score: {result['score']:.1f}")
        print(f"    Сделок: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}% | Max DD: {result['max_drawdown']:.2f}%")
        if i == 1:
            print(f"    🏆 ПОБЕДИТЕЛЬ - Параметры:")
        else:
            print(f"    Параметры:")
        for param, value in result['params'].items():
            if param != 'log_enabled':
                print(f"      {param}: {value}")

    # Тестируем лучшую комбинацию с логами
    test_best_combination(results[0], data_path)

    return results[0]


def run_smart_optimization():
    """
    УМНАЯ оптимизация с поэтапным подходом
    """
    print("\n🧠 SMART OPTIMIZATION - RSI_SMA_Strategy")
    print("="*60)

    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    )

    # ЭТАП 1: Грубая оптимизация основных параметров
    print("\n🔍 ЭТАП 1: Грубая оптимизация основных параметров")
    stage1_params = {
        'rsi_period': [10, 14, 18],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'sma_fast': [8, 10, 12],
        'sma_slow': [18, 20, 22],
        'position_size': [0.1],              # Фиксированный
        'stop_loss': [0.02],                 # Фиксированный
        'take_profit': [0.03],               # Фиксированный
        'log_enabled': [False]
    }

    stage1_results = run_optimization_stage(stage1_params, data_path, "Этап 1")

    if not stage1_results:
        print("❌ Этап 1 неудачен")
        return None

    # Берем лучшие параметры из этапа 1
    best_stage1 = stage1_results[0]['params']

    # Показываем лучший результат этапа 1
    show_best_result_summary(stage1_results[0], "🥇 ЛУЧШИЙ РЕЗУЛЬТАТ ЭТАПА 1")

    # ЭТАП 2: Точная настройка управления рисками
    print(f"\n🎯 ЭТАП 2: Точная настройка рисков")
    print(f"Базовые параметры: RSI={best_stage1['rsi_period']}, SMA={best_stage1['sma_fast']}/{best_stage1['sma_slow']}")
    stage2_params = {
        'rsi_period': [best_stage1['rsi_period']],
        'rsi_oversold': [best_stage1['rsi_oversold']],
        'rsi_overbought': [best_stage1['rsi_overbought']],
        'sma_fast': [best_stage1['sma_fast']],
        'sma_slow': [best_stage1['sma_slow']],
        'position_size': [0.05, 0.08, 0.1, 0.12, 0.15],
        'stop_loss': [0.015, 0.02, 0.025, 0.03],
        'take_profit': [0.025, 0.03, 0.035, 0.04],
        'log_enabled': [False]
    }

    stage2_results = run_optimization_stage(stage2_params, data_path, "Этап 2")

    if not stage2_results:
        print("❌ Используем результаты этапа 1")
        final_results = stage1_results
    else:
        final_results = stage2_results

    # Сохраняем финальные результаты
    save_optimization_results(final_results, "smart_optimization_results.json")

    # Показываем ФИНАЛЬНЫЙ лучший результат
    best_final = final_results[0]
    show_best_result_summary(best_final, "🎯 ФИНАЛЬНЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ УМНОЙ ОПТИМИЗАЦИИ")

    # Показываем финальные результаты
    print(f"\n📊 ТОП-3 ФИНАЛЬНЫХ РЕЗУЛЬТАТА:")
    print("="*80)

    for i, result in enumerate(final_results[:3], 1):
        print(f"\n#{i} | Доходность: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.3f}")
        print(f"    Сделок: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}% | Max DD: {result['max_drawdown']:.2f}%")
        if i == 1:
            print(f"    🏆 ФИНАЛЬНЫЙ ПОБЕДИТЕЛЬ - Параметры:")
        else:
            print(f"    Параметры:")
        for param, value in result['params'].items():
            if param != 'log_enabled':
                print(f"      {param}: {value}")

    # Тестируем лучшую комбинацию
    test_best_combination(final_results[0], data_path)

    return final_results[0]


def run_optimization_stage(param_ranges, data_path, stage_name):
    """
    Запуск одного этапа оптимизации
    """
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(itertools.product(*param_values))

    param_combinations = []
    for combination in combinations:
        params = dict(zip(param_names, combination))
        param_combinations.append(params)

    print(f"📊 {stage_name}: {len(combinations)} комбинаций")

    results = []
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        future_to_params = {
            executor.submit(run_single_optimization, params, data_path): params
            for params in param_combinations
        }

        with tqdm(total=len(combinations), desc=stage_name) as pbar:
            for future in as_completed(future_to_params):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)

    end_time = datetime.now()
    print(f"⏱️  {stage_name} завершен за {(end_time - start_time).total_seconds():.1f} сек")

    if results:
        results.sort(key=lambda x: x['score'], reverse=True)
        print(f"🥇 Лучший результат: {results[0]['total_return']:.2f}% доходность")

    return results


def test_best_combination(best_result, data_path):
    """
    Тест лучшей комбинации с включенными логами
    """
    print(f"\n🎯 ДЕТАЛЬНОЕ ТЕСТИРОВАНИЕ ЛУЧШЕЙ КОМБИНАЦИИ:")
    print("="*60)

    # Показываем еще раз параметры перед тестом
    print("🔧 ИСПОЛЬЗУЕМЫЕ ПАРАМЕТРЫ:")
    for param, value in best_result['params'].items():
        if param != 'log_enabled':
            if isinstance(value, float):
                print(f"   {param}: {value:.3f}")
            else:
                print(f"   {param}: {value}")
    print("-" * 60)

    best_params = best_result['params'].copy()
    best_params['log_enabled'] = True

    runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
    data = runner.load_data_from_csv(data_path)

    if data is not None:
        runner.setup_cerebro(**best_params)
        result = runner.run_backtest(data)
        runner.analyze_results(result)


def save_optimization_results(results, filename):
    """
    Сохранение результатов оптимизации
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Результаты сохранены в {filename}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")


def load_optimization_results(filename):
    """
    Загрузка сохраненных результатов оптимизации
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"📂 Результаты загружены из {filename}")

        if results:
            # Показываем лучший результат из загруженных
            best_result = results[0]
            show_best_result_summary(best_result, f"🏆 ЛУЧШИЙ РЕЗУЛЬТАТ ИЗ {filename.upper()}")

        return results
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None


if __name__ == "__main__":
    print("🎯 МЕНЮ БЭКТЕСТИНГА RSI_SMA_Strategy")
    print("="*50)
    print("1. Дневной бэктест (1d)")
    print("2. 4-часовой бэктест (4h)")
    print("3. Оптимизация параметров (старая)")
    print("4. Расширенная оптимизация (старая)")
    print("5. ⚡ БЫСТРАЯ оптимизация (новая)")
    print("6. 🧠 УМНАЯ оптимизация (новая)")
    print("7. 📂 Загрузить сохраненные результаты")
    print("8. Все тесты подряд")

    choice = input("\nВыберите опцию (1-8): ").strip()

    if choice == "1":
        run_daily_backtest()
    elif choice == "2":
        run_4h_backtest()
    elif choice == "3":
        run_optimization()
    elif choice == "4":
        run_optimization_with_results()
    elif choice == "5":
        run_fast_optimization()  # Новая быстрая оптимизация
    elif choice == "6":
        run_smart_optimization()  # Новая умная оптимизация
    elif choice == "7":
        filename = input("Введите имя файла (например, fast_optimization_results.json): ")
        results = load_optimization_results(filename)
        if results:
            print(f"\n🏆 ЗАГРУЖЕННЫЕ РЕЗУЛЬТАТЫ:")
            for i, result in enumerate(results[:5], 1):
                print(f"\n#{i} | Доходность: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.3f}")
                print(f"    Параметры: {result['params']}")
    elif choice == "8":
        print("\n🚀 ЗАПУСК ВСЕХ ТЕСТОВ")
        run_daily_backtest()
        run_4h_backtest()
        run_fast_optimization()
    else:
        print("❌ Неверный выбор")