#!/usr/bin/env python3
"""
Система оптимизации стратегий для поиска лучших параметров
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import time
import itertools
from dataclasses import dataclass
import backtrader as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# Добавляем путь к универсальному бэктестеру
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from universal_backtester import UniversalBacktester


@dataclass
class OptimizationResult:
    """Результат одного прогона оптимизации"""
    params: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    final_value: float


class StrategyOptimizer:
    """
    Класс для оптимизации параметров торговых стратегий
    """
    
    def __init__(self, 
                 initial_cash: float = 100000,
                 commission: float = 0.001,
                 spread: float = 0.0005,
                 slippage: float = 0.0002):
        
        self.backtester = UniversalBacktester(
            initial_cash=initial_cash,
            commission=commission,
            spread=spread,
            slippage=slippage,
            require_position_size=True
        )
        
        self.optimization_results = []
    
    def display_strategy_menu(self) -> str:
        """Отображение меню выбора стратегии для оптимизации"""
        strategies = list(self.backtester.strategies_registry.keys())
        
        print("\n" + "="*80)
        print("🎯 ВЫБОР СТРАТЕГИИ ДЛЯ ОПТИМИЗАЦИИ")
        print("="*80)
        
        # Группируем стратегии по файлам
        strategies_by_file = defaultdict(list)
        for strategy_name in strategies:
            file_path = self.backtester.strategies_registry[strategy_name]['file']
            strategies_by_file[file_path].append(strategy_name)
        
        strategy_index = 1
        index_to_strategy = {}
        
        for file_path, file_strategies in strategies_by_file.items():
            print(f"\n📄 {os.path.basename(file_path)}:")
            for strategy_name in file_strategies:
                strategy_info = self.backtester.strategies_registry[strategy_name]
                params_count = len(strategy_info['default_params'])
                print(f"   {strategy_index:2d}. {strategy_name} ({params_count} параметров)")
                index_to_strategy[strategy_index] = strategy_name
                strategy_index += 1
        
        print("\n" + "="*80)
        
        while True:
            try:
                choice = input(f"Выберите стратегию (1-{len(strategies)}): ").strip()
                choice_num = int(choice)
                
                if choice_num in index_to_strategy:
                    selected_strategy = index_to_strategy[choice_num]
                    print(f"\n✅ Выбрана стратегия: {selected_strategy}")
                    return selected_strategy
                else:
                    print(f"❌ Введите число от 1 до {len(strategies)}")
            except ValueError:
                print("❌ Введите корректное число")
            except KeyboardInterrupt:
                print("\n👋 Выход из программы")
                sys.exit(0)
    
    def display_data_menu(self) -> Tuple[str, str, str]:
        """Отображение меню выбора данных"""
        available_data = self.backtester.data_manager.available_data
        
        print(f"\n📊 ВЫБОР ДАННЫХ ДЛЯ ОПТИМИЗАЦИИ")
        print("="*60)
        
        # Показываем доступные биржи
        exchanges = list(available_data.keys())
        print(f"Доступные биржи: {', '.join(exchanges)}")
        
        # Выбор биржи
        while True:
            exchange = input("Введите название биржи: ").strip()
            if exchange in exchanges:
                break
            print(f"❌ Биржа '{exchange}' не найдена. Доступные: {', '.join(exchanges)}")
        
        # Показываем доступные символы для выбранной биржи
        symbols = list(available_data[exchange].keys())
        print(f"Доступные символы: {', '.join(symbols)}")
        
        # Выбор символа
        while True:
            symbol = input("Введите символ: ").strip()
            if symbol in symbols:
                break
            print(f"❌ Символ '{symbol}' не найден. Доступные: {', '.join(symbols)}")
        
        # Показываем доступные таймфреймы
        timeframes = [tf_info['timeframe'] for tf_info in available_data[exchange][symbol]]
        print(f"Доступные таймфреймы: {', '.join(timeframes)}")
        
        # Выбор таймфрейма
        while True:
            timeframe = input("Введите таймфрейм: ").strip()
            if timeframe in timeframes:
                break
            print(f"❌ Таймфрейм '{timeframe}' не найден. Доступные: {', '.join(timeframes)}")
        
        print(f"\n✅ Выбраны данные: {exchange}:{symbol}:{timeframe}")
        return exchange, symbol, timeframe
    
    def get_strategy_param_ranges(self, strategy_name: str) -> Dict[str, List]:
        """
        Получение диапазонов параметров для оптимизации
        
        Args:
            strategy_name: Название стратегии
            
        Returns:
            Dict[str, List]: Словарь с диапазонами значений для каждого параметра
        """
        strategy_info = self.backtester.strategies_registry[strategy_name]
        default_params = strategy_info['default_params']
        
        print(f"\n⚙️ НАСТРОЙКА ПАРАМЕТРОВ ДЛЯ ОПТИМИЗАЦИИ: {strategy_name}")
        print("="*70)
        
        param_ranges = {}
        
        for param_name, default_value in default_params.items():
            # Пропускаем position_size - его не оптимизируем
            if param_name == 'position_size':
                continue
                
            print(f"\n📊 Параметр: {param_name} (текущее значение: {default_value})")
            
            if isinstance(default_value, (int, float)):
                self._setup_numeric_param_range(param_name, default_value, param_ranges)
            elif isinstance(default_value, bool):
                param_ranges[param_name] = [True, False]
                print(f"   Установлены значения: [True, False]")
            else:
                print(f"   ⚠️ Параметр '{param_name}' пропущен (неподдерживаемый тип: {type(default_value)})")
        
        return param_ranges
    
    def _setup_numeric_param_range(self, param_name: str, default_value: float, param_ranges: Dict):
        """Настройка диапазона для числового параметра"""
        print(f"   Введите диапазон для оптимизации:")
        print(f"   Формат: мин,макс,шаг (например: 10,30,5)")
        print(f"   Или нажмите Enter для автоматического диапазона")
        
        user_input = input(f"   {param_name}: ").strip()
        
        if not user_input:
            # Автоматический диапазон
            if isinstance(default_value, int):
                min_val = max(1, int(default_value * 0.5))
                max_val = int(default_value * 2)
                step = max(1, (max_val - min_val) // 10)
                param_ranges[param_name] = list(range(min_val, max_val + 1, step))
            else:
                min_val = round(default_value * 0.5, 3)
                max_val = round(default_value * 2, 3)
                step = round((max_val - min_val) / 10, 3)
                param_ranges[param_name] = [round(min_val + i * step, 3) 
                                          for i in range(11)]
            
            print(f"   ✅ Автоматический диапазон: {param_ranges[param_name][:3]}...{param_ranges[param_name][-3:]} ({len(param_ranges[param_name])} значений)")
        else:
            try:
                parts = user_input.split(',')
                if len(parts) == 3:
                    min_val, max_val, step = map(float, parts)
                    if isinstance(default_value, int):
                        min_val, max_val, step = int(min_val), int(max_val), int(step)
                        param_ranges[param_name] = list(range(min_val, max_val + 1, step))
                    else:
                        values = []
                        current = min_val
                        while current <= max_val:
                            values.append(round(current, 3))
                            current += step
                        param_ranges[param_name] = values
                    
                    print(f"   ✅ Установлен диапазон: {param_ranges[param_name][:3]}...{param_ranges[param_name][-3:]} ({len(param_ranges[param_name])} значений)")
                else:
                    raise ValueError("Неверный формат")
            except:
                print(f"   ❌ Неверный формат. Используется значение по умолчанию: [{default_value}]")
                param_ranges[param_name] = [default_value]
    
    def _run_single_optimization(self, strategy_name: str, exchange: str, symbol: str, 
                                timeframe: str, current_params: Dict[str, Any], 
                                combination_num: int, total_combinations: int) -> Tuple[OptimizationResult, bool]:
        """
        Запуск одного теста оптимизации (для многопоточности)
        
        Returns:
            Tuple[OptimizationResult, bool]: (результат, успешно ли выполнен)
        """
        try:
            # Запускаем бэктест с текущими параметрами
            result = self.backtester.run_single_backtest(
                strategy_name=strategy_name,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                strategy_params=current_params,
                show_plot=False,
                verbose=False,
                suppress_strategy_errors=True
            )
            
            # Создаем результат оптимизации
            opt_result = OptimizationResult(
                params=current_params.copy(),
                total_return=result.get('total_return', 0),
                sharpe_ratio=result.get('sharpe_ratio', 0),
                max_drawdown=result.get('max_drawdown', 0),
                total_trades=result.get('total_trades', 0),
                win_rate=result.get('win_rate', 0),
                profit_factor=result.get('profit_factor', 0),
                final_value=result.get('final_value', 0)
            )
            
            print(f"✅ [{combination_num:4d}/{total_combinations}] Доходность: {opt_result.total_return:+6.1f}% | Шарп: {opt_result.sharpe_ratio:.2f} | Сделок: {opt_result.total_trades}")
            
            return opt_result, True
            
        except Exception as e:
            print(f"❌ [{combination_num:4d}/{total_combinations}] Ошибка: {str(e)[:30]}...")
            return None, False

    def run_optimization(self, 
                        strategy_name: str,
                        exchange: str,
                        symbol: str,
                        timeframe: str,
                        param_ranges: Dict[str, List],
                        max_combinations: int = 1000,
                        num_threads: int = 15) -> List[OptimizationResult]:
        """
        Запуск оптимизации параметров с многопоточностью
        
        Args:
            strategy_name: Название стратегии
            exchange: Биржа
            symbol: Символ
            timeframe: Таймфрейм
            param_ranges: Диапазоны параметров
            max_combinations: Максимальное количество комбинаций
            num_threads: Количество потоков для параллельного выполнения
            
        Returns:
            List[OptimizationResult]: Результаты оптимизации
        """
        # Генерируем все комбинации параметров
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Ограничиваем количество комбинаций
        if len(all_combinations) > max_combinations:
            print(f"⚠️ Слишком много комбинаций ({len(all_combinations)}), ограничиваем до {max_combinations}")
            # Берем случайную выборку
            import random
            all_combinations = random.sample(all_combinations, max_combinations)
        
        print(f"\n🚀 ЗАПУСК МНОГОПОТОЧНОЙ ОПТИМИЗАЦИИ")
        print("="*80)
        print(f"📊 Стратегия: {strategy_name}")
        print(f"📈 Данные: {exchange}:{symbol}:{timeframe}")
        print(f"🔧 Параметров для оптимизации: {len(param_names)}")
        print(f"🎯 Комбинаций для тестирования: {len(all_combinations)}")
        print(f"🧵 Количество потоков: {num_threads}")
        print("="*80)
        
        results = []
        successful_tests = 0
        failed_tests = 0
        
        start_time = time.time()
        
        # Создаем задачи для многопоточности
        strategy_info = self.backtester.strategies_registry[strategy_name]
        
        # Готовим все комбинации параметров для многопоточности
        tasks = []
        for i, combination in enumerate(all_combinations, 1):
            # Создаем словарь параметров для текущей комбинации
            current_params = dict(zip(param_names, combination))
            
            # Добавляем position_size если его нет
            if 'position_size' in strategy_info['default_params']:
                current_params['position_size'] = strategy_info['default_params']['position_size']
            
            tasks.append((strategy_name, exchange, symbol, timeframe, current_params, i, len(all_combinations)))
        
        # Запускаем многопоточную оптимизацию
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Отправляем все задачи на выполнение
            future_to_task = {
                executor.submit(self._run_single_optimization, *task): task 
                for task in tasks
            }
            
            # Собираем результаты по мере завершения
            for future in as_completed(future_to_task):
                try:
                    opt_result, success = future.result()
                    if success and opt_result:
                        results.append(opt_result)
                        successful_tests += 1
                    else:
                        failed_tests += 1
                except Exception as e:
                    failed_tests += 1
                    print(f"❌ Ошибка выполнения задачи: {str(e)[:30]}...")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 МНОГОПОТОЧНАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ЗА {elapsed_time:.1f} СЕК")
        print(f"✅ Успешно: {successful_tests} | ❌ Ошибок: {failed_tests}")
        print(f"⚡ Ускорение: ~{num_threads}x (теоретически)")
        
        # Сортируем результаты по доходности
        results.sort(key=lambda x: x.total_return, reverse=True)
        
        self.optimization_results = results
        return results
    
    def display_optimization_results(self, results: List[OptimizationResult], top_n: int = 20):
        """
        Отображение результатов оптимизации
        
        Args:
            results: Результаты оптимизации
            top_n: Количество лучших результатов для отображения
        """
        if not results:
            print("❌ Нет результатов для отображения")
            return
        
        print(f"\n🏆 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ (ТОП-{min(top_n, len(results))})")
        print("="*120)
        
        # Заголовок таблицы
        header = f"{'Ранг':<4} {'Доходность':<12} {'Шарп':<8} {'Просадка':<10} {'Сделки':<8} {'Винрейт':<8} {'PF':<6} {'Параметры':<50}"
        print(header)
        print("-"*120)
        
        # Отображаем топ результатов
        for i, result in enumerate(results[:top_n], 1):
            # Форматируем параметры для отображения
            params_str = ", ".join([f"{k}={v}" for k, v in result.params.items() 
                                   if k != 'position_size'])
            if len(params_str) > 47:
                params_str = params_str[:44] + "..."
            
            row = (f"{i:<4} "
                   f"{result.total_return:>+10.1f}% "
                   f"{result.sharpe_ratio:>7.2f} "
                   f"{result.max_drawdown:>9.1f}% "
                   f"{result.total_trades:>7} "
                   f"{result.win_rate:>7.1f}% "
                   f"{result.profit_factor:>5.2f} "
                   f"{params_str:<50}")
            print(row)
        
        print("="*120)
        
        # Статистика
        if len(results) > 0:
            best = results[0]
            worst = results[-1]
            avg_return = sum(r.total_return for r in results) / len(results)
            
            print(f"\n📊 СТАТИСТИКА ОПТИМИЗАЦИИ:")
            print(f"   🥇 Лучший результат: {best.total_return:+.1f}% (Шарп: {best.sharpe_ratio:.2f})")
            print(f"   💔 Худший результат: {worst.total_return:+.1f}%")
            print(f"   📈 Средняя доходность: {avg_return:+.1f}%")
            print(f"   🔢 Всего протестировано: {len(results)} комбинаций")
            
            # Лучшие параметры
            print(f"\n🎯 ЛУЧШИЕ ПАРАМЕТРЫ:")
            for param, value in best.params.items():
                if param != 'position_size':
                    print(f"   • {param}: {value}")
    
    def save_optimization_results(self, results: List[OptimizationResult], 
                                 strategy_name: str, exchange: str, symbol: str, timeframe: str):
        """Сохранение результатов оптимизации в CSV"""
        if not results:
            return
        
        # Создаем DataFrame
        data = []
        for i, result in enumerate(results, 1):
            row = {
                'rank': i,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'final_value': result.final_value
            }
            
            # Добавляем параметры
            for param, value in result.params.items():
                row[f'param_{param}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Создаем директорию для результатов
        results_dir = os.path.join(os.path.dirname(__file__), "optimization_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Формируем имя файла
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"opt_{strategy_name}_{exchange}_{symbol}_{timeframe}_{timestamp}.csv"
        filepath = os.path.join(results_dir, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"\n💾 Результаты оптимизации сохранены в: {filepath}")
    
    def run_interactive_optimization(self):
        """Запуск интерактивной сессии оптимизации"""
        print("🚀 ИНТЕРАКТИВНАЯ ОПТИМИЗАЦИЯ СТРАТЕГИЙ")
        print("="*80)
        print("Поиск оптимальных параметров для максимизации прибыли")
        print("="*80)
        
        try:
            # Шаг 1: Выбор стратегии
            strategy_name = self.display_strategy_menu()
            
            # Шаг 2: Выбор данных
            exchange, symbol, timeframe = self.display_data_menu()
            
            # Шаг 3: Настройка параметров
            param_ranges = self.get_strategy_param_ranges(strategy_name)
            
            if not param_ranges:
                print("❌ Нет параметров для оптимизации!")
                return
            
            # Шаг 4: Настройка ограничений
            print(f"\n⚙️ ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ")
            print("-" * 40)
            
            total_combinations = 1
            for param_values in param_ranges.values():
                total_combinations *= len(param_values)
            
            print(f"Общее количество комбинаций: {total_combinations}")
            
            max_combinations = 1000
            if total_combinations > max_combinations:
                max_input = input(f"Ограничить до {max_combinations} комбинаций? (y/n, по умолчанию y): ").strip().lower()
                if max_input in ['n', 'no', 'н', 'нет']:
                    try:
                        max_combinations = int(input("Введите максимальное количество комбинаций: "))
                    except ValueError:
                        print("Используется значение по умолчанию: 1000")
            
            # Шаг 5: Запуск оптимизации
            results = self.run_optimization(
                strategy_name=strategy_name,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                param_ranges=param_ranges,
                max_combinations=max_combinations
            )
            
            # Шаг 6: Отображение результатов
            if results:
                self.display_optimization_results(results)
                
                # Предложение сохранить результаты
                save_choice = input(f"\n💾 Сохранить результаты в CSV? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes', 'д', 'да']:
                    self.save_optimization_results(results, strategy_name, exchange, symbol, timeframe)
                
                # Предложение запустить лучшую стратегию
                run_best = input(f"\n🚀 Запустить бэктест с лучшими параметрами? (y/n): ").strip().lower()
                if run_best in ['y', 'yes', 'д', 'да']:
                    best_params = results[0].params
                    print(f"\n🎯 Запуск бэктеста с оптимальными параметрами...")
                    
                    self.backtester.run_single_backtest(
                        strategy_name=strategy_name,
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_params=best_params,
                        show_plot=True,
                        verbose=True,
                        suppress_strategy_errors=False
                    )
            
        except KeyboardInterrupt:
            print("\n👋 Оптимизация прервана пользователем")
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")


def main():
    """Основная функция запуска оптимизатора"""
    
    # Настройки по умолчанию
    optimizer = StrategyOptimizer(
        initial_cash=100000,    # $100,000
        commission=0.001,       # 0.1%
        spread=0.0005,         # 0.05%
        slippage=0.0002        # 0.02%
    )
    
    # Запуск интерактивной сессии
    optimizer.run_interactive_optimization()


if __name__ == "__main__":
    main()