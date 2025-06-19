#!/usr/bin/env python3
"""
Интерактивный бэктестер для тестирования стратегий на всех доступных парах
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import time

# Добавляем путь к универсальному бэктестеру
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from universal_backtester import UniversalBacktester


class InteractiveMultiPairBacktester:
    """
    Интерактивный бэктестер для тестирования одной стратегии на всех доступных парах
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
        
        self.results_cache = {}
        
    def get_all_data_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Получение всех доступных пар данных (exchange, symbol, timeframe)
        
        Returns:
            List[Tuple[str, str, str]]: Список кортежей (exchange, symbol, timeframe)
        """
        all_pairs = []
        
        for exchange, symbols_data in self.backtester.data_manager.available_data.items():
            for symbol, timeframe_data in symbols_data.items():
                for tf_info in timeframe_data:
                    timeframe = tf_info['timeframe']
                    all_pairs.append((exchange, symbol, timeframe))
        
        return sorted(all_pairs)
    
    def display_strategy_menu(self) -> str:
        """
        Отображение меню выбора стратегии
        
        Returns:
            str: Выбранная стратегия
        """
        strategies = list(self.backtester.strategies_registry.keys())
        
        print("\n" + "="*80)
        print("🎯 ВЫБОР СТРАТЕГИИ ДЛЯ МУЛЬТИ-ТЕСТИРОВАНИЯ")
        print("="*80)
        
        # Группируем стратегии по файлам для удобства
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
                position_size = strategy_info['default_params'].get('position_size', 'НЕТ')
                print(f"   {strategy_index:2d}. {strategy_name} (position_size: {position_size})")
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
    
    def display_timeframe_menu(self, all_pairs: List[Tuple[str, str, str]]) -> str:
        """
        Отображение меню выбора таймфрейма
        
        Args:
            all_pairs: Список всех доступных пар
            
        Returns:
            str: Выбранный таймфрейм
        """
        # Получаем уникальные таймфреймы
        timeframes = sorted(list(set(pair[2] for pair in all_pairs)))
        
        print(f"\n📊 ДОСТУПНЫЕ ТАЙМФРЕЙМЫ:")
        print("-" * 40)
        for i, tf in enumerate(timeframes, 1):
            # Подсчитываем количество пар для каждого таймфрейма
            pair_count = len([p for p in all_pairs if p[2] == tf])
            print(f"   {i}. {tf} ({pair_count} пар)")
        
        print(f"   {len(timeframes) + 1}. Все таймфреймы")
        
        while True:
            try:
                choice = input(f"Выберите таймфрейм (1-{len(timeframes) + 1}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(timeframes):
                    selected_tf = timeframes[choice_num - 1]
                    print(f"✅ Выбран таймфрейм: {selected_tf}")
                    return selected_tf
                elif choice_num == len(timeframes) + 1:
                    print("✅ Выбраны все таймфреймы")
                    return "all"
                else:
                    print(f"❌ Введите число от 1 до {len(timeframes) + 1}")
            except ValueError:
                print("❌ Введите корректное число")
            except KeyboardInterrupt:
                print("\n👋 Выход из программы")
                sys.exit(0)
    
    def run_strategy_on_all_pairs(self, 
                                 strategy_name: str, 
                                 selected_timeframe: str = "all",
                                 custom_params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Запуск выбранной стратегии на всех доступных парах
        
        Args:
            strategy_name: Название стратегии
            selected_timeframe: Выбранный таймфрейм или "all"
            custom_params: Кастомные параметры стратегии
            
        Returns:
            pd.DataFrame: Результаты тестирования
        """
        all_pairs = self.get_all_data_pairs()
        
        # Фильтруем по таймфрейму если нужно
        if selected_timeframe != "all":
            all_pairs = [pair for pair in all_pairs if pair[2] == selected_timeframe]
        
        print(f"\n🚀 ЗАПУСК МУЛЬТИ-ТЕСТИРОВАНИЯ")
        print("="*80)
        print(f"📊 Стратегия: {strategy_name}")
        print(f"⏰ Таймфрейм: {selected_timeframe}")
        print(f"💎 Пар для тестирования: {len(all_pairs)}")
        print(f"💰 Начальный капитал: ${self.backtester.initial_cash:,.0f}")
        
        if custom_params:
            print(f"⚙️ Параметры: {custom_params}")
        
        print("="*80)
        
        results = []
        successful_tests = 0
        failed_tests = 0
        
        start_time = time.time()
        
        for i, (exchange, symbol, timeframe) in enumerate(all_pairs, 1):
            pair_name = f"{exchange}:{symbol}:{timeframe}"
            print(f"⏳ [{i:2d}/{len(all_pairs)}] Тестирование: {pair_name}")
            
            try:
                result = self.backtester.run_single_backtest(
                    strategy_name=strategy_name,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_params=custom_params,
                    show_plot=False,
                    verbose=False,
                    suppress_strategy_errors=True
                )
                
                # Добавляем информацию о паре к результату
                result['exchange'] = exchange
                result['symbol'] = symbol
                result['timeframe'] = timeframe
                result['pair_name'] = f"{exchange}:{symbol}"
                result['full_pair_name'] = pair_name
                
                results.append(result)
                successful_tests += 1
                
                # Краткий вывод результата
                return_pct = result.get('total_return', 0)
                trades_count = result.get('total_trades', 0)
                print(f"✅ {return_pct:+6.1f}% | {trades_count:3d} сделок")
                
            except Exception as e:
                failed_tests += 1
                print(f"❌ Ошибка: {str(e)[:50]}...")
                continue
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 ЗАВЕРШЕНО ЗА {elapsed_time:.1f} СЕК")
        print(f"✅ Успешно: {successful_tests} | ❌ Ошибок: {failed_tests}")
        
        if not results:
            print("❌ Нет успешных результатов для анализа!")
            return pd.DataFrame()
        
        # Создаем DataFrame с результатами
        df_results = pd.DataFrame(results)
        
        # Сортируем по доходности
        df_results = df_results.sort_values('total_return', ascending=False)
        
        return df_results
    
    def display_results_summary(self, results_df: pd.DataFrame, strategy_name: str):
        """
        Отображение сводки результатов тестирования
        
        Args:
            results_df: DataFrame с результатами
            strategy_name: Название стратегии
        """
        if results_df.empty:
            print("❌ Нет результатов для отображения")
            return
        
        print(f"\n🏆 РЕЗУЛЬТАТЫ МУЛЬТИ-ТЕСТИРОВАНИЯ: {strategy_name}")
        print("="*100)
        
        # Топ-10 лучших результатов
        print("🥇 ТОП-10 ЛУЧШИХ РЕЗУЛЬТАТОВ:")
        print("-"*100)
        
        top_results = results_df.head(10)
        
        header = f"{'Ранг':<4} {'Пара':<20} {'Таймфрейм':<10} {'Доходность':<12} {'Сделки':<8} {'Винрейт':<8} {'Шарп':<8}"
        print(header)
        print("-"*100)
        
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            pair_name = row['pair_name']
            timeframe = row['timeframe']
            total_return = row.get('total_return', 0)
            total_trades = row.get('total_trades', 0)
            win_rate = row.get('win_rate', 0)  # Убрано умножение на 100 - винрейт уже в процентах
            sharpe = row.get('sharpe_ratio', 0)
            
            print(f"{i:<4} {pair_name:<20} {timeframe:<10} {total_return:>+10.1f}% "
                  f"{total_trades:>6} {win_rate:>6.1f}% {sharpe:>6.2f}")
        
        print("="*100)
        
        # Статистика по биржам
        print("\n📊 СТАТИСТИКА ПО БИРЖАМ:")
        exchange_stats = results_df.groupby('exchange').agg({
            'total_return': ['mean', 'max', 'min', 'count'],
            'total_trades': 'mean'
        }).round(2)
        
        print(exchange_stats)
        
        # Статистика по символам
        print("\n💎 СТАТИСТИКА ПО СИМВОЛАМ:")
        symbol_stats = results_df.groupby('symbol').agg({
            'total_return': ['mean', 'max', 'count']
        }).round(2)
        
        # Показываем только топ-5 символов по средней доходности
        symbol_stats_sorted = symbol_stats.sort_values(('total_return', 'mean'), ascending=False)
        print(symbol_stats_sorted.head())
        
        # Лучший результат
        best_result = results_df.iloc[0]
        print(f"\n🏆 ЛУЧШИЙ РЕЗУЛЬТАТ:")
        print(f"   Пара: {best_result['full_pair_name']}")
        print(f"   Доходность: {best_result['total_return']:+.2f}%")
        print(f"   Прибыль: ${best_result.get('profit_loss', 0):+,.2f}")
        print(f"   Сделок: {best_result.get('total_trades', 0)}")
        print(f"   Винрейт: {best_result.get('win_rate', 0):.1f}%")
        
        # Худший результат
        worst_result = results_df.iloc[-1]
        print(f"\n💔 ХУДШИЙ РЕЗУЛЬТАТ:")
        print(f"   Пара: {worst_result['full_pair_name']}")
        print(f"   Доходность: {worst_result['total_return']:+.2f}%")
        
        print("="*100)
    
    def save_results_to_csv(self, results_df: pd.DataFrame, strategy_name: str, timeframe: str):
        """
        Сохранение результатов в CSV файл
        
        Args:
            results_df: DataFrame с результатами
            strategy_name: Название стратегии
            timeframe: Выбранный таймфрейм
        """
        if results_df.empty:
            return
        
        # Создаем директорию для результатов
        results_dir = os.path.join(os.path.dirname(__file__), "multi_pair_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Формируем имя файла
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_{timeframe}_{timestamp}.csv"
        filepath = os.path.join(results_dir, filename)
        
        # Выбираем ключевые колонки для сохранения
        save_columns = [
            'full_pair_name', 'exchange', 'symbol', 'timeframe',
            'total_return', 'profit_loss', 'total_trades', 'win_rate',
            'profit_factor', 'sharpe_ratio', 'max_drawdown'
        ]
        
        # Фильтруем доступные колонки
        available_columns = [col for col in save_columns if col in results_df.columns]
        
        results_df[available_columns].to_csv(filepath, index=False, encoding='utf-8')
        print(f"\n💾 Результаты сохранены в: {filepath}")
    
    def run_interactive_session(self):
        """
        Запуск интерактивной сессии мульти-тестирования
        """
        print("🚀 ИНТЕРАКТИВНЫЙ МУЛЬТИ-БЭКТЕСТЕР")
        print("="*80)
        print("Тестирование одной стратегии на всех доступных криптопарах")
        print("="*80)
        
        try:
            # Показываем доступные данные
            all_pairs = self.get_all_data_pairs()
            print(f"\n📊 Доступно {len(all_pairs)} пар данных:")
            
            exchanges = set(pair[0] for pair in all_pairs)
            symbols = set(pair[1] for pair in all_pairs)
            timeframes = set(pair[2] for pair in all_pairs)
            
            print(f"   Биржи: {', '.join(sorted(exchanges))}")
            print(f"   Символы: {', '.join(sorted(symbols))}")
            print(f"   Таймфреймы: {', '.join(sorted(timeframes))}")
            
            # Выбор стратегии
            selected_strategy = self.display_strategy_menu()
            
            # Выбор таймфрейма
            selected_timeframe = self.display_timeframe_menu(all_pairs)
            
            # Запрос кастомных параметров
            print(f"\n⚙️ Хотите изменить параметры стратегии? (y/n): ", end="")
            change_params = input().strip().lower()
            
            custom_params = {}
            if change_params in ['y', 'yes', 'д', 'да']:
                print("Введите параметры в формате: param1=value1,param2=value2")
                print("Например: position_size=0.8,rsi_period=21")
                params_input = input("Параметры: ").strip()
                
                if params_input:
                    try:
                        for param_pair in params_input.split(','):
                            key, value = param_pair.split('=')
                            key = key.strip()
                            value = value.strip()
                            
                            # Пытаемся определить тип значения
                            try:
                                if '.' in value:
                                    custom_params[key] = float(value)
                                else:
                                    custom_params[key] = int(value)
                            except ValueError:
                                custom_params[key] = value
                        
                        print(f"✅ Установлены параметры: {custom_params}")
                    except Exception as e:
                        print(f"❌ Ошибка в параметрах: {e}")
                        print("Используются параметры по умолчанию")
            
            # Запуск тестирования
            results_df = self.run_strategy_on_all_pairs(
                strategy_name=selected_strategy,
                selected_timeframe=selected_timeframe,
                custom_params=custom_params if custom_params else None
            )
            
            if not results_df.empty:
                # Показываем результаты
                self.display_results_summary(results_df, selected_strategy)
                
                # Предлагаем сохранить результаты
                print(f"\n💾 Сохранить результаты в CSV? (y/n): ", end="")
                save_csv = input().strip().lower()
                
                if save_csv in ['y', 'yes', 'д', 'да']:
                    self.save_results_to_csv(results_df, selected_strategy, selected_timeframe)
                
                # Предлагаем повторить с другой стратегией
                print(f"\n🔄 Протестировать другую стратегию? (y/n): ", end="")
                repeat = input().strip().lower()
                
                if repeat in ['y', 'yes', 'д', 'да']:
                    self.run_interactive_session()
            
        except KeyboardInterrupt:
            print("\n👋 Программа завершена пользователем")
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")


def main():
    """Основная функция запуска интерактивного бэктестера"""
    
    # Настройки по умолчанию
    backtester = InteractiveMultiPairBacktester(
        initial_cash=100000,    # $100,000
        commission=0.001,       # 0.1%
        spread=0.0005,         # 0.05%
        slippage=0.0002        # 0.02%
    )
    
    # Запуск интерактивной сессии
    backtester.run_interactive_session()


if __name__ == "__main__":
    main()