import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import List
from universal_backtester import UniversalBacktester

warnings.filterwarnings('ignore')


class InteractiveBacktester(UniversalBacktester):
    """
    Интерактивный бэктестер с расширенной визуализацией и выбором стратегий
    """
    
    def __init__(self, initial_cash: float = 100000, commission: float = 0.001):
        super().__init__(initial_cash, commission)
        self.portfolio_values = []
        self.dates = []
        self.trades_history = []

    def get_available_timeframes(self) -> List[str]:
        """Получить список доступных таймфреймов на основе существующих файлов"""
        data_base_path = os.path.join(os.path.dirname(__file__), '../../data/binance/BTCUSDT/')
        available_timeframes = []

        if os.path.exists(data_base_path):
            for item in os.listdir(data_base_path):
                timeframe_path = os.path.join(data_base_path, item)
                if os.path.isdir(timeframe_path):
                    # Проверяем, есть ли любые CSV файлы в папке
                    csv_files = [f for f in os.listdir(timeframe_path) if f.endswith('.csv')]
                    if csv_files:
                        available_timeframes.append(item)

        return sorted(available_timeframes) if available_timeframes else ["1d"]

    def select_timeframe_interactive(self) -> str:
        """Интерактивный выбор таймфрейма"""
        available_timeframes = self.get_available_timeframes()

        print("📊 ВЫБОР ТАЙМФРЕЙМА")
        print("-" * 30)
        print("Доступные таймфреймы:")

        for i, tf in enumerate(available_timeframes, 1):
            print(f"{i}. {tf}")

        while True:
            try:
                choice = input(f"Выберите таймфрейм (1-{len(available_timeframes)}) или Enter для {available_timeframes[0]}: ").strip()

                if not choice:
                    return available_timeframes[0]

                choice_num = int(choice)
                if 1 <= choice_num <= len(available_timeframes):
                    selected_tf = available_timeframes[choice_num - 1]
                    print(f"✅ Выбран таймфрейм: {selected_tf}")
                    return selected_tf
                else:
                    print(f"❌ Введите число от 1 до {len(available_timeframes)}")

            except ValueError:
                print("❌ Введите корректный номер")

    def select_strategy_interactive(self):
        """Интерактивный выбор стратегии"""
        print("\n🎯 ВЫБОР СТРАТЕГИИ ДЛЯ ТЕСТИРОВАНИЯ")
        print("=" * 60)
        
        if not self.strategies_registry:
            print("❌ Стратегии не найдены!")
            return None
            
        strategies_list = list(self.strategies_registry.keys())
        
        for i, strategy_name in enumerate(strategies_list, 1):
            strategy_info = self.strategies_registry[strategy_name]
            print(f"{i:2d}. 🎯 {strategy_name}")
            print(f"     📝 {strategy_info['description'][:60]}...")
            print(f"     📄 Файл: {strategy_info['file']}")
            print(f"     ⚙️  Параметров: {len(strategy_info['default_params'])}")
            print()
        
        while True:
            try:
                choice = input(f"Выберите стратегию (1-{len(strategies_list)}) или 'q' для выхода: ").strip()
                
                if choice.lower() == 'q':
                    return None
                    
                choice_num = int(choice)
                if 1 <= choice_num <= len(strategies_list):
                    selected_strategy = strategies_list[choice_num - 1]
                    print(f"\n✅ Выбрана стратегия: {selected_strategy}")
                    return selected_strategy
                else:
                    print(f"❌ Введите число от 1 до {len(strategies_list)}")
                    
            except ValueError:
                print("❌ Введите корректный номер стратегии")
                
    def run_enhanced_backtest(self, strategy_name: str, strategy_params: dict = None,
                            data_path: str = None, timeframe: str = "1d") -> dict:
        """Расширенный бэктест с сбором данных для графиков"""
        
        print(f"\n🚀 ЗАПУСК РАСШИРЕННОГО БЭКТЕСТА: {strategy_name}")
        print("=" * 70)
        
        # Очищаем предыдущие данные
        self.portfolio_values = []
        self.dates = []
        self.trades_history = []
        
        # Автоматически находим подходящий файл данных для таймфрейма
        if data_path is None:
            data_path = self._find_data_file(timeframe)

        # Запускаем обычный бэктест
        result = self.run_backtest(
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            data_path=data_path,
            timeframe=timeframe,
            show_plot=False,
            verbose=True
        )
        
        # Получаем данные для графиков
        self._collect_backtest_data(strategy_name, strategy_params, data_path, timeframe)
        
        return result
        
    def _collect_backtest_data(self, strategy_name: str, strategy_params: dict,
                             data_path: str, timeframe: str):
        """Сбор данных для построения графиков"""
        import backtrader as bt
        
        # Создаем новый cerebro для сбора данных
        cerebro = bt.Cerebro()
        
        # Создаем стратегию с трекингом
        strategy_info = self.strategies_registry[strategy_name]
        strategy_class = strategy_info['class']
        final_params = strategy_info['default_params'].copy()
        if strategy_params:
            final_params.update(strategy_params)
            
        # Создаем обертку стратегии для сбора данных
        class TrackingStrategy(strategy_class):
            def __init__(self):
                super().__init__()
                self.parent_backtester = None
                
            def next(self):
                super().next()
                # Сохраняем данные портфеля
                if self.parent_backtester:
                    self.parent_backtester.portfolio_values.append(self.broker.getvalue())
                    self.parent_backtester.dates.append(self.data.datetime.date(0))
                    
            def notify_trade(self, trade):
                super().notify_trade(trade) if hasattr(super(), 'notify_trade') else None
                if trade.isclosed and self.parent_backtester:
                    self.parent_backtester.trades_history.append({
                        'date': self.data.datetime.date(0),
                        'pnl': trade.pnl,
                        'size': trade.size,
                        'price': trade.price,
                        'commission': trade.commission
                    })
        
        cerebro.addstrategy(TrackingStrategy, **final_params)
        
        # Настраиваем стратегию
        data_feed = self.load_data(data_path, timeframe)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Запускаем и передаем ссылку на себя
        results = cerebro.run()
        if results:
            results[0].parent_backtester = self
            
    def _find_data_file(self, timeframe: str) -> str:
        """Найти подходящий файл данных для таймфрейма"""
        timeframe_path = os.path.join(os.path.dirname(__file__), f'../../data/binance/BTCUSDT/{timeframe}/')

        if os.path.exists(timeframe_path):
            csv_files = [f for f in os.listdir(timeframe_path) if f.endswith('.csv')]
            if csv_files:
                # Берем первый найденный CSV файл
                return f"../../data/binance/BTCUSDT/{timeframe}/{csv_files[0]}"

        # Если не найден, используем дефолтный путь
        return f"../../data/binance/BTCUSDT/{timeframe}/2018_01_01-2025_01_01.csv"

    def plot_enhanced_results(self, strategy_name: str, data_path: str = None, timeframe: str = "1d"):
        """Построение расширенных графиков результатов"""
        
        # Автоматически находим подходящий файл данных для таймфрейма
        if data_path is None:
            data_path = self._find_data_file(timeframe)

        # Загружаем исходные данные
        data_df = self._load_price_data(data_path, timeframe)
        
        # Создаем фигуру с подграфиками
        fig = plt.figure(figsize=(16, 12))
        
        # График 1: Цена актива
        ax1 = plt.subplot(3, 1, 1)
        self._plot_price_chart(ax1, data_df, strategy_name)
        
        # График 2: Кривая баланса
        ax2 = plt.subplot(3, 1, 2)
        self._plot_portfolio_curve(ax2)
        
        # График 3: Распределение доходности сделок
        ax3 = plt.subplot(3, 1, 3)
        self._plot_trades_distribution(ax3)
        
        plt.tight_layout()
        plt.show()
        
    def _load_price_data(self, data_path: str, timeframe: str) -> pd.DataFrame:
        """Загрузка данных цен для графика"""
        if data_path is None:
            data_path = f"../../data/binance/BTCUSDT/{timeframe}/2018_01_01-2025_01_01.csv"
            
        full_path = os.path.join(os.path.dirname(__file__), data_path) if not os.path.isabs(data_path) else data_path
        df = pd.read_csv(full_path)
        
        # Автоматическое определение колонок
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
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        return df
        
    def _plot_price_chart(self, ax, data_df, strategy_name):
        """График цены актива"""
        ax.plot(data_df.index, data_df['close'], linewidth=1, color='blue', alpha=0.7)
        ax.set_title(f'📈 Цена актива - {strategy_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Цена', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Статистика цены
        price_change = ((data_df['close'].iloc[-1] - data_df['close'].iloc[0]) / data_df['close'].iloc[0]) * 100
        ax.text(0.02, 0.98, f'Изменение цены: {price_change:+.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
    def _plot_portfolio_curve(self, ax):
        """График кривой баланса портфеля"""
        if not self.portfolio_values or not self.dates:
            ax.text(0.5, 0.5, 'Нет данных портфеля', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
            
        dates = pd.to_datetime(self.dates)
        values = np.array(self.portfolio_values)
        
        # Основная кривая
        ax.plot(dates, values, linewidth=2, color='green', label='Портфель')
        
        # Базовая линия (начальный капитал)
        ax.axhline(y=self.initial_cash, color='red', linestyle='--', alpha=0.7, label='Начальный капитал')
        
        # Заливка области прибыли/убытка
        ax.fill_between(dates, values, self.initial_cash, 
                       where=(values >= self.initial_cash), alpha=0.3, color='green', label='Прибыль')
        ax.fill_between(dates, values, self.initial_cash,
                       where=(values < self.initial_cash), alpha=0.3, color='red', label='Убыток')
        
        ax.set_title('💰 Кривая баланса портфеля', fontsize=14, fontweight='bold')
        ax.set_ylabel('Стоимость портфеля', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Статистика
        total_return = ((values[-1] - self.initial_cash) / self.initial_cash) * 100
        max_value = np.max(values)
        min_value = np.min(values)
        max_dd = ((max_value - min_value) / max_value) * 100
        
        stats_text = f'Доходность: {total_return:+.1f}%\nМакс. просадка: {max_dd:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
    def _plot_trades_distribution(self, ax):
        """График распределения результатов сделок"""
        if not self.trades_history:
            ax.text(0.5, 0.5, 'Нет данных о сделках', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            return
            
        pnls = [trade['pnl'] for trade in self.trades_history]
        
        # Гистограмма результатов сделок
        ax.hist(pnls, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Безубыточность')
        
        ax.set_title('📊 Распределение результатов сделок', fontsize=14, fontweight='bold')
        ax.set_xlabel('Прибыль/Убыток за сделку', fontsize=12)
        ax.set_ylabel('Количество сделок', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Статистика сделок
        profitable_trades = len([p for p in pnls if p > 0])
        losing_trades = len([p for p in pnls if p < 0])
        win_rate = (profitable_trades / len(pnls)) * 100 if pnls else 0
        avg_profit = np.mean([p for p in pnls if p > 0]) if profitable_trades > 0 else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if losing_trades > 0 else 0
        
        stats_text = f'Сделок: {len(pnls)}\nВинрейт: {win_rate:.1f}%\nСр. прибыль: {avg_profit:.2f}\nСр. убыток: {avg_loss:.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
    def print_detailed_stats(self, result: dict):
        """Детальная статистика результатов"""
        print("\n📊 ДЕТАЛЬНАЯ СТАТИСТИКА")
        print("=" * 70)
        
        # Основные метрики
        print(f"💰 Начальный капитал:      ${result['initial_value']:,.2f}")
        print(f"💰 Финальный капитал:      ${result['final_value']:,.2f}")
        print(f"📈 Общая доходность:       {result['total_return']:+.2f}%")
        print(f"💵 Абсолютная прибыль:     ${result['profit_loss']:+,.2f}")
        
        # Торговые метрики
        if result.get('total_trades', 0) > 0:
            print(f"\n🔄 Торговая статистика:")
            print(f"   Всего сделок:           {result['total_trades']}")
            print(f"   Выигрышных сделок:      {result.get('won_trades', 0)}")
            print(f"   Проигрышных сделок:     {result.get('lost_trades', 0)}")
            print(f"   Винрейт:                {result.get('win_rate', 0):.1f}%")
            print(f"   Profit Factor:          {result.get('profit_factor', 0):.2f}")
        
        # Риск-метрики
        if 'sharpe_ratio' in result:
            print(f"\n📊 Риск-метрики:")
            print(f"   Коэффициент Шарпа:      {result['sharpe_ratio']:.3f}")
            print(f"   Максимальная просадка:  {result['max_drawdown']:.2f}%")
            
        # Параметры стратегии
        if result.get('parameters'):
            print(f"\n⚙️ Параметры стратегии:")
            for param, value in result['parameters'].items():
                print(f"   {param}: {value}")
                
        print("=" * 70)
        
    def run_interactive_session(self):
        """Запуск интерактивной сессии"""
        print("🎯 ИНТЕРАКТИВНЫЙ БЭКТЕСТЕР")
        print("=" * 50)
        print("Добро пожаловать в интерактивный режим тестирования стратегий!")
        print()
        
        while True:
            # Выбор стратегии
            strategy_name = self.select_strategy_interactive()
            if not strategy_name:
                print("\n👋 До свидания!")
                break
                
            # Настройка параметров
            print(f"\n⚙️ НАСТРОЙКА ПАРАМЕТРОВ")
            print("-" * 30)
            
            # Таймфрейм
            timeframe = self.select_timeframe_interactive()

            # Запуск тестирования
            try:
                result = self.run_enhanced_backtest(
                    strategy_name=strategy_name,
                    timeframe=timeframe
                )
                
                # Детальная статистика
                self.print_detailed_stats(result)
                
                # Вопрос о графиках
                show_plots = input("\n📊 Показать графики? (y/n, по умолчанию y): ").strip().lower()
                if show_plots != 'n':
                    print("\n📈 Построение графиков...")
                    self.plot_enhanced_results(strategy_name, timeframe=timeframe)
                
            except FileNotFoundError as e:
                print(f"\n❌ Ошибка: {e}")
                print(f"💡 Попробуйте использовать таймфрейм 1d")

            except Exception as e:
                print(f"\n❌ Ошибка тестирования: {e}")
                import traceback
                traceback.print_exc()

            # Продолжить?
            continue_testing = input("\n🔄 Протестировать другую стратегию? (y/n): ").strip().lower()
            if continue_testing != 'y':
                print("\n👋 Тестирование завершено!")
                break


def main():
    """Главная функция"""
    print("🚀 ЗАПУСК ИНТЕРАКТИВНОГО БЭКТЕСТЕРА")
    print("=" * 50)
    
    # Создаем бэктестер
    backtester = InteractiveBacktester(initial_cash=100000, commission=0.001)
    
    # Проверяем наличие стратегий
    if not backtester.strategies_registry:
        print("❌ Стратегии не найдены!")
        print("Убедитесь, что в папке strategies/TestStrategies/ есть файлы со стратегиями")
        return
        
    print(f"✅ Найдено {len(backtester.strategies_registry)} стратегий")
    
    # Запускаем интерактивную сессию
    backtester.run_interactive_session()


if __name__ == "__main__":
    main()