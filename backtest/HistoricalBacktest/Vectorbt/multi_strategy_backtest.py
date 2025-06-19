import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Добавляем путь к стратегиям
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'strategies', 'Vectorbt', 'TestStrategies'))

from RSI import RSIVectorbtStrategy
from MovingAverage import MovingAverageCrossoverStrategy
from BollingerBands import BollingerBandsStrategy, BollingerBandsMeanReversionStrategy


def load_data(data_path: str) -> pd.DataFrame:
    """Загрузка данных из CSV файла"""
    try:
        data = pd.read_csv(data_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        print(f"📊 Загрузка данных: {data_path}")
        print(f"✅ Загружено {len(data)} записей с {data.index[0]} по {data.index[-1]}")
        
        return data
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return None


def run_multiple_strategies(data: pd.DataFrame, initial_cash: float = 100000, fees: float = 0.001):
    """
    Запуск нескольких стратегий для сравнения
    
    Args:
        data: DataFrame с данными
        initial_cash: Начальный капитал
        fees: Комиссия за сделку
        
    Returns:
        Словарь с результатами всех стратегий
    """
    strategies = {
        'RSI Strategy': RSIVectorbtStrategy(rsi_period=14, oversold=30, overbought=70),
        'MA Crossover (20/50)': MovingAverageCrossoverStrategy(fast_period=20, slow_period=50),
        'MA Crossover (10/30)': MovingAverageCrossoverStrategy(fast_period=10, slow_period=30),
        'Bollinger Bands Breakout': BollingerBandsStrategy(period=20, std_dev=2.0),
        'Bollinger Bands Mean Reversion': BollingerBandsMeanReversionStrategy(period=20, std_dev=2.0, exit_at_middle=True)
    }
    
    results = {}
    
    print("\n🚀 Запуск сравнения стратегий")
    print("=" * 80)
    
    for name, strategy in strategies.items():
        print(f"\n📊 Тестирование: {name}")
        
        # Запуск бэктеста
        portfolio = strategy.backtest(data, initial_cash, fees)
        
        # Сбор результатов
        results[name] = {
            'portfolio': portfolio,
            'strategy_params': strategy.get_strategy_params(),
            'total_return': portfolio.total_return(),
            'total_return_pct': portfolio.total_return() * 100,
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown(),
            'max_drawdown_pct': portfolio.max_drawdown() * 100,
            'total_trades': portfolio.orders.count(),
            'win_rate': portfolio.trades.win_rate() if portfolio.trades.count() > 0 else 0,
            'profit_factor': portfolio.trades.profit_factor() if portfolio.trades.count() > 0 else 0
        }
        
        print(f"   Доходность: {results[name]['total_return_pct']:.2f}%")
        print(f"   Винрейт: {results[name]['win_rate']*100:.1f}%")
        print(f"   Сделок: {results[name]['total_trades']}")
    
    return results


def print_comparison_table(results: dict):
    """
    Вывод сравнительной таблицы результатов
    
    Args:
        results: Словарь с результатами всех стратегий
    """
    print(f"\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА СТРАТЕГИЙ")
    print("=" * 120)
    
    # Заголовок таблицы
    header = f"{'Стратегия':<35} {'Доходность':<12} {'Шарп':<8} {'Просадка':<10} {'Сделки':<8} {'Винрейт':<8} {'PF':<6}"
    print(header)
    print("-" * 120)
    
    # Строки с результатами
    for name, result in results.items():
        row = (f"{name:<35} "
               f"{result['total_return_pct']:>10.2f}% "
               f"{result['sharpe_ratio']:>7.2f} "
               f"{result['max_drawdown_pct']:>9.2f}% "
               f"{result['total_trades']:>7} "
               f"{result['win_rate']*100:>7.1f}% "
               f"{result['profit_factor']:>5.2f}")
        print(row)
    
    print("=" * 120)
    
    # Лучшие результаты
    best_return = max(results.items(), key=lambda x: x[1]['total_return_pct'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    lowest_drawdown = min(results.items(), key=lambda x: abs(x[1]['max_drawdown_pct']))
    
    print(f"\n🏆 ЛУЧШИЕ ПОКАЗАТЕЛИ:")
    print(f"   Лучшая доходность: {best_return[0]} ({best_return[1]['total_return_pct']:.2f}%)")
    print(f"   Лучший коэф. Шарпа: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})")
    print(f"   Наименьшая просадка: {lowest_drawdown[0]} ({lowest_drawdown[1]['max_drawdown_pct']:.2f}%)")


def create_comparison_plots(results: dict, save_plots: bool = True, plots_dir: str = "plots"):
    """
    Создание сравнительных графиков
    
    Args:
        results: Словарь с результатами всех стратегий
        save_plots: Сохранять ли графики
        plots_dir: Директория для сохранения графиков
    """
    import matplotlib
    matplotlib.use('TkAgg')
    
    # Создаем директорию для графиков
    if save_plots:
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # График 1: Сравнение стоимости портфелей
    plt.figure(figsize=(15, 12))
    
    # Подграфик 1: Эволюция портфелей
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        portfolio_value = result['portfolio'].value()
        plt.plot(portfolio_value.index, portfolio_value.values, label=name, alpha=0.8)
    
    plt.title('Сравнение стоимости портфелей')
    plt.ylabel('Стоимость ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Подграфик 2: Просадки
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        drawdown = result['portfolio'].drawdown()
        plt.plot(drawdown.index, drawdown.values * 100, label=name, alpha=0.8)
    
    plt.title('Сравнение просадок')
    plt.ylabel('Просадка (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Подграфик 3: Доходность по стратегиям (барчарт)
    plt.subplot(2, 2, 3)
    strategy_names = list(results.keys())
    returns = [results[name]['total_return_pct'] for name in strategy_names]
    
    bars = plt.bar(range(len(strategy_names)), returns, alpha=0.7)
    plt.title('Общая доходность по стратегиям')
    plt.ylabel('Доходность (%)')
    plt.xticks(range(len(strategy_names)), [name.replace(' ', '\n') for name in strategy_names], rotation=45, ha='right')
    
    # Цветовая кодировка баров
    for i, bar in enumerate(bars):
        if returns[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # Подграфик 4: Коэффициент Шарпа (барчарт)
    plt.subplot(2, 2, 4)
    sharpe_ratios = [results[name]['sharpe_ratio'] for name in strategy_names]
    
    bars = plt.bar(range(len(strategy_names)), sharpe_ratios, alpha=0.7, color='purple')
    plt.title('Коэффициент Шарпа по стратегиям')
    plt.ylabel('Коэффициент Шарпа')
    plt.xticks(range(len(strategy_names)), [name.replace(' ', '\n') for name in strategy_names], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{plots_dir}/strategies_comparison.png", dpi=300, bbox_inches='tight')
        print(f"📊 Сравнительные графики сохранены в {plots_dir}/strategies_comparison.png")
    
    plt.show()


def main():
    """Основная функция"""
    # Путь к данным
    data_path = r"C:\Users\Макар\PycharmProjects\trading\CryptoTrade\data\binance\BTCUSDT\1d\2018_01_01-now.csv"
    
    # Загрузка данных
    data = load_data(data_path)
    if data is None:
        return
    
    # Параметры бэктеста
    initial_cash = 100000
    fees = 0.001  # 0.1%
    
    # Запуск сравнения стратегий
    results = run_multiple_strategies(data, initial_cash, fees)
    
    # Вывод сравнительной таблицы
    print_comparison_table(results)
    
    # Создание сравнительных графиков
    create_comparison_plots(results, save_plots=True)


if __name__ == "__main__":
    main()