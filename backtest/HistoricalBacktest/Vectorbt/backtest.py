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


def load_data(data_path: str) -> pd.DataFrame:
    """
    Загрузка данных из CSV файла
    
    Args:
        data_path: Путь к файлу с данными
        
    Returns:
        DataFrame с данными
    """
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


def run_backtest(data: pd.DataFrame, strategy_params: dict, 
                 initial_cash: float = 100000, fees: float = 0.001) -> dict:
    """
    Запуск бэктеста RSI стратегии
    
    Args:
        data: DataFrame с данными
        strategy_params: Параметры стратегии
        initial_cash: Начальный капитал
        fees: Комиссия за сделку
        
    Returns:
        Словарь с результатами
    """
    print(f"\n🚀 Запуск бэктеста: {strategy_params.get('strategy_name', 'RSI Strategy')}")
    print("=" * 60)
    
    # Создание стратегии
    strategy = RSIVectorbtStrategy(
        rsi_period=strategy_params.get('rsi_period', 14),
        oversold=strategy_params.get('oversold', 30),
        overbought=strategy_params.get('overbought', 70)
    )
    
    # Запуск бэктеста
    portfolio = strategy.backtest(data, initial_cash, fees)
    
    # Получение результатов
    results = {
        'portfolio': portfolio,
        'strategy_params': strategy.get_strategy_params(),
        'initial_cash': initial_cash,
        'final_value': portfolio.value().iloc[-1],
        'total_return': portfolio.total_return(),
        'total_return_pct': portfolio.total_return() * 100,
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'max_drawdown_pct': portfolio.max_drawdown() * 100,
        'total_trades': portfolio.orders.count(),
        'win_rate': portfolio.trades.win_rate() if portfolio.trades.count() > 0 else 0,
        'profit_factor': portfolio.trades.profit_factor() if portfolio.trades.count() > 0 else 0,
        'avg_trade_duration': portfolio.trades.duration.mean() if portfolio.trades.count() > 0 else 0,
        'data_period': f"{data.index[0]} - {data.index[-1]}",
        'data_points': len(data)
    }
    
    return results


def print_results(results: dict):
    """
    Вывод результатов бэктеста
    
    Args:
        results: Словарь с результатами
    """
    print(f"\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("=" * 60)
    print(f"🎯 Стратегия: {results['strategy_params']['strategy_name']}")
    print(f"📅 Период: {results['data_period']}")
    print(f"📈 Точек данных: {results['data_points']}")
    
    print(f"\n💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Начальный капитал: ${results['initial_cash']:,.2f}")
    print(f"   Конечная стоимость: ${results['final_value']:,.2f}")
    print(f"   Общая доходность: {results['total_return_pct']:.2f}%")
    print(f"   Прибыль/Убыток: ${results['final_value'] - results['initial_cash']:,.2f}")
    
    print(f"\n📈 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print(f"   Коэффициент Шарпа: {results['sharpe_ratio']:.2f}")
    print(f"   Максимальная просадка: {results['max_drawdown_pct']:.2f}%")
    print(f"   Коэффициент прибыли: {results['profit_factor']:.2f}")
    
    print(f"\n🔄 ТОРГОВАЯ АКТИВНОСТЬ:")
    print(f"   Всего сделок: {results['total_trades']}")
    print(f"   Винрейт: {results['win_rate']*100:.1f}%")
    print(f"   Средняя длительность сделки: {results['avg_trade_duration']:.1f} дней")
    
    print(f"\n⚙️ ПАРАМЕТРЫ СТРАТЕГИИ:")
    for key, value in results['strategy_params'].items():
        if key != 'strategy_name':
            print(f"   {key}: {value}")
    print("=" * 60)


def create_plots(results: dict, save_plots: bool = True, plots_dir: str = "plots"):
    """
    Создание графиков результатов
    
    Args:
        results: Словарь с результатами
        save_plots: Сохранять ли графики
        plots_dir: Директория для сохранения графиков
    """
    import matplotlib
    matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system

    portfolio = results['portfolio']
    
    # Создаем директорию для графиков
    if save_plots:
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # График стоимости портфеля
    plt.figure(figsize=(15, 10))
    
    # Подграфик 1: Стоимость портфеля
    plt.subplot(2, 2, 1)
    portfolio.value().plot(title='Стоимость портфеля', color='blue')
    plt.ylabel('Стоимость ($)')
    plt.grid(True, alpha=0.3)
    
    # Подграфик 2: Просадка
    plt.subplot(2, 2, 2)
    portfolio.drawdown().plot.area(title='Просадка', color='red', alpha=0.7)
    plt.ylabel('Просадка')
    plt.grid(True, alpha=0.3)
    
    # Подграфик 3: Цена и сигналы
    plt.subplot(2, 2, 3)
    portfolio.close.plot(title='Цена и торговые сигналы', color='black', alpha=0.7)
    
    # Добавляем сигналы покупки и продажи
    if portfolio.orders.count() > 0:
        # Получаем ордера покупки и продажи
        orders = portfolio.orders.records_readable
        if len(orders) > 0:
            buy_orders = orders[orders['Side'] == 'Buy']
            sell_orders = orders[orders['Side'] == 'Sell']
            
            if len(buy_orders) > 0:
                buy_timestamps = pd.to_datetime(buy_orders['Timestamp']).values
                buy_prices = buy_orders['Price'].values
                plt.scatter(buy_timestamps, buy_prices, 
                           color='green', marker='^', s=100, label='Покупка')
            if len(sell_orders) > 0:
                sell_timestamps = pd.to_datetime(sell_orders['Timestamp']).values
                sell_prices = sell_orders['Price'].values
                plt.scatter(sell_timestamps, sell_prices, 
                           color='red', marker='v', s=100, label='Продажа')
            plt.legend()
    
    plt.ylabel('Цена ($)')
    plt.grid(True, alpha=0.3)
    
    # Подграфик 4: Кумулятивные доходности
    plt.subplot(2, 2, 4)
    portfolio.cumulative_returns().plot(title='Кумулятивная доходность', color='purple')
    plt.ylabel('Кумулятивная доходность')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{plots_dir}/backtest_results.png", dpi=300, bbox_inches='tight')
        print(f"📊 Графики сохранены в {plots_dir}/backtest_results.png")
    
    plt.show()





def main():
    """Основная функция"""
    # Путь к данным
    data_path = r"C:\Users\Макар\PycharmProjects\trading\CryptoTrade\data\binance\BTCUSDT\1d\2018_01_01-now.csv"
    
    # Загрузка данных
    data = load_data(data_path)
    if data is None:
        return
    
    # Параметры стратегии
    strategy_params = {
        'rsi_period': 14,
        'oversold': 24,
        'overbought': 70,
        'strategy_name': 'RSI Vectorbt Strategy'
    }
    
    # Параметры бэктеста
    initial_cash = 100000
    fees = 0.001  # 0.1%
    
    # Запуск бэктеста
    results = run_backtest(data, strategy_params, initial_cash, fees)
    
    # Вывод результатов
    print_results(results)
    
    # Создание графиков
    create_plots(results, save_plots=True)


if __name__ == "__main__":
    main()