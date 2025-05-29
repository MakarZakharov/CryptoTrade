"""
Быстрый запуск бэктеста RSI_SMA_Strategy
"""
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from rsi_sma_backtest import RSI_SMA_BacktestRunner


def quick_test():
    """
    Быстрый тест стратегии с базовыми параметрами
    """
    print("⚡ БЫСТРЫЙ ТЕСТ RSI_SMA_Strategy")
    print("="*40)
    
    # Путь к дневным данным
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"
    )
    
    # Создаем бэктестер
    runner = RSI_SMA_BacktestRunner(initial_cash=10000, commission=0.001)
    
    # Загружаем данные
    print("📊 Загрузка данных...")
    data = runner.load_data_from_csv(data_path)
    if data is None:
        print("❌ Ошибка загрузки данных")
        return
    
    # Быстрые параметры для агрессивной торговли
    params = {
        'rsi_period': 12,
        'rsi_overbought': 65,
        'rsi_oversold': 35,
        'sma_fast': 8,
        'sma_slow': 18,
        'position_size': 0.12,
        'stop_loss': 0.025,
        'take_profit': 0.04,
        'log_enabled': False  # Отключаем детальное логирование для чистого вывода
    }
    
    print("⚙️  Настройка стратегии...")
    runner.setup_cerebro(**params)
    
    print("🚀 Запуск бэктеста...")
    result = runner.run_backtest(data)
    
    print("📈 Анализ результатов...")
    runner.analyze_results(result)
    
    # Показываем график только при запросе
    show_plot = input("\n📊 Показать графики? (y/n): ").lower().strip()
    if show_plot == 'y':
        runner.plot_results()
    
    print("\n✅ Тест завершен!")


if __name__ == "__main__":
    quick_test()