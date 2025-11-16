"""
Универсальный скрипт для сбора данных любой торговой пары с Binance
Использование: python collect_symbol_data.py SYMBOL
Пример: python collect_symbol_data.py BTCUSDC
"""

import sys
import os
import importlib.util

# Загружаем модуль BinanceDataCollector из файла
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, "collect_data_parquet.py")

spec = importlib.util.spec_from_file_location("collect_data_parquet", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BinanceDataCollector = module.BinanceDataCollector


def main():
    """Основная функция запуска"""

    # Получаем символ из аргументов командной строки
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        # Список популярных торговых пар по умолчанию
        print("\n" + "="*60)
        print("🔍 Доступные торговые пары для сбора данных:")
        print("="*60)

        symbols = [
            "BTCUSDT",  # Bitcoin / Tether
            "BTCUSDC",  # Bitcoin / USD Coin
            "ETHUSDT",  # Ethereum / Tether
            "ETHUSDC",  # Ethereum / USD Coin
            "BNBUSDT",  # Binance Coin / Tether
            "XRPUSDT",  # Ripple / Tether
            "SOLUSDT",  # Solana / Tether
            "ADAUSDT",  # Cardano / Tether
        ]

        for i, sym in enumerate(symbols, 1):
            print(f"   {i}. {sym}")

        print("\n💡 Использование:")
        print(f"   python collect_symbol_data.py BTCUSDC")
        print(f"   python collect_symbol_data.py ETHUSDT")
        print("\n")

        choice = input("Введите номер или название торговой пары (по умолчанию BTCUSDC): ").strip()

        if not choice:
            symbol = "BTCUSDC"
        elif choice.isdigit() and 1 <= int(choice) <= len(symbols):
            symbol = symbols[int(choice) - 1]
        else:
            symbol = choice.upper()

    print("\n" + "="*60)
    print(f" "*10 + f"🚀 BINANCE DATA COLLECTOR: {symbol}")
    print(" "*20 + "Parquet Edition")
    print("="*60 + "\n")

    # Создаем экземпляр коллектора для выбранного символа
    collector = BinanceDataCollector(symbol=symbol)

    # Собираем данные для всех таймфреймов
    results = collector.collect_all_timeframes()

    if not results:
        print(f"❌ Не удалось собрать данные для {symbol}!")
        return

    # Генерируем сводный отчет
    collector.generate_summary_report(results)

    # Строим графики
    collector.plot_all_data(results)

    # Строим свечные графики
    collector.plot_candlestick_charts(results)

    print("\n" + "="*60)
    print(" "*15 + f"✅ {symbol} - ГОТОВО!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
