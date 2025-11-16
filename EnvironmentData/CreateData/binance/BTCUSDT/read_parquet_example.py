"""
Пример чтения и использования данных из Parquet файлов
"""

import pandas as pd
import os


def read_btcusdt_data(timeframe='1d'):
    """
    Чтение данных BTCUSDT из Parquet файла

    Args:
        timeframe (str): Таймфрейм (15m, 1h, 4h, 1d)

    Returns:
        pd.DataFrame: DataFrame с данными
    """
    # Путь к файлу
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(current_dir, "..", "..", "Date", "binance", "BTCUSDT", timeframe)

    # Находим файл parquet
    parquet_files = [f for f in os.listdir(parquet_path) if f.endswith('.parquet')]

    if not parquet_files:
        raise FileNotFoundError(f"Не найдены Parquet файлы в {parquet_path}")

    # Читаем первый найденный файл
    filepath = os.path.join(parquet_path, parquet_files[0])

    print(f"📖 Чтение файла: {filepath}")
    df = pd.read_parquet(filepath)

    # Устанавливаем timestamp как индекс
    df.set_index('timestamp', inplace=True)

    return df


def display_data_info(df, timeframe):
    """Вывод информации о данных"""
    print(f"\n{'='*60}")
    print(f"📊 Данные BTCUSDT - {timeframe.upper()}")
    print(f"{'='*60}\n")

    print(f"📅 Период: {df.index.min()} - {df.index.max()}")
    print(f"📈 Количество свечей: {len(df):,}")
    print(f"💾 Размер в памяти: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} МБ\n")

    print("📋 Структура данных:")
    print(df.info())

    print(f"\n📊 Статистика:")
    print(df.describe())

    print(f"\n🔝 Первые 5 записей:")
    print(df.head())

    print(f"\n🔚 Последние 5 записей:")
    print(df.tail())

    print(f"\n💰 Минимальная цена: ${df['low'].min():,.2f}")
    print(f"💰 Максимальная цена: ${df['high'].max():,.2f}")
    print(f"💰 Средняя цена закрытия: ${df['close'].mean():,.2f}")
    print(f"📊 Общий объем торгов: {df['volume'].sum():,.2f} BTC")


def get_date_range(df, start_date=None, end_date=None):
    """
    Получение данных за определенный период

    Args:
        df (pd.DataFrame): DataFrame с данными
        start_date (str): Начальная дата (формат: 'YYYY-MM-DD')
        end_date (str): Конечная дата (формат: 'YYYY-MM-DD')

    Returns:
        pd.DataFrame: Отфильтрованный DataFrame
    """
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    return df


def calculate_returns(df):
    """Расчет доходности"""
    df = df.copy()

    # Дневная доходность
    df['daily_return'] = df['close'].pct_change()

    # Кумулятивная доходность
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

    return df


def main():
    """Пример использования"""
    print("\n" + "="*60)
    print(" "*15 + "🔍 ПРИМЕР ЧТЕНИЯ ДАННЫХ")
    print("="*60 + "\n")

    # Доступные таймфреймы
    timeframes = ['15m', '1h', '4h', '1d']

    print("📊 Доступные таймфреймы:")
    for i, tf in enumerate(timeframes, 1):
        print(f"   {i}. {tf}")

    # Читаем данные для дневного таймфрейма
    print("\n🔄 Загрузка дневных данных (1d)...\n")
    df_daily = read_btcusdt_data('1d')

    # Выводим информацию
    display_data_info(df_daily, '1d')

    # Пример фильтрации по дате
    print(f"\n\n{'='*60}")
    print("📅 Пример фильтрации по дате (2024 год)")
    print(f"{'='*60}\n")

    df_2024 = get_date_range(df_daily, start_date='2024-01-01', end_date='2024-12-31')
    print(f"📊 Найдено записей за 2024 год: {len(df_2024)}")
    print(f"💰 Цена на начало 2024: ${df_2024.iloc[0]['close']:,.2f}")
    print(f"💰 Текущая цена: ${df_2024.iloc[-1]['close']:,.2f}")

    # Расчет доходности
    print(f"\n\n{'='*60}")
    print("📈 Расчет доходности")
    print(f"{'='*60}\n")

    df_with_returns = calculate_returns(df_2024)
    total_return = df_with_returns['cumulative_return'].iloc[-1] * 100

    print(f"📊 Доходность за 2024 год: {total_return:+.2f}%")
    print(f"📊 Средняя дневная доходность: {df_with_returns['daily_return'].mean()*100:.4f}%")
    print(f"📊 Волатильность (std): {df_with_returns['daily_return'].std()*100:.4f}%")

    # Пример работы с часовыми данными
    print(f"\n\n{'='*60}")
    print("⏰ Загрузка часовых данных (1h)")
    print(f"{'='*60}\n")

    df_hourly = read_btcusdt_data('1h')
    print(f"📊 Загружено {len(df_hourly):,} часовых свечей")
    print(f"📅 Период: {df_hourly.index.min()} - {df_hourly.index.max()}")

    # Последние 24 часа
    last_24h = df_hourly.tail(24)
    price_change_24h = ((last_24h['close'].iloc[-1] / last_24h['close'].iloc[0]) - 1) * 100
    print(f"\n📊 Изменение цены за последние 24 часа: {price_change_24h:+.2f}%")
    print(f"💰 Максимум за 24 часа: ${last_24h['high'].max():,.2f}")
    print(f"💰 Минимум за 24 часа: ${last_24h['low'].min():,.2f}")

    print("\n" + "="*60)
    print(" "*20 + "✅ ГОТОВО!")
    print("="*60 + "\n")

    # Совет по использованию
    print("💡 Совет: Вы можете легко использовать эти данные для:")
    print("   • Backtesting торговых стратегий")
    print("   • Обучения ML моделей")
    print("   • Технического анализа")
    print("   • Статистических исследований")
    print()


if __name__ == "__main__":
    main()
