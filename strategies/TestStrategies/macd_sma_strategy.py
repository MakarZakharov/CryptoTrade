import pandas as pd
import os
from ta.trend import macd, macd_signal
from typing import Optional, Tuple


def load_csv_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    try:
        df = pd.read_csv(file_path)

        # Перевірка наявності колонки 'close'
        if 'close' not in df.columns:
            raise ValueError("CSV файл повинен містити колонку 'close'")

        # Конвертація колонки close у float
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Видалення рядків з NaN значеннями
        df = df.dropna(subset=['close'])

        if df.empty:
            raise ValueError("Немає валідних даних у колонці 'close'")

        print(f"Завантажено {len(df)} рядків з файлу {file_path}")
        return df

    except Exception as e:
        raise ValueError(f"Помилка при читанні CSV файлу: {str(e)}")


def calculate_indicators(df: pd.DataFrame,
                         macd_fast: int = 12,
                         macd_slow: int = 26,
                         sma_fast: int = 10,
                         sma_slow: int = 50) -> pd.DataFrame:
    """
    Розраховує MACD, сигнальну лінію, SMA та сигнали купівлі/продажу
    """

    df = df.copy()
    df['close'] = df['close'].astype(float)

    # MACD
    df['macd'] = macd(df['close'], window_fast=macd_fast, window_slow=macd_slow)
    df['macd_signal'] = macd_signal(df['close'], window_fast=macd_fast, window_slow=macd_slow)

    # MACD сигнали
    df['buy_signal'] = (df['macd'] > df['macd_signal']) & \
                       (df['macd'].shift(1) <= df['macd_signal'].shift(1)) & \
                       (df['macd'] < 0)

    df['sell_signal'] = (df['macd'] < df['macd_signal']) & \
                        (df['macd'].shift(1) >= df['macd_signal'].shift(1)) & \
                        (df['macd'] > 0)

    # SMA-кросовери
    df['SMA_fast'] = df['close'].rolling(window=sma_fast).mean()
    df['SMA_slow'] = df['close'].rolling(window=sma_slow).mean()
    df['sma_signal'] = 0
    df.loc[df['SMA_fast'] > df['SMA_slow'], 'sma_signal'] = 1
    df.loc[df['SMA_fast'] < df['SMA_slow'], 'sma_signal'] = -1

    return df


def get_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Аналізує останній рядок DataFrame і повертає 'buy', 'sell' або None
    """

    last_row = df.iloc[-1]

    if last_row['buy_signal'] and last_row['sma_signal'] == 1:
        return 'buy'
    elif last_row['sell_signal'] and last_row['sma_signal'] == -1:
        return 'sell'
    else:
        return None


def strategy(df: pd.DataFrame) -> Optional[str]:
    """
    Основна стратегія: приймає df з колонкою 'close', повертає сигнал
    """
    df = calculate_indicators(df)
    return get_signal(df)


def run_strategy_from_csv(csv_file_path: str,
                          macd_fast: int = 12,
                          macd_slow: int = 26,
                          sma_fast: int = 10,
                          sma_slow: int = 50) -> Optional[str]:
    """
    Завантажує дані з CSV і запускає стратегію

    Args:
        csv_file_path (str): Шлях до CSV файлу
        macd_fast, macd_slow, sma_fast, sma_slow: Параметри індикаторів

    Returns:
        Optional[str]: Сигнал ('buy', 'sell' або None)
    """
    try:
        df = load_csv_data(csv_file_path)
        df_with_indicators = calculate_indicators(df, macd_fast, macd_slow, sma_fast, sma_slow)
        signal = get_signal(df_with_indicators)

        print(f"Сигнал: {signal}")
        print(df_with_indicators[['close', 'macd', 'macd_signal', 'buy_signal', 'sell_signal', 'SMA_fast', 'SMA_slow',
                                  'sma_signal']].tail(5))

        return signal

    except Exception as e:
        print(f"Помилка: {str(e)}")
        return None


# Приклад використання:
if __name__ == "__main__":
    # Замініть на шлях до вашого CSV файлу
    csv_path = "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    # Запуск стратегії
    signal = run_strategy_from_csv(csv_path)

    # Або використовуйте існуючий спосіб з DataFrame
    # df = load_csv_data(csv_path)
    # signal = strategy(df)