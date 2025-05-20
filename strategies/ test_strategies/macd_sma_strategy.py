import pandas as pd
from ta.trend import macd, macd_signal
from typing import Optional, Tuple


def calculate_indicators(df: pd.DataFrame,
                         macd_fast: int = 12,
                         macd_slow: int = 26,
                         sma_fast: int = 10,
                         sma_slow: int = 50) -> pd.DataFrame:

    df = df.copy()
    df['close'] = df['close'].astype(float)

    df['macd'] = macd(df['close'], window_fast=macd_fast, window_slow=macd_slow)
    df['macd_signal'] = macd_signal(df['close'], window_fast=macd_fast, window_slow=macd_slow)

    df['buy_signal'] = (df['macd'] > df['macd_signal']) & \
                       (df['macd'].shift(1) <= df['macd_signal'].shift(1)) & \
                       (df['macd'] < 0)

    df['sell_signal'] = (df['macd'] < df['macd_signal']) & \
                        (df['macd'].shift(1) >= df['macd_signal'].shift(1)) & \
                        (df['macd'] > 0)

    df['SMA_fast'] = df['close'].rolling(window=sma_fast).mean()
    df['SMA_slow'] = df['close'].rolling(window=sma_slow).mean()
    df['sma_signal'] = 0
    df.loc[df['SMA_fast'] > df['SMA_slow'], 'sma_signal'] = 1
    df.loc[df['SMA_fast'] < df['SMA_slow'], 'sma_signal'] = -1

    return df


def get_signal(df: pd.DataFrame) -> Optional[str]:

    last_row = df.iloc[-1]

    if last_row['buy_signal'] and last_row['sma_signal'] == 1:
        return 'buy'
    elif last_row['sell_signal'] and last_row['sma_signal'] == -1:
        return 'sell'
    else:
        return None


def strategy(df: pd.DataFrame) -> Optional[str]:
    df = calculate_indicators(df)
    return get_signal(df)