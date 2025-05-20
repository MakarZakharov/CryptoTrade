import pandas as pd

def sma_crossover(df, fast_period=10, slow_period=50):
    """
    SMA crossover стратегия.
    Возвращает DataFrame с колонкой 'signal': 1 (buy), -1 (sell), 0 (hold)
    """
    df = df.copy()
    df['SMA_fast'] = df['close'].rolling(window=fast_period).mean()
    df['SMA_slow'] = df['close'].rolling(window=slow_period).mean()
    df['signal'] = 0
    df.loc[df['SMA_fast'] > df['SMA_slow'], 'signal'] = 1
    df.loc[df['SMA_fast'] < df['SMA_slow'], 'signal'] = -1
    return df