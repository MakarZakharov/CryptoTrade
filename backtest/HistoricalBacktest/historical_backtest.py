import pandas as pd
from CryptoTrade.TestStrategies.first_strategy import sma_crossover


def simple_backtest(df, strategy_func, **kwargs):
    """
    df — DataFrame с колонкой 'close'
    strategy_func — функция-стратегия (например, sma_crossover)
    kwargs — параметры стратегии
    """
    # Получаем DataFrame с сигналами
    df_signals = strategy_func(df, **kwargs)
    capital = 1000
    position = 0
    buy_price = 0

    # Бэктест по сигналам
    for i in range(len(df_signals)):
        if df_signals['signal'].iloc[i] == 1 and position == 0:
            buy_price = df_signals['close'].iloc[i]
            position = 1
        elif df_signals['signal'].iloc[i] == -1 and position == 1:
            sell_price = df_signals['close'].iloc[i]
            capital *= sell_price / buy_price
            position = 0

    if position == 1:
        sell_price = df_signals['close'].iloc[-1]
        capital *= sell_price / buy_price

    print(f'Итоговый капитал: {capital:.2f} USDT')
    return capital

# Пример использования:
if __name__ == "__main__":
    df = pd.read_csv('твой_файл.csv')
    simple_backtest(df, sma_crossover, fast_period=10, slow_period=50)
