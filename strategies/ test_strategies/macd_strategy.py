import pandas as pd
import ta
from binance.client import Client

df['macd'] = ta.trend.macd(df['close'])
df['macd_signal'] = ta.trend.macd_signal(df['close'])

# LONG сигнал
df['long_signal'] = (df['macd'] > df['macd_signal']) & \
                    (df['macd'].shift(1) <= df['macd_signal'].shift(1)) & \
                    (df['macd'] < 0)

# SHORT сигнал
df['short_signal'] = (df['macd'] < df['macd_signal']) & \
                     (df['macd'].shift(1) >= df['macd_signal'].shift(1)) & \
                     (df['macd'] > 0)