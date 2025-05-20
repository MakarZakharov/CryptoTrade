import pandas as pd
import ta
from binance.client import Client

df['macd'] = ta.trend.macd(df['close'])
df['macd_signal'] = ta.trend.macd_signal(df['close'])

df['buy_signal'] = (df['macd'] > df['macd_signal']) & \
                   (df['macd'].shift(1) <= df['macd_signal'].shift(1)) & \
                   (df['macd'] < 0)

df['sell_signal'] = (df['macd'] < df['macd_signal']) & \
                    (df['macd'].shift(1) >= df['macd_signal'].shift(1)) & \
                    (df['macd'] > 0)
