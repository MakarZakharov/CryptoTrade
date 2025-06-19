import requests
import pandas as pd
import time
from datetime import datetime
import os
import mplfinance as mpf

def get_mexc_klines(symbol="BTCUSDT", interval="1d", start_str="2018-01-01", end_str=None):
    url = "https://api.mexc.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)
    all_klines = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_klines += data
        start_ts = data[-1][6] + 1
        time.sleep(0.2)

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "quote_volume": float
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
    return df

def plot_if_possible(df, symbol):
    if df.shape[0] > 500:
        df_plot = df.copy()
        df_plot.set_index("timestamp", inplace=True)
        df_plot = df_plot[["open", "high", "low", "close", "volume"]]
        title = f"{symbol} MEXC 1d Candles"
        mpf.plot(df_plot, type="candle", style="charles", volume=True, title=title, ylabel="Цена", ylabel_lower="Объём")
    else:
        print("Недостаточно данных для построения графика.")

def save_mexc():
    df = get_mexc_klines()
    if df.shape[0] < 100 or df.isnull().values.any():
        print("Проблема с полученными данными от MEXC.")
        return
    path = "../../../../data/MEXC/BTCUSDT/1d/2018_01_01-now.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"MEXC CSV сохранён: {path}")
    plot_if_possible(df, "BTCUSDT")

if __name__ == "__main__":
    save_mexc()
