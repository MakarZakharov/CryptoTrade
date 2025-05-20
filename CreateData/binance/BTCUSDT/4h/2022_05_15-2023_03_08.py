import requests
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime, timedelta

def get_klines(symbol, interval, start_str, end_str=None):
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else None

    all_klines = []
    limit = 1000
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": limit
        }
        if end_ts:
            params["endTime"] = end_ts

        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break

        all_klines += data
        start_ts = data[-1][6] + 1

        if len(data) < limit:
            break
    return all_klines

def klines_to_dataframe(klines):
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype({
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "float",
        "quote_volume": "float"
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
    return df

def plot_candles(df):
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot["timestamp"])
    df_plot = df_plot[["open", "high", "low", "close", "volume"]]
    mpf.plot(df_plot, type="candle", style="charles", volume=True, title="BTCUSDT 4H: 15.05.2022–08.03.2023", ylabel="Цена", ylabel_lower="Объем")

def save_and_show(symbol, interval, start_date, end_date, filename):
    print("Загружаем данные...")
    klines = get_klines(symbol, interval, start_date, end_date)
    df = klines_to_dataframe(klines)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, filename)

    df.to_csv(path, index=False)
    print(f"CSV сохранён как: {filename}")

    print("Открываем график...")
    plot_candles(df)

if __name__ == "__main__":
    save_and_show("BTCUSDT", "4h", "2022-05-15", "2023-03-08", "../../../../data/binance/BTCUSDT/4h/2022_05_15-2023_03_08.csv")