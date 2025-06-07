# Всё то же, только API другой
import requests
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime
import time

def get_klines(pair, interval, start_str):
    url = "https://api.kraken.com/0/public/OHLC"
    interval_map = {"1d": 1440, "4h": 240}
    start_ts = int(pd.Timestamp(start_str).timestamp())

    params = {
        "pair": pair,  # например: XBTUSDT
        "interval": interval_map[interval],
        "since": start_ts
    }

    response = requests.get(url, params=params)
    data = response.json()
    if "error" in data and data["error"]:
        return []

    pair_data = list(data["result"].values())[0]
    return pair_data

def klines_to_dataframe(klines):
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df

def plot_candles(df, symbol, interval):
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot["timestamp"])
    df_plot = df_plot[["open", "high", "low", "close", "volume"]]
    title = f"{symbol} {interval} Kraken: 01.01.2018–{datetime.today().strftime('%d.%m.%Y')}"
    mpf.plot(df_plot, type="candle", style="charles", volume=True, title=title)

def save_and_show(symbol, interval, start_date, filename):
    path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(3):
        print(f"Попытка {attempt + 1} загрузки с Kraken...")
        klines = get_klines(symbol, interval, start_date)
        df = klines_to_dataframe(klines)

        if df.shape[0] > 50:
            df.to_csv(path, index=False)
            print(f"CSV сохранён как: {filename}")
            plot_candles(df, symbol, interval)
            return
        else:
            print("Плохие данные, повтор...\n")
            time.sleep(2)

    print(f"Не удалось получить корректные данные для {symbol} после 3 попыток.")

if __name__ == "__main__":
    save_and_show(
        symbol="XBTUSDT",
        interval="1d",
        start_date="2018-01-01",
        filename="../../../../data/Kraken/BTCUSDT/1d/2018_01_01-now.csv"
    )
