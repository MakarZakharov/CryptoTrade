import requests
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime
import time

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
        if not data or isinstance(data, dict) and "code" in data:
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

def plot_candles(df, symbol, interval):
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot["timestamp"])
    df_plot = df_plot[["open", "high", "low", "close", "volume"]]
    title = f"{symbol} {interval}: 01.01.2018–{datetime.today().strftime('%d.%m.%Y')}"
    mpf.plot(df_plot, type="candle", style="charles", volume=True, title=title, ylabel="Цена", ylabel_lower="Объем")

def save_and_show(symbol, interval, start_date, filename):
    end_date = datetime.today().strftime("%Y-%m-%d")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(3):
        print(f"Попытка {attempt + 1} загрузки данных для {symbol}...")
        klines = get_klines(symbol, interval, start_date, end_date)
        df = klines_to_dataframe(klines)

        if df.shape[0] > 50 and not df.isnull().values.any():
            df.to_csv(path, index=False)
            print(f"CSV сохранён как: {filename}")
            print("Открываем график...")
            plot_candles(df, symbol, interval)
            return
        else:
            print("Данные некорректны, повторная попытка...\n")
            time.sleep(2)

    print(f"Не удалось получить корректные данные для {symbol} после 3 попыток.")

if __name__ == "__main__":
    save_and_show(
        symbol="SOLUSDC",
        interval="4h",
        start_date="2018-01-01",
        filename="../../../../data/binance/SOLUSDC/4h/2021_09_24-now.csv"
    )