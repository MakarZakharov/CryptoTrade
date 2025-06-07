import requests
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime
import time

def get_klines(symbol, interval, start_str, end_str=None):
    url = "https://api.kucoin.com/api/v1/market/candles"
    start_ts = int(pd.Timestamp(start_str).timestamp())
    end_ts = int(pd.Timestamp(end_str).timestamp()) if end_str else int(time.time())
    interval_map = {"1d": "1day", "4h": "4hour"}
    all_klines = []
    step = 1500
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "type": interval_map[interval],
            "startAt": start_ts,
            "endAt": end_ts
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Check for successful response and data availability
        if data.get("code") != "200000" or not data.get("data"):
            print(f"API Error or no data: {data.get('msg', 'Unknown error')}")
            break

        klines = sorted(data["data"], key=lambda x: int(x[0])) # Ensure sorting by integer timestamp
        all_klines.extend(klines)

        if len(klines) < step:
            break

        start_ts = int(klines[-1][0]) + 1
    return all_klines

def klines_to_dataframe(klines):
    df = pd.DataFrame(klines, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df

def plot_candles(df, symbol, interval):
    df_plot = df.copy()
    # Set the 'timestamp' column as the DataFrame index for mplfinance
    df_plot.index = df_plot["timestamp"]
    df_plot = df_plot[["open", "high", "low", "close", "volume"]]

    # Generate plot title
    title = f"{symbol} {interval} KuCoin: {df_plot.index.min().strftime('%d.%m.%Y')}–{datetime.today().strftime('%d.%m.%Y')}"

    # Plot the candlestick chart
    mpf.plot(df_plot, type="candle", style="charles", volume=True, title=title)

def save_and_show(symbol, interval, start_date, filename):
    end_date = datetime.today().strftime("%Y-%m-%d")
    path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(3):
        print(f"Попытка {attempt + 1} загрузки с KuCoin...")
        klines = get_klines(symbol, interval, start_date, end_date)
        df = klines_to_dataframe(klines)

        if df.shape[0] > 50: # Arbitrary threshold for "good data"
            df.to_csv(path, index=False)
            print(f"CSV сохранён как: {filename}")
            plot_candles(df, symbol, interval)
            return # Exit after successful operation
        else:
            print(f"Недостаточно данных ({df.shape[0]} строк), повтор...\n")
            time.sleep(2) # Wait before retrying

    print(f"Не удалось получить корректные данные для {symbol} после 3 попыток.")

if __name__ == "__main__":
    save_and_show(
        symbol="BTC-USDT",
        interval="4h",
        start_date="2018-01-01",
        filename="../../../../data/KuCoin/BTCUSDT/4h/2018_01_01-now.csv"
    )
