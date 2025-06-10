import requests
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime
import time

def get_klines(pair, interval, since_ts):
    url = "https://api.kraken.com/0/public/OHLC"
    interval_map = {"1d": 1440, "4h": 240}

    params = {
        "pair": pair,
        "interval": interval_map[interval],
        "since": since_ts
    }

    response = requests.get(url, params=params)
    data = response.json()
    if "error" in data and data["error"]:
        print("Ошибка в ответе API:", data["error"])
        return [], since_ts

    pair_data = list(data["result"].values())[0]
    next_since = data["result"]["last"]
    return pair_data, next_since

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

def save_and_show_full_history(symbol, interval, start_date, filename):
    path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    start_ts = int(pd.Timestamp(start_date).timestamp())
    all_klines = []
    since_ts = start_ts

    print("Начинаю загрузку данных по частям...\n")

    while True:
        print(f"Загружаю данные с {datetime.utcfromtimestamp(since_ts).strftime('%Y-%m-%d %H:%M:%S')}...")
        klines_chunk, next_since = get_klines(symbol, interval, since_ts)

        if not klines_chunk:
            print("Получен пустой ответ или ошибка. Завершаем загрузку.")
            break

        all_klines.extend(klines_chunk)

        # Получаем timestamp последней свечи в этом чанке
        last_timestamp_in_chunk = int(klines_chunk[-1][0])

        if next_since == since_ts or last_timestamp_in_chunk == since_ts:
            print("Достигнут конец доступной истории.")
            break

        # Обновляем since_ts -> last_timestamp_in_chunk + 1 секунда
        since_ts = last_timestamp_in_chunk + 1

        print(f"Продолжаем с {datetime.utcfromtimestamp(since_ts).strftime('%Y-%m-%d %H:%M:%S')}\n")

        time.sleep(1)

    print(f"\nВсего загружено свечей: {len(all_klines)}")

    df = klines_to_dataframe(all_klines)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(path, index=False)
    print(f"CSV сохранён как: {filename}")

    plot_candles(df, symbol, interval)


if __name__ == "__main__":
    save_and_show_full_history(
        symbol="XBTUSD",
        interval="4h",
        start_date="2018-01-01",
        filename="../../../../data/Kraken/BTCUSDT/4h/2018_01_01-now.csv"
    )
