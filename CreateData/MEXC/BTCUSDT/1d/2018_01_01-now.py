import requests
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime
import time

def get_klines(symbol, interval, start_str, end_str=None):
    url = "https://api.mexc.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)  # миллисекунды
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)

    all_klines = []
    limit = 1000

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data or isinstance(data, dict) and "code" in data:
            print(f"Ошибка ответа от API: {data}")
            break

        all_klines += data

        last_candle_close_ts = int(data[-1][0])  # timestamp в миллисекундах
        if last_candle_close_ts >= end_ts or len(data) < limit:
            break

        start_ts = last_candle_close_ts + 1  # следующий миллисекунд

        time.sleep(0.2)  # чтобы избежать rate limit

    return all_klines


def klines_to_dataframe(klines):
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "ignore"
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
    title = f"{symbol} {interval} MEXC: {df_plot.index.min().strftime('%d.%m.%Y')}–{datetime.today().strftime('%d.%m.%Y')}"
    mpf.plot(df_plot, type="candle", style="charles", volume=True, title=title)


def save_and_show(symbol, interval, start_date, filename):
    end_date = datetime.today().strftime("%Y-%m-%d")
    path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(3):
        print(f"Попытка {attempt + 1} загрузки с MEXC...")
        try:
            klines = get_klines(symbol, interval, start_date, end_date)
            df = klines_to_dataframe(klines)

            if df.shape[0] > 50:
                df.to_csv(path, index=False)
                print(f"CSV сохранён как: {filename}")
                plot_candles(df, symbol, interval)
                return
            else:
                print("Недостаточно данных, повтор...\n")
                time.sleep(2)

        except Exception as e:
            print(f"Ошибка: {e}\nПовтор через 2 секунды...\n")
            time.sleep(2)

    print(f"❌ Не удалось получить корректные данные для {symbol} после 3 попыток.")


if __name__ == "__main__":
    save_and_show(
        symbol="BTCUSDT",
        interval="1d",
        start_date="2018-01-01",
        filename="../../../../data/MEXC/BTCUSDT/1d/2018_01_01-now.csv"
    )
