import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

def get_mexc_klines(symbol="BTCUSDT", interval="1d", start_str="2018-01-01", end_str=None):
    url = "https://api.mexc.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)

    all_klines = []
    limit = 1000
    print("Загрузка данных с MEXC...")

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": limit
        }

        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            print("Нет данных, остановка.")
            break

        all_klines += data
        last_time = data[-1][0]
        start_ts = last_time + 1  # следующий миллисекунд
        time.sleep(0.2)

        if len(data) < limit:
            break

    print(f"Получено свечей: {len(all_klines)}")

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

def save_mexc():
    df = get_mexc_klines(start_str="2018-01-01")
    if df.shape[0] < 100 or df.isnull().values.any():
        print("Проблема с полученными данными от MEXC.")
        return

    path = "../../../../data/MEXC/BTCUSDT/1d/2018_01_01-now.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"MEXC CSV сохранён: {path}")
    print(f"Диапазон данных: {df['timestamp'].min().date()} — {df['timestamp'].max().date()}")
    print(f"Общее количество дневных свечей: {df.shape[0]}")

if __name__ == "__main__":
    save_mexc()
