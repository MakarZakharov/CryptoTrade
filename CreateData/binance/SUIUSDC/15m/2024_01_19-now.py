import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def get_klines(symbol, interval, start_ts, end_ts):
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    limit = 1000

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception as e:
            print("Ошибка запроса:", e)
            time.sleep(1)
            continue

        if not data or isinstance(data, dict) and "code" in data:
            print("Ошибка в данных:", data)
            break

        all_klines += data
        start_ts = data[-1][6] + 1

        if len(data) < limit:
            break
        time.sleep(0.1)
    return all_klines

def klines_to_dataframe(klines):
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype({
        "open": "float", "high": "float", "low": "float", "close": "float",
        "volume": "float", "quote_volume": "float"
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
    return df

def save_and_show(symbol, interval, start_date_str, filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.today()
    chunk_size_days = 10  # меньше дней в одном запросе = меньше данных в памяти

    # Если файл уже существует, загружаем только недостающие данные
    if os.path.exists(path):
        print("Файл существует. Пропускаем уже загруженные данные.")
        existing_df = pd.read_csv(path)
        last_timestamp = pd.to_datetime(existing_df["timestamp"]).max()
        start_date = last_timestamp + timedelta(minutes=15)
    else:
        existing_df = pd.DataFrame()

    with open(path, 'a', encoding='utf-8', newline='') as f:
        if existing_df.empty:
            f.write("timestamp,open,high,low,close,volume,quote_volume\n")

        while start_date < end_date:
            chunk_end = min(start_date + timedelta(days=chunk_size_days), end_date)
            print(f"  >> {start_date.date()} — {chunk_end.date()}")

            klines = get_klines(
                symbol, interval,
                int(start_date.timestamp() * 1000),
                int(chunk_end.timestamp() * 1000)
            )
            df_chunk = klines_to_dataframe(klines)

            if not df_chunk.empty:
                df_chunk.to_csv(f, header=False, index=False)

            start_date = chunk_end
            time.sleep(0.1)

    print(f"Готово: данные сохранены в {filename}")
    print("❗ График не строится, чтобы избежать перегрузки системы.")

if __name__ == "__main__":
    save_and_show(
        symbol="SUIUSDC",
        interval="15m",
        start_date_str="2018-01-01",
        filename="../../../../data/binance/SUIUSDC/15m/2024_01_19-now.csv"
    )
