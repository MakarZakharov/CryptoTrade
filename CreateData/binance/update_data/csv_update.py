import os
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime
import time

START_DATE = "2018-01-01"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'binance'))
API_URL = "https://api.binance.com/api/v3/klines"

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
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]

async def get_klines(session, symbol, interval, start_str, end_str=None):
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

        async with session.get(API_URL, params=params) as response:
            data = await response.json()
            if not data or isinstance(data, dict) and "code" in data:
                break

            all_klines.extend(data)
            start_ts = data[-1][6] + 1

            if len(data) < limit:
                break

    return all_klines

async def update_csv(session, symbol, interval, file_path, end_date):
    print(f"Обновление: {symbol} [{interval}] -> {file_path}")
    for attempt in range(3):
        try:
            klines = await get_klines(session, symbol, interval, START_DATE, end_date)
            df = klines_to_dataframe(klines)

            if df.shape[0] > 50 and not df.isnull().values.any():
                df.to_csv(file_path, index=False)
                print(f"✔ CSV обновлён: {file_path}")
                return
            else:
                print(f"✖ Данные некорректны для {symbol} [{interval}], повтор...\n")
                await asyncio.sleep(2)
        except Exception as e:
            print(f"Ошибка при загрузке {symbol} [{interval}]: {e}")
            await asyncio.sleep(2)

async def update_all_csvs_async():
    end_date = datetime.today().strftime("%Y-%m-%d")
    tasks = []

    async with aiohttp.ClientSession() as session:
        for symbol in os.listdir(DATA_DIR):
            symbol_path = os.path.join(DATA_DIR, symbol)
            if not os.path.isdir(symbol_path):
                continue

            for interval in os.listdir(symbol_path):
                interval_path = os.path.join(symbol_path, interval)
                if not os.path.isdir(interval_path):
                    continue

                for file in os.listdir(interval_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(interval_path, file)
                        tasks.append(update_csv(session, symbol, interval, file_path, end_date))

        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(update_all_csvs_async())