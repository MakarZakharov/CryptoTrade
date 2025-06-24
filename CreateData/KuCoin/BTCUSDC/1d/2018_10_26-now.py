import requests
import pandas as pd
import time
from datetime import datetime
import os
import mplfinance as mpf

def get_kucoin_klines(symbol="BTC-USDC", interval="1day", start_str="2018-10-26", end_str=None):
    url = "https://api.kucoin.com/api/v1/market/candles"
    start_ts = int(pd.Timestamp(start_str).timestamp())
    end_ts = int(pd.Timestamp(end_str).timestamp()) if end_str else int(time.time())
    all_klines = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "type": interval,
            "startAt": start_ts,
            "endAt": min(end_ts, start_ts + 86400 * 300)  # 300 дней
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        all_klines += data[::-1]  # обратный порядок
        start_ts = int(data[0][0]) + 86400
        time.sleep(0.2)

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "close", "high", "low", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "turnover": float
    })
    df.rename(columns={"turnover": "quote_volume"}, inplace=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
    return df

def plot_if_possible(df, symbol, interval):
    if df.shape[0] > 500:
        df_plot = df.copy()
        df_plot.set_index("timestamp", inplace=True)
        df_plot = df_plot[["open", "high", "low", "close", "volume"]]
        title = f"{symbol} KuCoin {interval} Candles"
        mpf.plot(df_plot, type="candle", style="charles", volume=True, title=title, ylabel="Цена", ylabel_lower="Объём")
    else:
        print("Недостаточно данных для построения графика.")

def save_kucoin():
    df = get_kucoin_klines()
    if df.shape[0] < 100 or df.isnull().values.any():
        print("Проблема с полученными данными от KuCoin.")
        return
    path = "../../../../data/KuCoin/BTCUSDC/1d/2018_10_26-now.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"KuCoin CSV сохранён: {path}")
    plot_if_possible(df, "BTCUSDC", "1d")

if __name__ == "__main__":
    save_kucoin()
