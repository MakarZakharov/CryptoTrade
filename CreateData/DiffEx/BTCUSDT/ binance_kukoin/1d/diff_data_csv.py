import requests
import pandas as pd
from datetime import datetime
import time


def get_binance_klines(symbol, interval, start_str, end_str=None):
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
        time.sleep(0.2)

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "quote_volume": float
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
    return df


def get_kucoin_klines(symbol="BTC-USDT", interval="1day", start_str="2018-01-01", end_str=None):
    url = "https://api.kucoin.com/api/v1/market/candles"
    start_ts = int(pd.Timestamp(start_str).timestamp())
    end_ts = int(pd.Timestamp(end_str).timestamp()) if end_str else int(time.time())

    all_klines = []
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "type": interval,
            "startAt": start_ts,
            "endAt": min(end_ts, start_ts + 86400 * 300)
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        all_klines += data[::-1]
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


def human_format(value):
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} млрд"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f} млн"
    else:
        return f"{value:,.2f}"


def compare_dataframes_avg(df1, df2):
    df_merged = pd.merge(df1, df2, on="timestamp", suffixes=("_binance", "_kucoin"))
    columns = ["open", "high", "low", "close", "volume", "quote_volume"]

    print("\n📊 Сравнение средних значений Binance vs KuCoin:\n")

    for col in columns:
        avg_binance = df_merged[f"{col}_binance"].mean()
        avg_kucoin = df_merged[f"{col}_kucoin"].mean()
        diff = abs(avg_binance - avg_kucoin)
        percent = 100 * diff / avg_binance if avg_binance != 0 else float("nan")

        print(f"🔸 {col.upper()}:")
        print(f"   - Binance: {human_format(avg_binance)}")
        print(f"   - KuCoin:  {human_format(avg_kucoin)}")
        print(f"   ➤ Разница: {human_format(diff)} ({percent:.2f}%)\n")


def get_orderbook_binance(symbol="BTCUSDT", limit=5):
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    bids = [float(bid[0]) for bid in data["bids"]]
    asks = [float(ask[0]) for ask in data["asks"]]
    return bids, asks


def get_orderbook_kucoin(symbol="BTC-USDT", limit=5):
    url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_100"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()["data"]
    bids = [float(bid[0]) for bid in data["bids"][:limit]]
    asks = [float(ask[0]) for ask in data["asks"][:limit]]
    return bids, asks


def compare_orderbooks():
    print("\n📈 Сравнение биржевого стакана (топ-5 заявок):\n")
    bin_bids, bin_asks = get_orderbook_binance()
    kuo_bids, kuo_asks = get_orderbook_kucoin()

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    avg_bid_bin = avg(bin_bids)
    avg_ask_bin = avg(bin_asks)
    avg_bid_kuo = avg(kuo_bids)
    avg_ask_kuo = avg(kuo_asks)

    def show(label, b_val, k_val):
        diff = abs(b_val - k_val)
        percent = 100 * diff / b_val if b_val else float("nan")
        print(f"🔸 {label}:")
        print(f"   - Binance: {b_val:,.2f}")
        print(f"   - KuCoin:  {k_val:,.2f}")
        print(f"   ➤ Разница: {diff:,.2f} ({percent:.2f}%)\n")

    show("Средняя цена BID (покупка)", avg_bid_bin, avg_bid_kuo)
    show("Средняя цена ASK (продажа)", avg_ask_bin, avg_ask_kuo)


if __name__ == "__main__":
    print("🔁 Загружаем данные с Binance и KuCoin...")
    binance_df = get_binance_klines("BTCUSDT", "1d", "2025-06-01")
    kucoin_df = get_kucoin_klines("BTC-USDT", "1day", "2025-06-01")

    compare_dataframes_avg(binance_df, kucoin_df)
    compare_orderbooks()
