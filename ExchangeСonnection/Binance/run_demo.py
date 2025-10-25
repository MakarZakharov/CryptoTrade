#!/usr/bin/env python3
"""
run_demo.py

Запускає основний скрипт у DEMO-режимі (без реального API).
Якщо --target не вказано — автоматично шукає головний файл у цій директорії.

Використання:
    python run_demo.py
    python run_demo.py --target main.py --print-balances
"""

import sys
import os
import json
import random
import argparse
import runpy
import types
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# MockBinanceClient
# ---------------------------
class MockBinanceClient:
    def __init__(self, demo_balances=None, demo_prices=None, taker_fee=0.001):
        self.balances = demo_balances or {'USDC': {'free': 5000.0, 'locked': 0.0}}
        self.prices = demo_prices or {}
        self.taker_fee = taker_fee
        self.symbols_info = {
            pair: {
                'symbol': pair,
                'filters': [
                    {'filterType': 'LOT_SIZE', 'stepSize': '0.000001'},
                    {'filterType': 'MIN_NOTIONAL', 'minNotional': '10'},
                ],
            }
            for pair in self.prices
        }

    def get_account(self):
        return {
            'balances': [{'asset': a, 'free': str(v['free']), 'locked': str(v['locked'])}
                         for a, v in self.balances.items()]
        }

    def get_asset_balance(self, asset):
        v = self.balances.get(asset, {'free': 0.0, 'locked': 0.0})
        return {'asset': asset, 'free': str(v['free']), 'locked': str(v['locked'])}

    def get_symbol_ticker(self, symbol):
        if symbol not in self.prices:
            raise Exception(f"[Mock] Symbol {symbol} not found.")
        return {'symbol': symbol, 'price': str(self.prices[symbol])}

    def get_symbol_info(self, symbol):
        return self.symbols_info.get(symbol, {
            'symbol': symbol,
            'filters': [{'filterType': 'LOT_SIZE', 'stepSize': '0.000001'}],
        })

    def order_market_buy(self, symbol, quantity):
        price = self.prices[symbol]
        cost = price * quantity
        fee = cost * self.taker_fee
        total_cost = cost + fee

        base = symbol.replace('USDC', '')
        if self.balances['USDC']['free'] < total_cost:
            raise Exception(f"[Mock] Not enough USDC to buy {symbol}. Need {total_cost}")

        self.balances['USDC']['free'] -= total_cost
        self.balances.setdefault(base, {'free': 0, 'locked': 0})
        self.balances[base]['free'] += quantity

        return {
            'orderId': random.randint(100000, 999999),
            'executedQty': str(quantity),
            'cummulativeQuoteQty': str(cost),  # ✅ додано
        }

    def order_market_sell(self, symbol, quantity):
        price = self.prices[symbol]
        proceeds = price * quantity
        fee = proceeds * self.taker_fee
        net = proceeds - fee

        base = symbol.replace('USDC', '')
        if self.balances.get(base, {'free': 0})['free'] < quantity:
            raise Exception(f"[Mock] Not enough {base} to sell.")

        self.balances[base]['free'] -= quantity
        self.balances['USDC']['free'] += net

        return {
            'orderId': random.randint(100000, 999999),
            'executedQty': str(quantity),
            'cummulativeQuoteQty': str(proceeds),  # ✅ додано
        }


# ---------------------------
# Підміна binance.client.Client
# ---------------------------
def inject_mock_binance_client(mock_client_cls):
    binance_mod = types.ModuleType("binance")
    client_mod = types.ModuleType("binance.client")
    client_mod.Client = mock_client_cls

    exceptions_mod = types.ModuleType("binance.exceptions")
    exceptions_mod.BinanceAPIException = Exception
    exceptions_mod.BinanceOrderException = Exception

    sys.modules["binance"] = binance_mod
    sys.modules["binance.client"] = client_mod
    sys.modules["binance.exceptions"] = exceptions_mod


# ---------------------------
# Основна логіка
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", help="Шлях до основного .py файлу для запуску")
    p.add_argument("--demo-balances", help="JSON з демо-балансами")
    p.add_argument("--demo-prices", help="JSON з демо-цінами")
    p.add_argument("--taker-fee", type=float, default=0.001, help="Комісія (0.001 = 0.1%)")
    p.add_argument("--dry-run", action="store_true", help="Тільки симуляція без змін")
    p.add_argument("--print-balances", action="store_true", help="Показати баланси після виконання")
    return p.parse_args()


def find_default_target():
    """Знаходимо .py файл у поточній папці, крім run_demo.py"""
    here = Path(__file__).parent
    candidates = [f for f in here.glob("*.py") if f.name != "run_demo.py"]
    if not candidates:
        print("[run_demo] ❌ Не знайдено жодного .py файлу для запуску.")
        sys.exit(2)
    # якщо є тільки один — беремо його
    if len(candidates) == 1:
        return candidates[0]
    # якщо кілька — вибираємо найновіший за часом зміни
    return max(candidates, key=lambda f: f.stat().st_mtime)


def load_json(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


from binance.client import Client  # використовується тільки при першому запуску

DEMO_FILE = "demo_state.json"

def load_or_create_demo_state(taker_fee=0.001):
    """Створює demo_state.json при першому запуску, або зчитує існуючий"""
    if os.path.exists(DEMO_FILE):
        with open(DEMO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[run_demo] ✅ Завантажено демо-стан із {DEMO_FILE}")
        return data

    print("[run_demo] 🟡 demo_state.json не знайдено — отримую реальні баланси з Binance...")

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Не знайдено API ключів Binance. Додай їх у середовище або .env")

    client = Client(api_key, api_secret)
    balances = {}
    account = client.get_account()
    for b in account["balances"]:
        free = float(b["free"])
        locked = float(b["locked"])
        if free + locked > 0:
            balances[b["asset"]] = {"free": free, "locked": locked}

    # Отримуємо поточні ціни для найпопулярніших пар (щоб потім не питати API)
    prices = {}
    tickers = client.get_all_tickers()
    for t in tickers:
        symbol = t["symbol"]
        if symbol.endswith("USDC") or symbol.endswith("USDT"):
            prices[symbol] = float(t["price"])

    data = {"balances": balances, "prices": prices, "taker_fee": taker_fee}
    with open(DEMO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[run_demo] 💾 Збережено демо-стан у {DEMO_FILE}")

    return data


def main():
    args = parse_args()
    target = args.target or find_default_target()

    # Якщо є або треба створити demo_state.json
    demo_state = load_or_create_demo_state(args.taker_fee)
    demo_balances = demo_state["balances"]
    demo_prices = demo_state["prices"]
    taker_fee = demo_state.get("taker_fee", args.taker_fee)

    # створюємо mock Binance API
    class _FactoryClient:
        def __init__(self, *a, **k):
            self._mock = MockBinanceClient(
                demo_balances=demo_balances,
                demo_prices=demo_prices,
                taker_fee=taker_fee
            )
        def __getattr__(self, item):
            return getattr(self._mock, item)

    inject_mock_binance_client(_FactoryClient)
    os.environ["DEMO_MODE"] = "1"
    if args.dry_run:
        os.environ["DEMO_DRY_RUN"] = "1"

    print(f"[run_demo] ▶ Запуск демо для: {target}")
    runpy.run_path(str(target), run_name="__main__")

    # Після завершення — зберігаємо оновлений демо-стан
    with open(DEMO_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "balances": demo_balances,
            "prices": demo_prices,
            "taker_fee": taker_fee
        }, f, indent=2, ensure_ascii=False)
    print(f"[run_demo] 💾 Оновлено {DEMO_FILE} після симуляції")

    if args.print_balances:
        print("\n=== DEMO BALANCES ===")
        for asset, vals in demo_balances.items():
            print(f"{asset}: free={vals['free']}, locked={vals['locked']}")
        print("=====================")


    # створюємо фабрику mock-клієнта
    class _FactoryClient:
        def __init__(self, *a, **k):
            self._mock = MockBinanceClient(
                demo_balances=demo_balances,
                demo_prices=demo_prices,
                taker_fee=args.taker_fee
            )
        def __getattr__(self, item):
            return getattr(self._mock, item)

    inject_mock_binance_client(_FactoryClient)
    os.environ["DEMO_MODE"] = "1"
    if args.dry_run:
        os.environ["DEMO_DRY_RUN"] = "1"

    runpy.run_path(str(target), run_name="__main__")

    if args.print_balances:
        print("\n=== DEMO BALANCES ===")
        for asset, vals in demo_balances.items():
            print(f"{asset}: free={vals['free']}, locked={vals['locked']}")
        print("=====================")


if __name__ == "__main__":
    main()
