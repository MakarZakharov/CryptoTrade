import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import logging
from decimal import Decimal
from typing import Dict, List, Optional
import time

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Завантаження змінних з .env файлу
load_dotenv()


class SimpleBinanceTrader:
    """
    Простий клас для роботи з Binance API та конвертації токенів
    """

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        """
        Ініціалізація клієнта Binance
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.testnet = testnet

        # Простий тестовий баланс
        self.test_balance = {
            'BTC': 0.5,
            'ETH': 10.0,
            'USDT': 10000.0,
            'BNB': 50.0,
            'ADA': 5000.0,
            'DOT': 100.0,
            'SOL': 25.0
        }

        # Простий курс валют для тестування (відносно USDT)
        self.test_rates = {
            'BTC': 104000.0,    # 1 BTC = 104000 USDT
            'ETH': 2500.0,      # 1 ETH = 2500 USDT
            'USDT': 1.0,        # 1 USDT = 1 USDT
            'BNB': 600.0,       # 1 BNB = 600 USDT
            'ADA': 0.7,         # 1 ADA = 0.7 USDT
            'DOT': 8.0,         # 1 DOT = 8 USDT
            'SOL': 200.0,       # 1 SOL = 200 USDT
            'TON': 5.0,         # 1 TON = 5 USDT
            'MATIC': 1.1,       # 1 MATIC = 1.1 USDT
            'LINK': 25.0        # 1 LINK = 25 USDT
        }

        # Спроба підключення до API
        self.client = None
        if self.api_key and self.api_secret:
            try:
                self.client = Client(self.api_key, self.api_secret, testnet=testnet)
                self.client.ping()
                print(f"✅ Підключено до {'тестової' if testnet else 'реальної'} мережі Binance")
            except Exception as e:
                print(f"❌ Помилка підключення: {e}")
                print("🧪 Використовується тестовий режим")
        else:
            print("⚠️ API ключі не знайдені. Використовується тестовий режим")

        # Показуємо доступний баланс
        if not self.client:
            print("💰 Доступний тестовий баланс:")
            for asset, amount in self.test_balance.items():
                if amount > 0:
                    print(f"  {asset}: {amount}")

    def get_usd_value(self, asset: str, amount: float) -> float:
        """Отримати USD вартість токена"""
        return amount * self.test_rates.get(asset.upper(), 1.0)

    def get_exchange_rate(self, from_asset: str, to_asset: str) -> float:
        """
        Отримати курс конвертації між двома токенами
        """
        from_asset = from_asset.upper()
        to_asset = to_asset.upper()

        if from_asset == to_asset:
            return 1.0

        # Спочатку спробуємо отримати реальний курс через API
        if self.client:
            try:
                # Спробуємо прямий курс
                direct_symbol = f"{from_asset}{to_asset}"
                public_client = Client()
                ticker = public_client.get_symbol_ticker(symbol=direct_symbol)
                rate = float(ticker['price'])
                # Покращене форматування курсу
                if rate < 0.001:
                    print(f"📈 Реальний курс {from_asset}/{to_asset}: {rate:.10f}")
                else:
                    print(f"📈 Реальний курс {from_asset}/{to_asset}: {rate:,.6f}")
                return rate
            except:
                try:
                    # Спробуємо обернений курс
                    reverse_symbol = f"{to_asset}{from_asset}"
                    ticker = public_client.get_symbol_ticker(symbol=reverse_symbol)
                    rate = 1.0 / float(ticker['price'])
                    # Покращене форматування курсу
                    if rate < 0.001:
                        print(f"📈 Реальний курс {from_asset}/{to_asset}: {rate:.10f}")
                    else:
                        print(f"📈 Реальний курс {from_asset}/{to_asset}: {rate:,.6f}")
                    return rate
                except:
                    pass

        # Використовуємо тестовий курс через USDT
        from_rate = self.test_rates.get(from_asset, 1.0)
        to_rate = self.test_rates.get(to_asset, 1.0)

        # Розраховуємо курс: from_asset -> USDT -> to_asset
        exchange_rate = from_rate / to_rate
        print(f"🧪 Тестовий курс {from_asset}/{to_asset}: {exchange_rate:,.6f}")
        print(f"   ({from_asset}: ${from_rate:,.2f} → {to_asset}: ${to_rate:,.2f})")
        return exchange_rate

    def get_balance(self, asset: str) -> float:
        """
        Отримати баланс токена (з тестового балансу або API)
        """
        if not self.client:
            return self.test_balance.get(asset.upper(), 0.0)

        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset.upper():
                    return float(balance['free'])
            return 0.0
        except Exception:
            # При помилці API використовуємо тестовий баланс
            return self.test_balance.get(asset.upper(), 0.0)

    def add_test_balance(self, asset: str, amount: float):
        """Додати тестовий баланс"""
        asset = asset.upper()
        if asset not in self.test_balance:
            self.test_balance[asset] = 0
        self.test_balance[asset] += amount
        print(f"💰 Додано {amount} {asset}. Новий баланс: {self.test_balance[asset]}")

    def show_test_balance(self):
        """Показати баланс"""
        print("\n💰 === БАЛАНС ===")
        total_usd = 0
        for asset, amount in sorted(self.test_balance.items()):
            if amount > 0:
                usd_value = self.get_usd_value(asset, amount)
                total_usd += usd_value
                print(f"  {asset}: {amount:,.8f} (~${usd_value:,.2f})")
        print(f"💵 Загальна вартість: ${total_usd:,.2f}")
        print("=================\n")

    def convert_tokens(self, from_asset: str, to_asset: str, amount, commission_rate: float = 0.001):
        """
        Конвертувати токени з комісією 0.1%
        """
        from_asset = from_asset.upper()
        to_asset = to_asset.upper()

        # 1. Перевірка балансу
        available_balance = self.get_balance(from_asset)
        if available_balance <= 0:
            print(f"❌ Немає балансу {from_asset}")
            return False

        # 2. Визначення кількості для конвертації
        if str(amount).lower() == 'max':
            convert_amount = available_balance
        else:
            convert_amount = float(amount)
            if convert_amount > available_balance:
                print(f"❌ Недостатньо коштів. Доступно: {available_balance}")
                return False

        # 3. Отримання курсу (один раз)
        exchange_rate = self.get_exchange_rate(from_asset, to_asset)

        # 4. Розрахунки з високою точністю
        commission = round(convert_amount * commission_rate, 8)
        final_amount = round(convert_amount - commission, 8)
        expected_receive = round(final_amount * exchange_rate, 8)

        # 5. Розрахунок вартості комісії в доларах
        commission_usd = self.get_usd_value(from_asset, commission)

        # 6. Відображення інформації про конвертацію
        print(f"\n💱 === КОНВЕРТАЦІЯ ===")
        print(f"📊 Сума: {convert_amount:,.8f} {from_asset}")
        print(f"💸 Комісія (0.1%): {commission:,.8f} {from_asset} (${commission_usd:.2f})")

        # Покращене відображення курсу
        if exchange_rate < 0.001:
            print(f"📈 Курс: 1 {from_asset} = {exchange_rate:.10f} {to_asset}")
        else:
            print(f"📈 Курс: 1 {from_asset} = {exchange_rate:,.6f} {to_asset}")

        print(f"🎯 Отримаєте: {expected_receive:,.8f} {to_asset}")
        print(f"========================")

        # 7. Оновлення балансу
        self.test_balance[from_asset] = round(self.test_balance.get(from_asset, 0) - convert_amount, 8)
        self.test_balance[to_asset] = round(self.test_balance.get(to_asset, 0) + expected_receive, 8)

        # 8. Результат конвертації
        print("✅ КОНВЕРТАЦІЯ ВИКОНАНА!")
        print(f"📉 {from_asset}: {self.test_balance[from_asset]:,.8f}")
        print(f"📈 {to_asset}: {self.test_balance[to_asset]:,.8f}")

        # 9. USD еквівалент нових балансів
        from_usd = self.get_usd_value(from_asset, self.test_balance[from_asset])
        to_usd = self.get_usd_value(to_asset, self.test_balance[to_asset])

        print(f"💰 USD еквівалент:")
        print(f"   {from_asset}: ~${from_usd:,.2f}")
        print(f"   {to_asset}: ~${to_usd:,.2f}")

        # 10. Підсумок комісії
        print(f"💸 Загальна комісія: ${commission_usd:.2f}")

        return True


def interactive_converter():
    """
    Простий інтерактивний конвертер
    """
    print("🔄 === КОНВЕРТЕР ТОКЕНІВ ===")
    print("Команди:")
    print("  convert - конвертувати токени")
    print("  balance - показати баланс")
    print("  add     - додати тестові токени")
    print("  exit    - вийти\n")

    trader = SimpleBinanceTrader(testnet=True)

    while True:
        try:
            print("\n" + "="*50)
            command = input("👉 Команда: ").strip().lower()

            if command == 'exit':
                print("👋 До побачення!")
                break
            elif command == 'balance':
                trader.show_test_balance()

            elif command == 'add':
                asset = input("Токен: ").strip().upper()
                try:
                    amount = float(input("Кількість: ").strip())
                    trader.add_test_balance(asset, amount)
                except ValueError:
                    print("❌ Неправильна кількість")

            elif command == 'convert':
                print("\n🔄 Конвертація")

                # Введення токенів
                from_token = input("З: ").strip().upper()
                if not from_token:
                    continue

                to_token = input("В: ").strip().upper()
                if not to_token:
                    continue

                if from_token == to_token:
                    print("❌ Однакові токени")
                    continue

                # Перевірка балансу
                balance = trader.get_balance(from_token)
                balance_usd = trader.get_usd_value(from_token, balance)
                print(f"💰 Баланс {from_token}: {balance:,.8f} (~${balance_usd:,.2f})")

                if balance <= 0:
                    print(f"❌ Немає {from_token}")
                    continue

                # Введення кількості
                amount = input(f"Кількість (або 'max'): ").strip()
                if not amount:
                    continue

                # Попередній перегляд
                print(f"\n📋 Попередній перегляд:")
                rate = trader.get_exchange_rate(from_token, to_token)

                if rate < 0.001:
                    print(f"📈 Курс {from_token}/{to_token}: {rate:.10f}")
                else:
                    print(f"📈 Курс {from_token}/{to_token}: {rate:,.6f}")

                # Розрахунок попереднього результату
                if amount.lower() == 'max':
                    preview_amount = balance
                else:
                    preview_amount = float(amount)

                preview_commission = preview_amount * 0.001
                preview_commission_usd = trader.get_usd_value(from_token, preview_commission)
                preview_final = preview_amount - preview_commission
                preview_receive = preview_final * rate

                print(f"💡 Отримаєте: {preview_receive:,.8f} {to_token}")
                print(f"💸 Комісія: {preview_commission:,.8f} {from_token} (${preview_commission_usd:.2f})")

                # Підтвердження
                confirm = input(f"\nПідтвердити? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', 'так']:
                    trader.convert_tokens(from_token, to_token, amount)
                    print("🎉 Готово!")
                else:
                    print("❌ Скасовано")

            else:
                print("❌ Невідома команда!")
                print("Доступні: convert, balance, add, exit")

        except KeyboardInterrupt:
            print("\n👋 Вихід")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    interactive_converter()
