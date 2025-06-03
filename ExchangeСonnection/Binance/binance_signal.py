import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from decimal import Decimal
import time

logging.basicConfig(level=logging.INFO)
load_dotenv()


class SimpleBinanceTrader:
    """Спрощений клас для конвертації токенів на Binance"""

    def __init__(self, testnet: bool = True):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.testnet = testnet
        self.client = None

        # Додати кеш для фільтрів
        self.exchange_info_cache = None
        self.cache_time = 0

        # Тестовий баланс та курси
        self.test_balance = {
            'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 'BNB': 50.0,
            'ADA': 5000.0, 'DOT': 100.0, 'SOL': 25.0
        }
        self.rates = {
            'USDT': 1.0, 'BTC': 104000.0, 'ETH': 2500.0, 'BNB': 600.0,
            'ADA': 0.7, 'DOT': 8.0, 'SOL': 200.0
        }

        if self.api_key and self.api_secret and not testnet:
            try:
                self.client = Client(self.api_key, self.api_secret)
                self.client.ping()
                self.update_exchange_info()  # Додати цей виклик
                self.update_rates()
                print("✅ Підключено до Binance")
            except Exception as e:
                print(f"❌ Помилка підключення: {e}")
                raise
        else:
            print("🧪 Тестовий режим")

    def update_exchange_info(self):
        """Оновити інформацію про біржу"""
        if not self.client:
            return

        try:
            self.exchange_info_cache = self.client.get_exchange_info()
            self.cache_time = time.time()
            print("✅ Оновлено інформацію про біржу")
        except Exception as e:
            print(f"❌ Помилка оновлення інформації: {e}")

    def update_rates(self):
        """Оновити курси валют"""
        if not self.client:
            return
        try:
            tickers = self.client.get_all_tickers()
            new_rates = {'USDT': 1.0}
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith('USDT') and len(symbol) > 4:
                    asset = symbol[:-4]
                    new_rates[asset] = float(ticker['price'])
            self.rates = new_rates
            print(f"✅ Оновлено {len(new_rates)} курсів")
        except Exception as e:
            print(f"❌ Помилка оновлення курсів: {e}")

    def get_symbol_filters(self, symbol: str) -> dict:
        """Отримати фільтри символу"""
        default_filters = {
            'stepSize': '1', 'minQty': '1', 'maxQty': '99999999',
            'minNotional': '1', 'tickSize': '0.01'
        }

        if not self.client or not self.exchange_info_cache:
            return default_filters

        try:
            for s in self.exchange_info_cache['symbols']:
                if s['symbol'] == symbol:
                    filters = {'tickSize': '0.01'}

                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            filters.update({
                                'stepSize': f['stepSize'],
                                'minQty': f['minQty'],
                                'maxQty': f['maxQty']
                            })
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            filters['minNotional'] = f['minNotional']
                        elif f['filterType'] == 'PRICE_FILTER':
                            filters['tickSize'] = f['tickSize']
                    return filters

            return default_filters
        except Exception as e:
            print(f"❌ Помилка отримання фільтрів: {e}")
            return default_filters

    def format_quantity(self, quantity: float, step_size: str) -> str:
        """Форматувати кількість згідно з step_size"""
        try:
            step = Decimal(step_size)
            qty = Decimal(str(quantity))

            if step == 0:
                return f"{quantity:.8f}".rstrip('0').rstrip('.')

            # Округлюємо вниз до step_size
            rounded = (qty // step) * step
            precision = abs(step.as_tuple().exponent)
            formatted = f"{rounded:.{precision}f}".rstrip('0').rstrip('.')

            return formatted if formatted else "0"
        except Exception:
            return f"{quantity:.8f}".rstrip('0').rstrip('.')

    def safe_market_order(self, symbol: str, side: str, quantity: float, is_quote_qty: bool = False, force_max: bool = False):
        """Виконати ринковий ордер"""
        try:
            filters = self.get_symbol_filters(symbol)

            if is_quote_qty:
                min_notional = float(filters.get('minNotional', '1'))
                if quantity < min_notional:
                    print(f"❌ Сума менша за мінімальну: {min_notional}")
                    return None
                formatted_qty = f"{quantity:.8f}".rstrip('0').rstrip('.')
            else:
                step_size = filters.get('stepSize', '1')
                min_qty = float(filters.get('minQty', '0'))

                if force_max and side == 'sell':
                    # Для max sell спочатку спробуємо округлити до stepSize
                    formatted_qty = self.format_quantity(quantity, step_size)
                    actual_qty = float(formatted_qty) if formatted_qty != '0' else 0

                    print(f"🔧 Округлено до stepSize: {formatted_qty}")

                    if actual_qty < min_qty:
                        print(f"🔄 Округлена кількість {actual_qty} менша за min {min_qty}")
                        # Спробуємо продати залишок через quoteOrderQty
                        return self._sell_dust_via_quote(symbol, quantity)

                    # Спробуємо виконати ордер з округленою кількістю
                    try:
                        test_order = self.client.order_market_sell(symbol=symbol, quantity=formatted_qty)
                        return test_order
                    except BinanceAPIException as e:
                        if "LOT_SIZE" in str(e):
                            print(f"🔄 LOT_SIZE помилка з округленою кількістю, спроба через quoteOrderQty")
                            return self._sell_dust_via_quote(symbol, quantity)
                        else:
                            raise e
                else:
                    formatted_qty = self.format_quantity(quantity, step_size)
                    actual_qty = float(formatted_qty) if formatted_qty != '0' else 0

                    if actual_qty < min_qty:
                        print(f"❌ Кількість менша за мінімальну: {min_qty}")
                        return None
                    print(f"🔧 Форматована кількість: {formatted_qty}")

            # Виконати ордер для звичайних випадків
            if side == 'sell':
                return self.client.order_market_sell(symbol=symbol, quantity=formatted_qty)
            else:
                if is_quote_qty:
                    return self.client.order_market_buy(symbol=symbol, quoteOrderQty=formatted_qty)
                else:
                    return self.client.order_market_buy(symbol=symbol, quantity=formatted_qty)

        except BinanceAPIException as e:
            print(f"❌ Помилка Binance API: {e}")
            return None
        except Exception as e:
            print(f"❌ Помилка ордера: {e}")
            return None

    def _sell_dust_via_quote(self, symbol: str, dust_quantity: float):
        """Продати через quoteOrderQty для максимальної конвертації"""
        try:
            # Отримати поточну ціну
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # Розрахувати вартість в quote валюті
            quote_value = dust_quantity * current_price

            filters = self.get_symbol_filters(symbol)
            min_notional = float(filters.get('minNotional', '1'))

            if quote_value < min_notional:
                print(f"💸 Вартість ${quote_value:.8f} менша за мінімальну ${min_notional}")
                return None

            # Спроба продати весь баланс через quoteOrderQty
            print(f"🔄 Максимальна конвертація: {dust_quantity} токенів за ~${quote_value:.6f}")

            # Використовуємо 99.95% від розрахункової вартості для уникнення округлень
            safe_quote_value = quote_value * 0.9995
            formatted_quote = f"{safe_quote_value:.8f}".rstrip('0').rstrip('.')

            print(f"🎯 Використано quoteOrderQty: {formatted_quote}")
            order = self.client.order_market_sell(symbol=symbol, quoteOrderQty=formatted_quote)
            return order

        except Exception as e:
            print(f"❌ Не вдалося продати через quoteOrderQty: {e}")
            return None

    def clear_dust(self, asset: str = None):
        """Очистити пил з рахунку"""
        if self.testnet:
            print("❌ Тільки для реального режиму")
            return

        try:
            account = self.client.get_account()
            dust_assets = []

            for balance in account['balances']:
                asset_name = balance['asset']
                free_balance = float(balance['free'])

                if free_balance > 0 and (asset is None or asset_name == asset.upper()):
                    # Перевірити чи є це пил
                    if asset_name != 'USDT':  # Не чистимо USDT
                        usdt_symbol = f"{asset_name}USDT"
                        try:
                            filters = self.get_symbol_filters(usdt_symbol)
                            min_qty = float(filters.get('minQty', '0'))

                            if 0 < free_balance < min_qty:
                                dust_assets.append((asset_name, free_balance))
                        except:
                            continue

            if not dust_assets:
                print("✅ Пилу не знайдено")
                return

            print(f"🧹 Знайдено пил в {len(dust_assets)} активах:")
            for asset_name, amount in dust_assets:
                print(f"  {asset_name}: {amount:.8f}")

            # Спроба конвертувати пил в BNB (якщо доступно)
            try:
                dust_transfer = self.client.transfer_dust(asset=[asset[0] for asset in dust_assets])
                if dust_transfer:
                    print("✅ Пил успішно конвертовано в BNB")
                    return True
            except Exception as e:
                print(f"⚠️ Автоматична конвертація пилу недоступна: {e}")

            # Альтернативний підхід - спроба продати через quoteOrderQty
            for asset_name, amount in dust_assets:
                symbol = f"{asset_name}USDT"
                print(f"🔄 Спроба очистити {asset_name}...")
                order = self._sell_dust_via_quote(symbol, amount)
                if order:
                    print(f"✅ Очищено {asset_name}, ID: {order['orderId']}")
                else:
                    print(f"❌ Не вдалося очистити {asset_name}")

        except Exception as e:
            print(f"❌ Помилка очищення пилу: {e}")

    def get_balance(self, asset: str) -> float:
        """Отримати баланс"""
        if self.testnet or not self.client:
            return self.test_balance.get(asset.upper(), 0.0)

        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset.upper():
                    return float(balance['free'])
            return 0.0
        except Exception:
            return 0.0

    def get_rate(self, from_asset: str, to_asset: str) -> float:
        """Курс конвертації"""
        from_asset, to_asset = from_asset.upper(), to_asset.upper()
        if from_asset == to_asset:
            return 1.0

        from_rate = self.rates.get(from_asset, 1.0)
        to_rate = self.rates.get(to_asset, 1.0)
        return from_rate / to_rate

    def get_trading_symbol(self, from_asset: str, to_asset: str):
        """Знайти торговий символ"""
        from_asset, to_asset = from_asset.upper(), to_asset.upper()

        if not self.client:
            return f"{from_asset}{to_asset}", 'buy'

        try:
            exchange_info = self.client.get_exchange_info()
            available_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}

            # Спробувати прямі пари
            direct_symbol = f"{from_asset}{to_asset}"
            reverse_symbol = f"{to_asset}{from_asset}"

            if direct_symbol in available_symbols:
                return direct_symbol, 'sell'
            elif reverse_symbol in available_symbols:
                return reverse_symbol, 'buy'
            else:
                return None, None  # Потрібна конвертація через USDT
        except Exception:
            return f"{from_asset}{to_asset}", 'buy'

    def execute_order(self, symbol: str, side: str, amount: float, use_quote: bool = False):
        """Виконати ордер"""
        try:
            if side == 'sell':
                order = self.client.order_market_sell(symbol=symbol, quantity=f"{amount:.8f}")
            else:
                if use_quote:
                    order = self.client.order_market_buy(symbol=symbol, quoteOrderQty=f"{amount:.8f}")
                else:
                    order = self.client.order_market_buy(symbol=symbol, quantity=f"{amount:.8f}")
            return order
        except BinanceAPIException as e:
            print(f"❌ Помилка API: {e}")
            return None
        except Exception as e:
            print(f"❌ Помилка ордера: {e}")
            return None

    def convert(self, from_asset: str, to_asset: str, amount):
        """Конвертувати токени"""
        from_asset, to_asset = from_asset.upper(), to_asset.upper()

        balance = self.get_balance(from_asset)
        if balance <= 0:
            print(f"❌ Немає {from_asset}")
            return False

        convert_amount = balance if str(amount).lower() == 'max' else float(amount)
        if convert_amount > balance:
            print(f"❌ Недостатньо коштів")
            return False

        is_max_conversion = str(amount).lower() == 'max'

        if self.testnet:
            # Тестова конвертація з симуляцією комісії
            commission = convert_amount * 0.001
            final_amount = convert_amount - commission
            rate = self.get_rate(from_asset, to_asset)
            receive_amount = final_amount * rate

            print(f"\n💱 Тестова конвертація:")
            print(f"📊 Сума: {convert_amount:,.8f} {from_asset}")
            print(f"💸 Комісія: {commission:,.8f} {from_asset}")
            print(f"🎯 Отримаєте: {receive_amount:,.8f} {to_asset}")

            self.test_balance[from_asset] -= convert_amount
            self.test_balance[to_asset] = self.test_balance.get(to_asset, 0) + receive_amount
            print("✅ Тестова конвертація виконана!")
            return True
        else:
            # Реальна конвертація
            rate = self.get_rate(from_asset, to_asset)
            estimated_receive = convert_amount * rate

            print(f"\n💱 Реальна конвертація:")
            print(f"📊 Сума: {convert_amount:,.8f} {from_asset}")
            print(f"🎯 Очікується: {estimated_receive:,.8f} {to_asset}")

            symbol, side = self.get_trading_symbol(from_asset, to_asset)

            if symbol:
                # Пряма конвертація з правильною валідацією
                if side == 'sell':
                    # Для sell ордерів при max використовуємо весь баланс без округлення
                    order = self.safe_market_order(symbol, 'sell', convert_amount,
                                                 is_quote_qty=False, force_max=is_max_conversion)
                else:
                    order = self.safe_market_order(symbol, 'buy', convert_amount, is_quote_qty=True)

                if order:
                    print(f"✅ Конвертація виконана! ID: {order['orderId']}")
                    return True
                return False
            else:
                # Конвертація через USDT
                return self._convert_via_usdt(from_asset, to_asset, convert_amount, is_max_conversion)

    def _convert_via_usdt(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        """Конвертація через USDT"""
        try:
            # Крок 1: В USDT
            usdt_symbol, usdt_side = self.get_trading_symbol(from_asset, 'USDT')
            if not usdt_symbol:
                print("❌ Неможливо конвертувати в USDT")
                return False

            if usdt_side == 'sell':
                usdt_order = self.safe_market_order(usdt_symbol, 'sell', amount,
                                                  is_quote_qty=False, force_max=is_max)
            else:
                usdt_order = self.safe_market_order(usdt_symbol, 'buy', amount, is_quote_qty=True)

            if not usdt_order:
                return False

            print(f"✅ Конвертовано в USDT, ID: {usdt_order['orderId']}")
            time.sleep(1)

            # Крок 2: З USDT в цільову валюту
            target_symbol, target_side = self.get_trading_symbol('USDT', to_asset)
            if not target_symbol:
                print(f"❌ Неможливо конвертувати з USDT в {to_asset}")
                return False

            usdt_balance = self.get_balance('USDT')

            if target_side == 'sell':
                target_order = self.safe_market_order(target_symbol, 'sell', usdt_balance,
                                                    is_quote_qty=False, force_max=True)
            else:
                target_order = self.safe_market_order(target_symbol, 'buy', usdt_balance, is_quote_qty=True)

            if target_order:
                print(f"✅ Конвертовано в {to_asset}, ID: {target_order['orderId']}")
                return True
            return False
        except Exception as e:
            print(f"❌ Помилка конвертації через USDT: {e}")
            return False

    def show_balance(self):
        """Показати баланс"""
        print(f"\n💰 Баланс ({'ТЕСТ' if self.testnet else 'РЕАЛ'})")

        if self.testnet:
            balances = self.test_balance
        else:
            try:
                account = self.client.get_account()
                balances = {b['asset']: float(b['free']) for b in account['balances'] if float(b['free']) > 0}
            except Exception:
                print("❌ Помилка отримання балансу")
                return

        total_usd = 0
        for asset, amount in balances.items():
            if amount > 0:
                usd_value = amount * self.rates.get(asset, 1.0)
                total_usd += usd_value
                print(f"  {asset}: {amount:,.8f} (~${usd_value:,.2f})")
        print(f"💵 Загалом: ${total_usd:,.2f}\n")

    def add_test_balance(self, asset: str, amount: float):
        """Додати тестовий баланс"""
        if not self.testnet:
            print("❌ Тільки для тестового режиму")
            return
        asset = asset.upper()
        self.test_balance[asset] = self.test_balance.get(asset, 0) + amount
        print(f"💰 Додано {amount} {asset}")


def interactive_converter():
    """Інтерактивний конвертер"""
    print("🔄 === КОНВЕРТЕР ТОКЕНІВ ===")

    while True:
        mode = input("Режим (test/real): ").strip().lower()
        if mode in ['test', 'тест']:
            testnet = True
            break
        elif mode in ['real', 'реал']:
            testnet = False
            break
        else:
            print("❌ Оберіть 'test' або 'real'")

    try:
        trader = SimpleBinanceTrader(testnet=testnet)
    except Exception as e:
        print(f"❌ Не вдалося ініціалізувати: {e}")
        return

    print("\nКоманди: balance, convert, add, update, clean, exit")

    while True:
        try:
            command = input("\n👉 Команда: ").strip().lower()

            if command == 'exit':
                print("👋 До побачення!")
                break
            elif command == 'balance':
                trader.show_balance()

            elif command == 'update':
                trader.update_rates()

            elif command == 'refresh':
                trader.update_exchange_info()

            elif command == 'clean':
                if not trader.testnet:
                    asset = input("Токен для очищення (або Enter для всіх): ").strip().upper()
                    trader.clear_dust(asset if asset else None)
                else:
                    print("❌ Тільки для реального режиму")

            elif command == 'add':
                if not trader.testnet:
                    print("❌ Тільки для тестового режиму")
                    continue

                asset = input("Токен: ").strip().upper()
                try:
                    amount = float(input("Кількість: "))
                    trader.add_test_balance(asset, amount)
                except ValueError:
                    print("❌ Неправильна кількість")

            elif command == 'convert':
                from_token = input("З: ").strip().upper()
                to_token = input("В: ").strip().upper()

                if not from_token or not to_token or from_token == to_token:
                    print("❌ Неправильні токени")
                    continue

                balance = trader.get_balance(from_token)
                if balance <= 0:
                    print(f"❌ Немає {from_token}")
                    continue

                print(f"💰 Баланс {from_token}: {balance:,.8f}")
                amount_input = input("Кількість (або 'max'): ").strip()
                if not amount_input:
                    continue

                confirm = input("⚠️ Підтвердити конвертацію? (y/n): ")
                if confirm.lower() in ['y', 'yes', 'так']:
                    trader.convert(from_token, to_token, amount_input)
                else:
                    print("❌ Скасовано")

            else:
                print("❌ Невідома команда")

        except KeyboardInterrupt:
            print("\n👋 Вихід")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    interactive_converter()
