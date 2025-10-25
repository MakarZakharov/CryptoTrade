import os
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

load_dotenv()


class BTCETH_CMC20_Trader:
    """Автоматизований трейдер з розподілом портфеля між BTC та ETH на основі CMC20 Index"""

    def __init__(self):
        # Binance API
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = Client(self.binance_api_key, self.binance_api_secret)

        # CoinMarketCap API
        self.cmc_api_key = os.getenv("COINMARKETCAP_API_KEY")
        self.cmc_api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        self.update_interval = int(os.getenv("CMC_INDEX_UPDATE_INTERVAL", 3600))

        # Список стейблкоїнів для виключення
        self.stablecoins = ['USDT', 'USDC', 'BUSD', 'FDUSD', 'USDe', 'DAI', 'TUSD', 'USDP', 'USDD', 'GUSD', 'PYUSD']

    def get_binance_balance(self, asset="USDC") -> float:
        """Отримує доступний баланс на Binance"""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            free_balance = float(balance['free'])
            print(f"💰 Доступний баланс {asset}: {free_balance:,.2f}")
            return free_balance
        except BinanceAPIException as e:
            print(f"❌ Помилка отримання балансу: {e}")
            return 0.0

    def get_all_binance_balances(self) -> dict:
        """Отримує всі баланси на Binance з вартістю в USDC"""
        try:
            account = self.client.get_account()
            balances = {}
            total_portfolio_usdc = 0.0

            print("\n💼 Поточні баланси на Binance:")
            print("-" * 90)

            for balance in account['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked

                if total > 0:
                    asset = balance['asset']

                    # Розраховуємо вартість в USDC
                    usdc_value = 0.0
                    if asset in self.stablecoins:
                        usdc_value = total
                    else:
                        try:
                            ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDC")
                            price = float(ticker['price'])
                            usdc_value = total * price
                        except:
                            try:
                                ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                                price = float(ticker['price'])
                                usdc_value = total * price
                            except:
                                try:
                                    ticker_btc = self.client.get_symbol_ticker(symbol=f"{asset}BTC")
                                    btc_price = float(ticker_btc['price'])
                                    btc_usdc = float(self.client.get_symbol_ticker(symbol="BTCUSDC")['price'])
                                    usdc_value = total * btc_price * btc_usdc
                                except:
                                    usdc_value = 0.0

                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': total,
                        'usdc_value': usdc_value
                    }

                    total_portfolio_usdc += usdc_value

                    print(f"{asset:6s} | Вільно: {free:12,.6f} | Заблоковано: {locked:12,.6f} | "
                          f"≈ ${usdc_value:10,.2f} USDC")

            print("-" * 90)
            print(f"{'РАЗОМ':6s} | {'':12s}   {'':12s}   {'':15s}   ≈ ${total_portfolio_usdc:10,.2f} USDC")
            print("-" * 90)

            return balances, total_portfolio_usdc

        except BinanceAPIException as e:
            print(f"❌ Помилка отримання балансів: {e}")
            return {}, 0.0

    def get_btc_eth_allocation_from_cmc20(self) -> dict:
        """Отримує топ-20 токенів, бере ваги BTC та ETH + перерозподіл решти 18 ПРОПОРЦІЙНО"""
        try:
            headers = {
                'X-CMC_PRO_API_KEY': self.cmc_api_key,
                'Accept': 'application/json'
            }

            params = {
                'start': '1',
                'limit': '50',
                'convert': 'USD'
            }

            response = requests.get(self.cmc_api_url, headers=headers, params=params)
            data = response.json()

            if response.status_code != 200:
                print(f"❌ Помилка API: {data.get('status', {}).get('error_message')}")
                return {}

            coins = data['data']

            # Видаляємо всі стейблкоїни зі списку
            coins = [coin for coin in coins if coin['symbol'] not in self.stablecoins]

            # Беремо топ-20 (без стейблкоїнів)
            top20_coins = coins[:20]

            # Розраховуємо загальну ринкову капіталізацію топ-20
            total_market_cap = sum(coin['quote']['USD']['market_cap'] for coin in top20_coins)

            # Знаходимо BTC та ETH
            btc_data = None
            eth_data = None
            other_18_total_market_cap = 0.0

            for coin in top20_coins:
                market_cap = coin['quote']['USD']['market_cap']

                if coin['symbol'] == 'BTC':
                    btc_data = coin
                elif coin['symbol'] == 'ETH':
                    eth_data = coin
                else:
                    other_18_total_market_cap += market_cap

            if not btc_data or not eth_data:
                print("❌ BTC або ETH не знайдено в топ-20")
                return {}

            # Розраховуємо початкові ваги BTC та ETH в топ-20
            btc_original_weight = (btc_data['quote']['USD']['market_cap'] / total_market_cap) * 100
            eth_original_weight = (eth_data['quote']['USD']['market_cap'] / total_market_cap) * 100

            # Розраховуємо вагу решти 18 токенів
            other_18_weight = (other_18_total_market_cap / total_market_cap) * 100

            # Ділимо вагу решти 18 токенів ПОРІВНУ (50/50) між BTC та ETH
            redistribution_per_token = other_18_weight / 2

            # Фінальні ваги: кожен отримує свою оригінальну вагу + 50% від решти 18
            btc_final_weight = btc_original_weight + redistribution_per_token
            eth_final_weight = eth_original_weight + redistribution_per_token

            # Формуємо результат
            allocation_data = {
                'BTC': {
                    'rank': btc_data['cmc_rank'],
                    'name': btc_data['name'],
                    'original_weight': btc_original_weight,
                    'redistribution_bonus': redistribution_per_token,
                    'weight': btc_final_weight,
                    'market_cap': btc_data['quote']['USD']['market_cap'],
                    'price': btc_data['quote']['USD']['price'],
                    'change_24h': btc_data['quote']['USD']['percent_change_24h']
                },
                'ETH': {
                    'rank': eth_data['cmc_rank'],
                    'name': eth_data['name'],
                    'original_weight': eth_original_weight,
                    'redistribution_bonus': redistribution_per_token,
                    'weight': eth_final_weight,
                    'market_cap': eth_data['quote']['USD']['market_cap'],
                    'price': eth_data['quote']['USD']['price'],
                    'change_24h': eth_data['quote']['USD']['percent_change_24h']
                }
            }

            # Виводимо інформацію про розподіл для верифікації
            print(f"\n🔍 ДЕТАЛІ РОЗПОДІЛУ CMC20 (50/50):")
            print(f"   📊 Загальна капіталізація топ-20: ${total_market_cap:,.0f}")
            print(f"   💰 BTC оригінал: {btc_original_weight:.2f}% (${btc_data['quote']['USD']['market_cap']:,.0f})")
            print(f"   💰 ETH оригінал: {eth_original_weight:.2f}% (${eth_data['quote']['USD']['market_cap']:,.0f})")
            print(f"   📦 Решта 18 токенів: {other_18_weight:.2f}% (${other_18_total_market_cap:,.0f})")
            print(f"   ➗ Розподіл 18 токенів: 50% BTC + 50% ETH")
            print(f"   ➕ BTC отримує: +{redistribution_per_token:.2f}%")
            print(f"   ➕ ETH отримує: +{redistribution_per_token:.2f}%")
            print(f"   ✅ BTC фінал: {btc_final_weight:.2f}%")
            print(f"   ✅ ETH фінал: {eth_final_weight:.2f}%")
            print(f"   🎯 Перевірка суми: {btc_final_weight + eth_final_weight:.2f}% (має бути 100%)\n")

            return allocation_data

        except Exception as e:
            print(f"❌ Помилка отримання CMC20 індексу: {e}")
            return {}

    def display_btc_eth_allocation_chart(self, total_portfolio_value: float):
        """Відображає BTC та ETH з перерозподілом решти 18 токенів"""
        print("\n" + "=" * 120)
        print("📈 BTC + ETH PORTFOLIO (НА ОСНОВІ COINMARKETCAP TOP-20 INDEX)")
        print("=" * 120)

        allocation_data = self.get_btc_eth_allocation_from_cmc20()

        if not allocation_data:
            print("❌ Не вдалося отримати дані індексу")
            return {}

        print(f"\n💼 Загальна вартість портфеля: ${total_portfolio_value:,.2f} USDC")
        print(f"🎯 Цільовий розподіл: BTC + ETH з перерозподілом решти 18 токенів CMC20 порівну\n")

        # Сортуємо за рангом
        sorted_coins = sorted(allocation_data.items(), key=lambda x: x[1]['rank'])

        print("┌" + "─" * 118 + "┐")
        print(f"│ {'#':>3} │ {'Токен':^8} │ {'Назва':<18} │ {'Початкова %':>13} │ {'Бонус %':>10} │ "
              f"{'Фінальна %':>12} │ {'Цільова сума $':>16} │ {'Ціна USD':>14} │ {'24h %':>8} │")
        print("├" + "─" * 118 + "┤")

        final_allocation = {}
        total_allocated = 0.0

        for display_num, (symbol, data) in enumerate(sorted_coins, 1):
            target_value = total_portfolio_value * (data['weight'] / 100)
            total_allocated += target_value

            final_allocation[symbol] = {
                'weight': data['weight'] / 100,
                'target_value': target_value,
                'price': data['price'],
                'change_24h': data['change_24h'],
                'rank': data['rank']
            }

            # Форматування
            change_color = "+" if data['change_24h'] >= 0 else ""
            target_str = f"${target_value:,.2f}"
            original_weight_str = f"{data['original_weight']:.2f}%"
            bonus_str = f"+{data['redistribution_bonus']:.2f}%"
            final_weight_str = f"{data['weight']:.2f}%"

            print(f"│ {display_num:>3} │ {symbol:^8} │ {data['name']:<18.18} │ "
                  f"{original_weight_str:>13} │ {bonus_str:>10} │ {final_weight_str:>12} │ "
                  f"{target_str:>16} │ ${data['price']:>13,.2f} │ {change_color}{data['change_24h']:>7.2f}% │")

        print("└" + "─" * 118 + "┘")

        # Підсумок
        print(f"\n💼 Цільова сума BTC+ETH: ${total_allocated:,.2f} USDC (100% від ${total_portfolio_value:,.2f})")
        print(
            f"📊 Кожен токен отримав додатково: +{allocation_data['BTC']['redistribution_bonus']:.2f}% (50% від решти 18 токенів)")
        print(
            f"⚖️ Фінальне співвідношення: BTC {allocation_data['BTC']['weight']:.2f}% / ETH {allocation_data['ETH']['weight']:.2f}%")
        print("=" * 120 + "\n")

        return final_allocation

    def display_rebalancing_table(self, current_balances: dict, target_allocation: dict, total_portfolio_value: float):
        """Відображає таблицю з поточними балансами та необхідним ребалансуванням (BTC+ETH)"""
        print("\n" + "=" * 120)
        print("⚖️ ТАБЛИЦЯ РЕБАЛАНСУВАННЯ ПОРТФЕЛЯ (BTC + ETH)")
        print("=" * 120)

        print(f"\n💰 Загальна вартість портфеля: ${total_portfolio_value:,.2f} USDC\n")

        print("┌" + "─" + "─" * 118 + "┐")
        print(f"│ {'Токен':^8} │ {'Поточна к-сть':>15} │ {'Поточна $':>14} │ {'Поточна %':>11} │ "
              f"{'Цільова $':>14} │ {'Цільова %':>11} │ {'Різниця $':>14} │ {'Дія':^15} │")
        print("├" + "─" * 118 + "┤")

        # Беремо тільки BTC та ETH
        tokens = ['BTC', 'ETH']
        total_difference = 0.0

        for token in tokens:
            # Поточні дані
            current_balance = current_balances.get(token, {}).get('total', 0)
            current_value = current_balances.get(token, {}).get('usdc_value', 0)
            current_percent = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

            # Цільові дані
            target_value = target_allocation.get(token, {}).get('target_value', 0)
            target_percent = target_allocation.get(token, {}).get('weight', 0) * 100

            # Різниця
            difference = target_value - current_value
            total_difference += abs(difference)

            # Визначаємо дію
            if abs(difference) < 1:
                action = "✓ OK"
            elif difference > 0:
                action = f"🟢 КУПИТИ"
            else:
                action = f"🔴 ПРОДАТИ"

            # Форматування
            current_str = f"{current_balance:,.8f}".rstrip('0').rstrip('.')
            difference_str = f"{difference:+,.2f}" if difference != 0 else "0.00"

            print(f"│ {token:^8} │ {current_str:>15} │ ${current_value:>12,.2f} │ {current_percent:>10.2f}% │ "
                  f"${target_value:>12,.2f} │ {target_percent:>10.2f}% │ ${difference_str:>12} │ {action:^15} │")

        print("└" + "─" * 118 + "┘")

        # Показуємо стейблкоїни окремо
        stablecoins_total = sum(current_balances.get(coin, {}).get('usdc_value', 0)
                                for coin in self.stablecoins)

        print(f"\n💵 Доступні стейблкоїни для купівлі: ${stablecoins_total:,.2f} USDC")
        print(f"📊 Загальна різниця для ребалансування: ${total_difference / 2:,.2f}")
        print("=" * 120 + "\n")

    def get_binance_price(self, symbol: str) -> float:
        """Отримує поточну ціну токена на Binance"""
        try:
            pair = f"{symbol}USDC"
            ticker = self.client.get_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        except BinanceAPIException:
            try:
                pair = f"{symbol}USDT"
                ticker = self.client.get_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            except BinanceAPIException as e:
                print(f"❌ Не вдалося отримати ціну {symbol}: {e}")
                return 0.0

    def get_trading_pair(self, symbol: str) -> str:
        """Визначає доступну торгову пару"""
        try:
            pair = f"{symbol}USDC"
            self.client.get_symbol_ticker(symbol=pair)
            return "USDC"
        except BinanceAPIException:
            try:
                pair = f"{symbol}USDT"
                self.client.get_symbol_ticker(symbol=pair)
                return "USDT"
            except BinanceAPIException:
                return None

    def execute_market_order(self, symbol: str, side: str, quantity: float, quote_currency: str = "USDC") -> bool:
        """Виконує ринковий ордер (для сум >$5)"""
        try:
            pair = f"{symbol}{quote_currency}"
            info = self.client.get_symbol_info(pair)
            step_size = None

            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    break

            if step_size:
                precision = len(str(step_size).rstrip('0').split('.')[-1])
                quantity = round(quantity, precision)

            print(f"📊 Виконується {'КУПІВЛЯ' if side == 'BUY' else 'ПРОДАЖ'} {quantity} {symbol} (MARKET ORDER)...")

            if side == 'BUY':
                order = self.client.order_market_buy(symbol=pair, quantity=quantity)
            else:
                order = self.client.order_market_sell(symbol=pair, quantity=quantity)

            print(f"✅ Ордер виконано: {order['orderId']}")
            print(f"   {'Куплено' if side == 'BUY' else 'Продано'}: {order['executedQty']} {symbol}")
            print(f"   {'Витрачено' if side == 'BUY' else 'Отримано'}: {order['cummulativeQuoteQty']} {quote_currency}")
            return True

        except BinanceAPIException as e:
            print(f"❌ Помилка ордеру {symbol}: {e}")
            return False

    def execute_convert(self, from_asset: str, to_asset: str, amount: float) -> bool:
        """Виконує конвертацію через Binance Convert API"""
        try:
            print(f"🔄 Конвертація {amount:.8f} {from_asset} → {to_asset}...")

            # Використовуємо Binance Convert API
            result = self.client.convert_asset(
                fromAsset=from_asset,
                toAsset=to_asset,
                fromAmount=amount
            )

            if result and result.get('orderId'):
                print(f"✅ Конвертацію виконано успішно!")
                print(f"   Order ID: {result['orderId']}")
                print(f"   Конвертовано: {result.get('fromAmount', amount)} {from_asset}")
                print(f"   Отримано: {result.get('toAmount', 'N/A')} {to_asset}")
                return True
            else:
                print(f"❌ Помилка конвертації: невідома відповідь від API")
                return False

        except BinanceAPIException as e:
            print(f"❌ Помилка конвертації {from_asset} → {to_asset}: {e}")
            return False
        except Exception as e:
            print(f"❌ Невідома помилка конвертації: {e}")
            return False

    def calculate_rebalancing_orders(self, current_balances: dict, target_allocation: dict,
                                     total_portfolio_value: float) -> dict:
        """Розраховує необхідні операції для ребалансування (гібридна система)"""
        operations = {
            'sell_orders': {},  # Продаж через market orders (>$5)
            'sell_convert': {},  # Продаж через convert (<=$5)
            'buy_orders': {},  # Купівля через market orders (>$5)
            'buy_convert': {}  # Купівля через convert (<=$5)
        }

        THRESHOLD = 5.0  # Поріг для вибору між ордерами та конвертацією

        print(f"\n💵 Розрахунок операцій для ребалансування:")
        print(f"📊 Поріг: ордери для сум >${THRESHOLD}$, конвертація для сум <=${THRESHOLD}$")
        print("-" * 80)

        # Визначаємо, який стейблкоїн використовувати
        quote_currency = None
        quote_balance = 0

        for stable in ['USDC', 'USDT', 'BUSD', 'FDUSD']:
            balance = current_balances.get(stable, {}).get('total', 0)
            if balance > 0.1:
                quote_currency = stable
                quote_balance = balance
                break

        if not quote_currency:
            quote_currency = 'USDC'
            quote_balance = 0
            print(f"⚠️ Немає стейблкоїнів, використовуємо {quote_currency} (буде поповнено з продажу)")
        else:
            print(f"💰 Поточний баланс {quote_currency}: ${quote_balance:.2f}")

        # Розраховуємо всі операції
        total_sell_value = 0
        total_buy_value = 0

        for symbol, target_data in target_allocation.items():
            current_value = current_balances.get(symbol, {}).get('usdc_value', 0)
            current_quantity = current_balances.get(symbol, {}).get('total', 0)
            target_value = target_data['target_value']

            difference_value = target_value - current_value

            if abs(difference_value) < 1:
                continue

            # Отримуємо ціну для розрахунку кількості
            price = self.get_binance_price(symbol)
            if price == 0:
                continue

            if difference_value > 0:
                # КУПІВЛЯ
                quantity = difference_value / price
                total_buy_value += difference_value

                if difference_value > THRESHOLD:
                    # Великі суми -> market order
                    operations['buy_orders'][symbol] = {
                        'quantity': quantity,
                        'value_usdc': difference_value,
                        'price': price,
                        'quote_currency': quote_currency
                    }
                    print(f"🟢 MARKET BUY {symbol}: {quantity:,.8f} токенів на ${difference_value:,.2f}")
                else:
                    # Малі суми -> convert
                    operations['buy_convert'][symbol] = {
                        'from_asset': quote_currency,
                        'to_asset': symbol,
                        'amount': difference_value,
                        'type': 'convert'
                    }
                    print(f"🔵 CONVERT {quote_currency}→{symbol}: ${difference_value:,.2f}")
            else:
                # ПРОДАЖ
                quantity = abs(difference_value) / price
                total_sell_value += abs(difference_value)

                if abs(difference_value) > THRESHOLD:
                    # Великі суми -> market order
                    operations['sell_orders'][symbol] = {
                        'quantity': quantity,
                        'value_usdc': abs(difference_value),
                        'price': price,
                        'quote_currency': quote_currency
                    }
                    print(f"🔴 MARKET SELL {symbol}: {quantity:,.8f} токенів на ${abs(difference_value):,.2f}")
                else:
                    # Малі суми -> convert
                    operations['sell_convert'][symbol] = {
                        'from_asset': symbol,
                        'to_asset': quote_currency,
                        'amount': abs(difference_value),
                        'current_quantity': current_quantity,
                        'type': 'convert'
                    }
                    print(f"🟠 CONVERT {symbol}→{quote_currency}: ${abs(difference_value):,.2f}")

        # Перевірка балансу
        if any(operations.values()):
            available_after_sell = quote_balance + total_sell_value
            print(f"\n💰 Баланс після продажу: ${available_after_sell:.2f}")
            print(f"📊 Потрібно для купівлі: ${total_buy_value:.2f}")

            if available_after_sell >= total_buy_value:
                print(f"✅ Достатньо коштів для ребалансування")
            else:
                print(f"⚠️ Недостатньо коштів! Бракує: ${total_buy_value - available_after_sell:.2f}")

        print("-" * 80)
        return operations

    def execute_portfolio_rebalance(self, dry_run=False):
        """Виконує ребалансування портфеля (гібридна система: ордери + конвертація)"""
        print("\n" + "=" * 80)
        print(f"🚀 ПОЧАТОК РЕБАЛАНСУВАННЯ ПОРТФЕЛЯ (BTC + ETH)")
        print(f"🕐 Час: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        current_balances, total_portfolio_value = self.get_all_binance_balances()

        if total_portfolio_value <= 0:
            print("❌ Портфель порожній")
            return

        target_allocation = self.display_btc_eth_allocation_chart(total_portfolio_value)

        if not target_allocation:
            print("❌ Не вдалося отримати дані з CoinMarketCap")
            return

        self.display_rebalancing_table(current_balances, target_allocation, total_portfolio_value)

        operations = self.calculate_rebalancing_orders(current_balances, target_allocation, total_portfolio_value)

        if not any(operations.values()):
            print("✅ Портфель вже збалансований")
            return

        if dry_run:
            print("\n" + "=" * 80)
            print("⚠️ DRY RUN MODE - операції НЕ будуть виконані")
            print("=" * 80)

            total_ops = sum(len(ops) for ops in operations.values())
            print(f"\n📋 Всього заплановано операцій: {total_ops}")

            if operations['sell_orders']:
                print(f"\n🔴 Market Sell ордери ({len(operations['sell_orders'])}):")
                for symbol, data in operations['sell_orders'].items():
                    print(f"   Продати {data['quantity']:,.8f} {symbol} ≈ ${data['value_usdc']:,.2f}")

            if operations['sell_convert']:
                print(f"\n🟠 Convert Sell операції ({len(operations['sell_convert'])}):")
                for symbol, data in operations['sell_convert'].items():
                    print(f"   Конвертувати {symbol}→{data['to_asset']} ≈ ${data['amount']:,.2f}")

            if operations['buy_orders']:
                print(f"\n🟢 Market Buy ордери ({len(operations['buy_orders'])}):")
                for symbol, data in operations['buy_orders'].items():
                    print(f"   Купити {data['quantity']:,.8f} {symbol} ≈ ${data['value_usdc']:,.2f}")

            if operations['buy_convert']:
                print(f"\n🔵 Convert Buy операції ({len(operations['buy_convert'])}):")
                for symbol, data in operations['buy_convert'].items():
                    print(f"   Конвертувати {data['from_asset']}→{symbol} ≈ ${data['amount']:,.2f}")

            print("\n💡 Щоб виконати реальні операції, встановіть dry_run=False")
        else:
            print("\n" + "=" * 80)
            print("🔄 ПОЧИНАЄМО ВИКОНАННЯ ОПЕРАЦІЙ (РЕАЛЬНІ ТРЕЙДИ!)")
            print("=" * 80)

            # ЕТАП 1: ПРОДАЖ (спочатку market orders, потім convert)
            if operations['sell_orders'] or operations['sell_convert']:
                print("\n" + "=" * 80)
                print("📤 ЕТАП 1: ПРОДАЖ ТОКЕНІВ")
                print("=" * 80)

                # 1.1 Market Sell Orders (великі суми)
                if operations['sell_orders']:
                    print("\n🔴 Виконання Market Sell ордерів:")
                    print("-" * 80)
                    for symbol, data in operations['sell_orders'].items():
                        success = self.execute_market_order(
                            symbol=symbol,
                            side='SELL',
                            quantity=data['quantity'],
                            quote_currency=data['quote_currency']
                        )
                        if success:
                            time.sleep(1)

                # 1.2 Convert Sell (малі суми)
                if operations['sell_convert']:
                    print("\n🟠 Виконання Convert Sell операцій:")
                    print("-" * 80)
                    for symbol, data in operations['sell_convert'].items():
                        token_price = current_balances.get(symbol, {}).get('usdc_value', 0) / max(
                            current_balances.get(symbol, {}).get('total', 1), 1)
                        quantity_to_convert = data['amount'] / token_price if token_price > 0 else 0

                        if quantity_to_convert > 0:
                            success = self.execute_convert(
                                from_asset=data['from_asset'],
                                to_asset=data['to_asset'],
                                amount=quantity_to_convert
                            )
                            if success:
                                time.sleep(2)

            # ЕТАП 2: КУПІВЛЯ (спочатку market orders, потім convert)
            if operations['buy_orders'] or operations['buy_convert']:
                print("\n" + "=" * 80)
                print("📥 ЕТАП 2: КУПІВЛЯ ТОКЕНІВ")
                print("=" * 80)

                # 2.1 Market Buy Orders (великі суми)
                if operations['buy_orders']:
                    print("\n🟢 Виконання Market Buy ордерів:")
                    print("-" * 80)
                    for symbol, data in operations['buy_orders'].items():
                        success = self.execute_market_order(
                            symbol=symbol,
                            side='BUY',
                            quantity=data['quantity'],
                            quote_currency=data['quote_currency']
                        )
                        if success:
                            time.sleep(1)

                # 2.2 Convert Buy (малі суми)
                if operations['buy_convert']:
                    print("\n🔵 Виконання Convert Buy операцій:")
                    print("-" * 80)
                    for symbol, data in operations['buy_convert'].items():
                        success = self.execute_convert(
                            from_asset=data['from_asset'],
                            to_asset=data['to_asset'],
                            amount=data['amount']
                        )
                        if success:
                            time.sleep(2)

        print("\n" + "=" * 80)
        print("✅ РЕБАЛАНСУВАННЯ ЗАВЕРШЕНО")
        print("=" * 80)

    def run_continuous_rebalance(self, dry_run=False):
        """Постійне ребалансування кожні N секунд згідно з .env"""
        interval_seconds = self.update_interval

        print("\n" + "=" * 80)
        print("🤖 ЗАПУСК АВТОМАТИЧНОГО РЕБАЛАНСУВАННЯ (BTC + ETH)")
        print("=" * 80)
        print(f"⏱ Інтервал оновлення: {interval_seconds} секунд ({interval_seconds / 60:.1f} хвилин)")
        print(f"⚠️ Режим: {'DRY RUN (тестовий)' if dry_run else 'РЕАЛЬНІ КОНВЕРТАЦІЇ'}")
        print(f"🕐 Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        cycle_count = 0

        while True:
            cycle_count += 1
            print(f"\n\n{'=' * 80}")
            print(f"🔄 ЦИКЛ РЕБАЛАНСУВАННЯ #{cycle_count}")
            print(f"{'=' * 80}")

            try:
                self.execute_portfolio_rebalance(dry_run=dry_run)

                next_run = datetime.now() + timedelta(seconds=interval_seconds)

                print(f"\n⏰ Наступне ребалансування через {interval_seconds} секунд")
                print(f"📅 Заплановано на: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"\n{'=' * 80}")
                print("😴 Очікування...")
                print(f"{'=' * 80}\n")

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                print("\n\n" + "=" * 80)
                print("⛔ ЗУПИНКА АВТОМАТИЧНОГО РЕБАЛАНСУВАННЯ")
                print("=" * 80)
                print(f"📊 Всього виконано циклів: {cycle_count}")
                print(f"🕐 Час зупинки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                break
            except Exception as e:
                print(f"\n❌ Помилка в циклі ребалансування: {e}")
                print(f"⏰ Спроба повторного запуску через {interval_seconds} секунд...")
                time.sleep(interval_seconds)


# Приклад використання
if __name__ == "__main__":
    trader = BTCETH_CMC20_Trader()

    # РЕЖИМ 1: Одноразове ребалансування (тестовий режим)
    # trader.execute_portfolio_rebalance(dry_run=True)

    # РЕЖИМ 2: Одноразове ребалансування (реальні конвертації)
    # trader.execute_portfolio_rebalance(dry_run=False)

    # РЕЖИМ 3: Постійне автоматичне ребалансування кожну годину (як в .env)
    # Тестовий режим (безпечно для тестування)
    trader.run_continuous_rebalance(dry_run=False)

    # Реальні конвертації (розкоментуй коли готовий до продакшену)
    # trader.run_continuous_rebalance(dry_run=False)
