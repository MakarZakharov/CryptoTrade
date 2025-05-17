import ccxt
import os
import sys
from dotenv import load_dotenv
import time
import logging
from decimal import Decimal
from typing import Dict, Optional, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("mini_binance.log"), logging.StreamHandler()])
logger = logging.getLogger("mini_binance")

load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
USE_TESTNET = os.getenv('USE_TESTNET', 'True').lower() in ('true', 't', '1', 'yes')

MICRO_NOTIONAL_THRESHOLD = Decimal('0.0001')
DUST_THRESHOLD = Decimal('0.00001')


class BinanceBot:
    def __init__(self, api_key: str, api_secret: str, use_testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.exchange = None
        self.markets = {}
        self._balance_cache = {}
        self._balance_time = 0
        self.intermediaries = ['USDT', 'BTC', 'ETH', 'BNB']
        self.conversion_history: List[Dict] = []

        self.testnet_balance = {
            'BTC': Decimal('0.01'),
            'ETH': Decimal('0.1'),
            'BNB': Decimal('1.0'),
            'USDT': Decimal('100.0')
        }

    def connect(self) -> bool:
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'createMarketBuyOrderRequiresPrice': False,
                    'recvWindow': 60000,
                }
            })

            if self.use_testnet:
                logger.info("TESTNET MODE ENABLED")
                self.exchange.set_sandbox_mode(True)

            self.exchange.load_markets()
            self.markets = self.exchange.markets
            logger.info(f"Connected! Markets available: {len(self.markets)}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def get_balance(self, currency: str = None, force_refresh: bool = False) -> Dict:
        current_time = time.time()

        if not force_refresh and current_time - self._balance_time < 30 and self._balance_cache:
            balance = self._balance_cache
        else:
            try:
                if self.use_testnet:
                    balance = {
                        'free': {k: v for k, v in self.testnet_balance.items()},
                        'used': {'USDT': Decimal('0.1'), 'BTC': Decimal('0.001')},
                        'total': {}
                    }
                    for curr in set(balance['free']) | set(balance['used']):
                        free = balance['free'].get(curr, Decimal('0'))
                        used = balance['used'].get(curr, Decimal('0'))
                        balance['total'][curr] = free + used
                else:
                    raw = self.exchange.fetch_balance()
                    balance = {
                        'free': {k: Decimal(str(v)) for k, v in raw.get('free', {}).items() if float(v) > 0},
                        'used': {k: Decimal(str(v)) for k, v in raw.get('used', {}).items() if float(v) > 0},
                        'total': {k: Decimal(str(v)) for k, v in raw.get('total', {}).items() if float(v) > 0}
                    }

                self._balance_cache = balance
                self._balance_time = current_time

            except Exception as e:
                logger.error(f"Failed to get balance: {e}")
                if not self._balance_cache:
                    return {'free': Decimal('0'), 'used': Decimal('0'), 'total': Decimal('0')} if currency else {}
                balance = self._balance_cache

        if currency:
            curr = currency.upper()
            return {
                'free': balance.get('free', {}).get(curr, Decimal('0')),
                'used': balance.get('used', {}).get(curr, Decimal('0')),
                'total': balance.get('total', {}).get(curr, Decimal('0'))
            }
        return balance

    def get_market_info(self, symbol: str) -> Dict:
        try:
            market = self.markets[symbol]
            min_notional = Decimal('0.001')

            for filter_item in market.get('info', {}).get('filters', []):
                if filter_item.get('filterType') == 'MIN_NOTIONAL':
                    min_notional = Decimal(str(filter_item.get('minNotional', '0.001')))
                    break

            return {
                'min_amount': Decimal(str(market['limits']['amount']['min'])),
                'min_cost': Decimal(str(market.get('limits', {}).get('cost', {}).get('min', 0))),
                'min_notional': min_notional,
                'amount_precision': market['precision']['amount'],
                'price_precision': market['precision']['price'],
                'base': market['base'],
                'quote': market['quote']
            }
        except Exception as e:
            logger.error(f"Failed to get market info for {symbol}: {e}")
            return None

    def find_direct_path(self, from_curr: str, to_curr: str) -> Dict:
        from_curr, to_curr = from_curr.upper(), to_curr.upper()

        if from_curr == to_curr:
            return {'success': True, 'path': [], 'rate': Decimal('1.0')}

        direct_symbol = f"{from_curr}/{to_curr}"
        inverse_symbol = f"{to_curr}/{from_curr}"

        try:
            if direct_symbol in self.markets:
                ticker = self.exchange.fetch_ticker(direct_symbol)
                return {
                    'success': True,
                    'path': [{'symbol': direct_symbol, 'action': 'sell', 'from': from_curr, 'to': to_curr}],
                    'rate': Decimal(str(ticker['last']))
                }

            if inverse_symbol in self.markets:
                ticker = self.exchange.fetch_ticker(inverse_symbol)
                return {
                    'success': True,
                    'path': [{'symbol': inverse_symbol, 'action': 'buy', 'from': from_curr, 'to': to_curr}],
                    'rate': Decimal('1') / Decimal(str(ticker['last']))
                }

            for intermediary in self.intermediaries:
                if intermediary == from_curr or intermediary == to_curr:
                    continue

                step1_symbol = f"{from_curr}/{intermediary}"
                step2_symbol = f"{to_curr}/{intermediary}"

                if step1_symbol in self.markets and step2_symbol in self.markets:
                    ticker1 = self.exchange.fetch_ticker(step1_symbol)
                    ticker2 = self.exchange.fetch_ticker(step2_symbol)

                    return {
                        'success': True,
                        'path': [
                            {'symbol': step1_symbol, 'action': 'sell', 'from': from_curr, 'to': intermediary},
                            {'symbol': step2_symbol, 'action': 'sell', 'from': intermediary, 'to': to_curr}
                        ],
                        'rate': Decimal(str(ticker1['last'])) / Decimal(str(ticker2['last'])),
                        'intermediary': intermediary
                    }

            return {'success': False, 'path': []}
        except Exception as e:
            logger.error(f"Path finding error: {e}")
            return {'success': False, 'path': []}

    def place_micro_order(self, symbol: str, side: str, amount: Decimal, is_quote: bool = False) -> Optional[Dict]:
        try:
            market_info = self.get_market_info(symbol)
            if not market_info:
                return None

            price = Decimal(str(self.exchange.fetch_ticker(symbol)['last']))
            base, quote = market_info['base'], market_info['quote']

            is_micro = False
            notional_value = amount * price if side == 'sell' else amount

            if notional_value < market_info['min_notional']:
                is_micro = True
                logger.info(f"Micro trade detected: {notional_value} < {market_info['min_notional']}")
                order_options = {
                    'test': False,
                    'recvWindow': 60000,
                    'allowExtraSmall': True,
                    'ignoreMinAmountRestriction': True
                }
            else:
                order_options = {}

            if is_micro and not self.use_testnet:
                if side == 'sell':
                    if self.try_micro_conversion(base, quote, amount):
                        return {'id': 'micro-convert', 'status': 'closed'}
                else:
                    if self.try_micro_conversion(quote, base, amount * price):
                        return {'id': 'micro-convert', 'status': 'closed'}

            if self.use_testnet:
                logger.info(f"TESTNET: {side} order {symbol} amount: {float(amount)}")
                if side == 'sell':
                    if base in self.testnet_balance:
                        self.testnet_balance[base] -= amount
                    if quote in self.testnet_balance:
                        self.testnet_balance[quote] += amount * price * Decimal('0.999')
                else:
                    cost = amount * price if not is_quote else amount
                    if quote in self.testnet_balance:
                        self.testnet_balance[quote] -= cost
                    if base in self.testnet_balance:
                        received = amount if not is_quote else cost / price * Decimal('0.999')
                        self.testnet_balance[base] += received

                return {'id': f'test_{time.time()}', 'status': 'closed'}

            if side == 'sell':
                order = self.exchange.create_market_sell_order(symbol, float(amount), order_options)
            else:
                if is_quote:
                    order = self.exchange.create_market_order(symbol, 'buy', None, None, order_options)
                else:
                    order = self.exchange.create_market_buy_order(symbol, float(amount), order_options)

            self._balance_cache = {}
            return order

        except Exception as e:
            logger.error(f"Order error ({symbol}, {side}): {e}")
            return None

    def try_micro_conversion(self, from_curr: str, to_curr: str, amount: Decimal) -> bool:
        """Try alternative methods for handling very small amounts"""
        try:
            logger.info(f"Attempting micro conversion: {from_curr}->{to_curr} amount:{amount}")

            try:
                result = self.exchange.sapi_post_convert_trade_order({
                    'fromAsset': from_curr,
                    'toAsset': to_curr,
                    'fromAmount': float(amount)
                })
                if result and 'orderId' in result:
                    logger.info(f"Convert API success: {result['orderId']}")
                    return True
            except Exception as e:
                logger.warning(f"Convert API failed: {e}")

            if to_curr == 'BNB':
                try:
                    result = self.exchange.sapi_post_asset_dust({'asset': [from_curr]})
                    if result and 'totalTransfered' in result:
                        logger.info(f"Dust conversion success: {result['totalTransfered']} BNB")
                        return True
                except Exception as e:
                    logger.warning(f"Dust conversion failed: {e}")

            try:
                result = self.exchange.sapi_post_asset_convert_transfer({
                    'fromAsset': from_curr,
                    'toAsset': to_curr,
                    'amount': float(amount)
                })
                if result and result.get('success'):
                    logger.info(f"Small assets exchange success")
                    return True
            except Exception as e:
                logger.warning(f"Small assets exchange failed: {e}")

            return False

        except Exception as e:
            logger.error(f"Micro conversion error: {e}")
            return False

    def convert(self, amount: Decimal, from_curr: str, to_curr: str) -> bool:
        """Convert currency with special handling for microtrades"""
        from_curr = from_curr.upper()
        to_curr = to_curr.upper()

        balance = self.get_balance(from_curr)['free']
        if balance < amount:
            logger.error(f"Insufficient {from_curr}. Need: {amount}, Available: {balance}")
            return False

        is_micro = amount < MICRO_NOTIONAL_THRESHOLD
        if is_micro:
            logger.info(f"Micro trade detected: {amount} {from_curr}")

        path_info = self.find_direct_path(from_curr, to_curr)
        if not path_info['success']:
            logger.error(f"No conversion path found: {from_curr} -> {to_curr}")
            return False

        rate = path_info['rate']
        est_result = amount * rate
        print(f"\nConvert {amount:.8f} {from_curr} -> ~{est_result:.8f} {to_curr}")

        conversion_record = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'from_currency': from_curr,
            'from_amount': amount,
            'to_currency': to_curr,
            'to_amount': est_result,
            'rate': rate
        }

        success = False
        if len(path_info['path']) == 1:
            step = path_info['path'][0]
            if is_micro and not self.use_testnet:
                if self.try_micro_conversion(from_curr, to_curr, amount):
                    print("✅ Micro conversion successful")
                    success = True

            if not success:
                order = self.place_micro_order(step['symbol'], step['action'], amount,
                                               is_quote=(step['action'] == 'buy'))
                success = order is not None

        elif len(path_info['path']) == 2:
            mid = path_info['intermediary']
            logger.info(f"Two-step conversion via {mid}")

            step1 = path_info['path'][0]
            order1 = self.place_micro_order(step1['symbol'], step1['action'], amount,
                                            is_quote=(step1['action'] == 'buy'))
            if not order1:
                logger.error("Step 1 failed")
                return False

            time.sleep(2)
            mid_balance = self.get_balance(mid, force_refresh=True)['free']

            step2 = path_info['path'][1]
            order2 = self.place_micro_order(step2['symbol'], step2['action'], mid_balance,
                                            is_quote=(step2['action'] == 'buy'))
            success = order2 is not None

        if success:
            if len(self.conversion_history) >= 5:
                self.conversion_history.pop(0)
            self.conversion_history.append(conversion_record)

        return success

    def show_balance(self, min_value: Decimal = DUST_THRESHOLD) -> None:
        balance = self.get_balance(force_refresh=True)
        print("\n--- Balance ---")

        for curr, total in sorted(balance.get('total', {}).items()):
            if total > min_value:
                free = balance['free'].get(curr, Decimal('0'))
                print(f"{curr}: {free:.8f}")

        dust_currencies = [curr for curr, total in balance.get('total', {}).items()
                           if total > Decimal('0') and total <= min_value]
        if dust_currencies:
            print("\n--- Dust Balances ---")
            for curr in dust_currencies:
                free = balance['free'].get(curr, Decimal('0'))
                print(f"{curr}: {free:.8f}")

        if self.conversion_history:
            print("\n--- Recent Conversions ---")
            for i, conv in enumerate(self.conversion_history, 1):
                print(
                    f"{i}. [{conv['timestamp']}] {conv['from_amount']:.8f} {conv['from_currency']} → {conv['to_amount']:.8f} {conv['to_currency']}")

    def run(self) -> None:
        """Main program loop"""
        print("\n=== Mini Binance Bot (UltraTink Edition) ===")
        if self.use_testnet:
            print("⚠️ TESTNET MODE ACTIVE ⚠️")

        if not self.connect():
            sys.exit(1)

        while True:
            print("\n1. Convert Currency")
            print("2. Show Balance")
            print("3. Exit")

            try:
                choice = input("Choice (1-3): ").strip()

                if choice == '1':
                    self.show_balance()
                    from_c = input("From currency: ").upper().strip()
                    to_c = input("To currency: ").upper().strip()

                    balance = self.get_balance(from_c)['free']
                    if balance <= 0:
                        print(f"No {from_c} available")
                        continue

                    print(f"Available: {balance:.8f} {from_c}")
                    amount_input = input(f"Amount to convert (or 'max'): ").strip().lower()
                    amount = balance if amount_input == 'max' else Decimal(amount_input.replace(',', '.'))

                    if self.convert(amount, from_c, to_c):
                        print("✅ Conversion complete")
                        time.sleep(1)
                        self.show_balance()
                    else:
                        print("❌ Conversion failed")

                elif choice == '2':
                    self.show_balance()

                elif choice == '3':
                    print("Exiting")
                    break

            except KeyboardInterrupt:
                print("\nExiting")
                break
            except ValueError:
                print("Invalid input")
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")


if __name__ == "__main__":
    try:
        use_testnet = USE_TESTNET
        if len(sys.argv) > 1:
            if sys.argv[1].lower() in ('--testnet', '-t'):
                use_testnet = True
            elif sys.argv[1].lower() in ('--live', '-l'):
                use_testnet = False

        bot = BinanceBot(API_KEY, API_SECRET, use_testnet)
        bot.run()
    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        logger.critical(f"Critical error: {e}")
