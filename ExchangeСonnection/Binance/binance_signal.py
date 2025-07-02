import os
import time
import random
import decimal
from typing import Dict, Optional, Tuple, Union

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from web3 import Web3
from uniswap import Uniswap
import ccxt

load_dotenv()

class UnifiedCryptoTrader:
    """Універсальний крипто-трейдер для конвертації токенів через Binance та Uniswap"""
    
    # Константи класу
    DEFAULT_DELAYS = {'pre_conversion': 2, 'binance_step': 3, 'uniswap_processing': 2, 'approval_wait': 5, 'test_simulation': 1}
    DEFAULT_TEST_BALANCE = {'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 'BNB': 50.0, 'ADA': 5000.0, 'DOT': 100.0, 'SOL': 25.0, 'USDC': 5000.0}
    DEFAULT_RATES = {'USDT': 1.0, 'BTC': 104000.0, 'ETH': 2500.0, 'BNB': 600.0, 'ADA': 0.7, 'DOT': 8.0, 'SOL': 200.0, 'USDC': 1.0}
    
    def __init__(self, testnet: bool = True, use_uniswap: bool = False):
        self.testnet = testnet
        self.use_uniswap = use_uniswap
        self.delays = self.DEFAULT_DELAYS.copy()
        self.test_balance = self.DEFAULT_TEST_BALANCE.copy()
        self.rates = self.DEFAULT_RATES.copy()
        
        # Ініціалізація клієнтів
        self.binance_client = None
        self.ccxt_exchange = None
        self.exchange_info_cache = None
        self.web3 = None
        self.uniswap = None
        
        self._initialize_clients()

    def _initialize_clients(self):
        """Ініціалізація всіх клієнтів"""
        self._init_binance()
        self._init_ccxt()
        if self.use_uniswap:
            self._init_uniswap()

    def _init_binance(self):
        api_key, api_secret = os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET')
        if api_key and api_secret and not self.testnet:
            try:
                self.binance_client = Client(api_key, api_secret)
                self.binance_client.ping()
                self._update_binance_info()
                self._log("✅ Binance підключено")
            except Exception as e:
                self._log(f"❌ Помилка Binance: {e}")
        else:
            self._log("🧪 Binance в тестовому режимі")

    def _init_ccxt(self):
        """Ініціалізація CCXT для кращої конвертації"""
        try:
            api_key, api_secret = os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET')
            if api_key and api_secret and not self.testnet:
                self.ccxt_exchange = ccxt.binance({
                    'apiKey': api_key, 'secret': api_secret, 'sandbox': False,
                    'enableRateLimit': True, 'options': {'defaultType': 'spot'}
                })
                self.ccxt_exchange.load_markets()
                self._log("✅ CCXT Binance підключено")
            else:
                self._log("🧪 CCXT в тестовому режимі")
        except Exception as e:
            self._log(f"❌ Помилка CCXT: {e}")
            self.ccxt_exchange = None

    def _init_uniswap(self):
        try:
            infura_url, private_key = os.getenv('INFURA_URL'), os.getenv('ETH_PRIVATE_KEY')
            if self.testnet:
                return self._log("🧪 Uniswap в тестовому режимі")
            if not infura_url or not private_key:
                return self._log("❌ INFURA_URL або ETH_PRIVATE_KEY не знайдено")
            
            self.web3 = Web3(Web3.HTTPProvider(infura_url))
            if not self.web3.is_connected():
                raise Exception("Не вдалося підключитися до Ethereum мережі")
            
            self.address = self.web3.eth.account.from_key(private_key).address
            self.uniswap = Uniswap(address=self.address, private_key=private_key, version=3, provider=infura_url, web3=self.web3)
            
            # Скорочений список токенів (найпопулярніші)
            token_list = {
                'ETH': '0x0000000000000000000000000000000000000000',
                'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
                'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
                'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
                'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
                'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA'
            }
            self.token_addresses = {k: self.web3.to_checksum_address(v) for k, v in token_list.items()}
            
            self._log("✅ Uniswap підключено")
            self._log(f"📍 Адреса гаманця: {self.address}")
        except Exception as e:
            self._log(f"❌ Помилка Uniswap: {e}")
            self.web3 = self.uniswap = None

    def _log(self, message):
        """Централізоване логування"""
        print(message)

    def _update_binance_info(self):
        if not self.binance_client:
            return
        try:
            self.exchange_info_cache = self.binance_client.get_exchange_info()
            tickers = self.binance_client.get_all_tickers()
            new_rates = {'USDT': 1.0}
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith('USDT') and len(symbol) > 4:
                    asset = symbol[:-4]
                    new_rates[asset] = float(ticker['price'])
            self.rates.update(new_rates)
            self._log("✅ Binance дані оновлено")
        except Exception as e:
            self._log(f"❌ Помилка оновлення Binance: {e}")

    def get_balance(self, asset: str) -> float:
        asset = asset.upper()
        if self.testnet:
            return self.test_balance.get(asset, 0.0)
        if self.binance_client:
            try:
                account = self.binance_client.get_account()
                for balance in account['balances']:
                    if balance['asset'] == asset:
                        return float(balance['free'])
            except Exception as e:
                print(f"❌ Помилка Binance балансу: {e}")
        if self.web3 and self.address:
            try:
                if asset == 'ETH':
                    balance = self.web3.eth.get_balance(self.address)
                    return float(self.web3.from_wei(balance, 'ether'))
                elif asset in self.token_addresses:
                    token_address = self.token_addresses[asset]
                    if token_address != '0x0000000000000000000000000000000000000000':
                        balance = self.uniswap.get_token_balance(token_address)
                        return balance
            except Exception as e:
                print(f"❌ Помилка {asset} балансу: {e}")
        return 0.0

    def _get_symbol_filters(self, symbol: str) -> dict:
        filters = {}
        if not self.exchange_info_cache:
            return filters
        for symbol_info in self.exchange_info_cache.get('symbols', []):
            if symbol_info['symbol'] == symbol:
                for filter_info in symbol_info.get('filters', []):
                    if filter_info['filterType'] == 'LOT_SIZE':
                        filters['minQty'] = float(filter_info['minQty'])
                        filters['maxQty'] = float(filter_info['maxQty'])
                        filters['stepSize'] = float(filter_info['stepSize'])
                    elif filter_info['filterType'] == 'MIN_NOTIONAL':
                        filters['minNotional'] = float(filter_info['minNotional'])
                    elif filter_info['filterType'] == 'PRICE_FILTER':
                        filters['tickSize'] = float(filter_info['tickSize'])
                break
        return filters

    def _format_amount(self, amount: float, symbol: str = None, round_down: bool = False) -> str:
        if symbol and self.exchange_info_cache:
            filters = self._get_symbol_filters(symbol)
            if 'stepSize' in filters:
                step_size = filters['stepSize']
                if 'minQty' in filters and amount < filters['minQty']:
                    return "0"
                
                if round_down:
                    decimal.getcontext().prec = 50
                    amount_decimal = decimal.Decimal(str(amount))
                    step_decimal = decimal.Decimal(str(step_size))
                    # Агресивне округлення вниз для повної конвертації
                    max_steps = int(amount_decimal / step_decimal)
                    # Додатково зменшуємо на 1 step_size для гарантії
                    if max_steps > 0:
                        max_steps -= 1
                    steps = max_steps
                else:
                    steps = round(amount / step_size)
                    
                formatted_amount = steps * step_size
                if 'minQty' in filters and formatted_amount < filters['minQty']:
                    return "0"
                
                if step_size >= 1:
                    return str(int(formatted_amount))
                else:
                    step_str = f"{step_size:.20f}".rstrip('0').rstrip('.')
                    decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
                    if decimals > 0:
                        result = f"{formatted_amount:.{decimals}f}"
                        if decimals > 8:
                            result = f"{formatted_amount:.8f}".rstrip('0').rstrip('.')
                        return result
                    else:
                        return str(int(formatted_amount))
        
        formatted = f"{amount:.8f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0"

    def _binance_convert(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        try:
            if (from_asset == 'USDC' and to_asset == 'USDT') or (from_asset == 'USDT' and to_asset == 'USDC'):
                return self._convert_stablecoins(from_asset, to_asset, amount, is_max)
            symbol = f"{from_asset}{to_asset}"
            reverse_symbol = f"{to_asset}{from_asset}"
            available_symbols = {s['symbol'] for s in self.exchange_info_cache.get('symbols', []) if s['status'] == 'TRADING'}
            if not self._check_min_notional(from_asset, to_asset, amount):
                return self._binance_convert_via_usdt(from_asset, to_asset, amount, is_max)
            convert_amount = self.get_balance(from_asset) if is_max else amount
            if convert_amount <= 0:
                return False
            if symbol in available_symbols:
                # Враховуємо комісію та step_size для зменшення залишку
                if is_max:
                    filters = self._get_symbol_filters(symbol)
                    step_size = filters.get('stepSize', 0)
                    if step_size > 0:
                        convert_amount = self.get_max_tradeable(convert_amount, step_size, 0.001)
                        print(f"🔧 Макс. торгова сума з комісією: {convert_amount}")
                
                formatted_amount = self._format_amount(convert_amount, symbol, round_down=is_max)
                if formatted_amount == "0":
                    return False
                order = self.binance_client.order_market_sell(symbol=symbol, quantity=formatted_amount)
            elif reverse_symbol in available_symbols:
                quote_amount = convert_amount * self.rates.get(from_asset, 1.0)
                order = self.binance_client.order_market_buy(symbol=reverse_symbol, quoteOrderQty=self._format_amount(quote_amount))
            else:
                return self._binance_convert_via_usdt(from_asset, to_asset, amount, is_max)
            print(f"✅ Binance успішно: {order['orderId']}")
            return True
        except BinanceAPIException as e:
            if "NOTIONAL" in str(e):
                return self._binance_convert_via_usdt(from_asset, to_asset, amount, is_max)
            elif "-2010" in str(e) or "not permitted" in str(e).lower():
                print(f"⚠️ Пряма пара {from_asset}/{to_asset} недоступна для акаунта, спроба через USDT...")
                return self._binance_convert_via_usdt(from_asset, to_asset, amount, is_max)
            print(f"❌ Binance помилка: {e}")
            return False

    def _check_min_notional(self, from_asset: str, to_asset: str, amount: float) -> bool:
        try:
            symbol = f"{from_asset}{to_asset}"
            reverse_symbol = f"{to_asset}{from_asset}"
            check_symbol = None
            if self.exchange_info_cache:
                available_symbols = {s['symbol'] for s in self.exchange_info_cache['symbols'] if s['status'] == 'TRADING'}
                check_symbol = symbol if symbol in available_symbols else (reverse_symbol if reverse_symbol in available_symbols else None)
            if not check_symbol:
                return False
            filters = self._get_symbol_filters(check_symbol)
            min_notional = filters.get('minNotional', 0)
            if min_notional <= 0:
                return True
            order_value = amount if from_asset == 'USDT' else amount * self.rates.get(from_asset, 1.0)
            return order_value >= min_notional
        except:
            return True

    def _check_min_notional_for_symbol(self, symbol: str, amount: float, asset: str) -> bool:
        try:
            filters = self._get_symbol_filters(symbol)
            min_notional = filters.get('minNotional', 0)
            if min_notional <= 0:
                return True
            order_value = amount if asset == 'USDT' else amount * self.rates.get(asset, 1.0)
            return order_value >= min_notional
        except:
            return True

    def _get_min_notional_for_symbol(self, symbol: str) -> float:
        try:
            filters = self._get_symbol_filters(symbol)
            return filters.get('minNotional', 10.0)
        except:
            return 10.0

    def _binance_convert_via_usdt(self, from_asset: str, to_asset: str, amount: float, is_max: bool) -> bool:
        try:
            if (from_asset == 'USDC' and to_asset == 'USDT') or (from_asset == 'USDT' and to_asset == 'USDC'):
                return self._convert_stablecoins(from_asset, to_asset, amount, is_max)
            usdt_symbol, target_symbol = f"{from_asset}USDT", f"{to_asset}USDT"
            available_symbols = {s['symbol'] for s in self.exchange_info_cache.get('symbols', []) if s['status'] == 'TRADING'}
            if usdt_symbol not in available_symbols or target_symbol not in available_symbols:
                print(f"❌ Пари недоступні: {usdt_symbol}, {target_symbol}")
                return False
            step1_amount = self.get_balance(from_asset) if is_max else amount
            if not self._check_min_notional_for_symbol(usdt_symbol, step1_amount, from_asset):
                print(f"💰 Сума занадто мала: ${step1_amount * self.rates.get(from_asset, 1.0):.2f}")
                return False
            formatted_amount = self._format_amount(step1_amount, usdt_symbol, round_down=is_max)
            order1 = self.binance_client.order_market_sell(symbol=usdt_symbol, quantity=formatted_amount)
            time.sleep(2)
            usdt_balance = self.get_balance('USDT')
            if usdt_balance <= 0 or not self._check_min_notional_for_symbol(target_symbol, usdt_balance, 'USDT'):
                print(f"💰 USDT залишається на балансі: {usdt_balance:.6f}")
                return False
            order2 = self.binance_client.order_market_buy(symbol=target_symbol, quoteOrderQty=self._format_amount(usdt_balance))
            print(f"✅ Конвертація через USDT: {order1['orderId']}, {order2['orderId']}")
            return True
        except BinanceAPIException as e:
            if "NOTIONAL" in str(e):
                print("💰 Сума занадто мала навіть для USDT")
            print(f"❌ Binance помилка: {e}")
            return False

    def _convert_stablecoins(self, from_asset: str, to_asset: str, amount: float, is_max: bool) -> bool:
        try:
            symbol = f"{from_asset}{to_asset}"
            reverse_symbol = f"{to_asset}{from_asset}"
            available_symbols = {s['symbol'] for s in self.exchange_info_cache.get('symbols', []) if s['status'] == 'TRADING'}
            convert_amount = self.get_balance(from_asset) if is_max else amount
            if symbol in available_symbols:
                if not self._check_min_notional_for_symbol(symbol, convert_amount, from_asset):
                    return False
                formatted_amount = self._format_amount(convert_amount, symbol)
                order = self.binance_client.order_market_sell(symbol=symbol, quantity=formatted_amount)
            elif reverse_symbol in available_symbols:
                quote_amount = convert_amount * self.rates.get(from_asset, 1.0)
                if not self._check_min_notional_for_symbol(reverse_symbol, quote_amount, from_asset):
                    return False
                order = self.binance_client.order_market_buy(symbol=reverse_symbol, quoteOrderQty=self._format_amount(quote_amount))
            else:
                return False
            print(f"✅ Stablecoin: {order['orderId']}")
            return True
        except BinanceAPIException as e:
            print(f"❌ Stablecoin помилка: {e}")
            return False

    def _round_to_step_size(self, amount: float, step_size: float) -> float:
        """Округлює кількість вниз до step_size для зменшення залишку"""
        if step_size <= 0:
            return amount
        return step_size * int(amount / step_size)
    
    def get_max_tradeable(self, balance: float, step_size: float, fee_percent: float = 0.001) -> float:
        """Обчислює максимальну торгову кількість з урахуванням комісії та step_size"""
        if balance <= 0 or step_size <= 0:
            return 0
        tradeable = balance * (1 - fee_percent)
        return self._round_to_step_size(tradeable, step_size)
    
    def _get_ccxt_step_size(self, symbol: str) -> float:
        """Отримує step_size з CCXT для точного округлення"""
        try:
            if not self.ccxt_exchange or symbol not in self.ccxt_exchange.markets:
                return 0
            market = self.ccxt_exchange.markets[symbol]
            precision = market.get('precision', {}).get('amount', 8)
            return 10 ** (-precision)
        except:
            return 0

    def _ccxt_convert(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        """Конвертація через CCXT для більш точних обчислень"""
        try:
            if not self.ccxt_exchange:
                return False
                
            # Перевіряємо статус біржі
            try:
                status = self.ccxt_exchange.fetch_status()
                if status.get('status') != 'ok':
                    print("⚠️ Binance API тимчасово недоступна")
                    return False
            except:
                pass
                
            symbol = f"{from_asset}/{to_asset}"
            reverse_symbol = f"{to_asset}/{from_asset}"
            trade_symbol = order_side = None
            trade_amount = amount
            
            if symbol in self.ccxt_exchange.markets:
                market_info = self.ccxt_exchange.markets[symbol]
                if not market_info.get('active', True):
                    print(f"⚠️ Ринок {symbol} тимчасово недоступний")
                    return self._ccxt_convert_via_usdt(from_asset, to_asset, amount, is_max)
                trade_symbol, order_side = symbol, 'sell'
                
            elif reverse_symbol in self.ccxt_exchange.markets:
                market_info = self.ccxt_exchange.markets[reverse_symbol]
                if not market_info.get('active', True):
                    print(f"⚠️ Ринок {reverse_symbol} тимчасово недоступний")
                    return self._ccxt_convert_via_usdt(from_asset, to_asset, amount, is_max)
                trade_symbol, order_side = reverse_symbol, 'buy'
            else:
                return self._ccxt_convert_via_usdt(from_asset, to_asset, amount, is_max)
            
            market_info = self.ccxt_exchange.markets[trade_symbol]
            min_amount = market_info['limits']['amount']['min'] or 0
            min_cost = market_info['limits']['cost']['min'] or 0
            
            if order_side == 'sell':
                step_size = self._get_ccxt_step_size(trade_symbol)
                if step_size > 0:
                    if is_max:
                        trade_amount = self.get_max_tradeable(trade_amount, step_size, 0.001)
                        print(f"🔧 Макс. торгова сума з комісією: {trade_amount}")
                    else:
                        trade_amount = self._round_to_step_size(trade_amount, step_size)
                        print(f"🔧 Округлено до step_size: {trade_amount}")
                
                if trade_amount < min_amount:
                    print(f"❌ Кількість {trade_amount} менша за мінімальну {min_amount}")
                    return False
                order = self.ccxt_exchange.create_market_sell_order(trade_symbol, trade_amount)
            else:
                quote_amount = amount
                if min_cost > 0 and quote_amount < min_cost:
                    print(f"❌ Сума {quote_amount} менша за мінімальну {min_cost}")
                    return False
                order = self.ccxt_exchange.create_order(
                    symbol=trade_symbol, type='market', side='buy', amount=None, 
                    price=None, params={'quoteOrderQty': quote_amount}
                )
                
            print(f"✅ CCXT успішно: {order['id']}")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'market is closed' in error_msg:
                print("⚠️ Ринок тимчасово недоступний, спроба через USDT...")
                return self._ccxt_convert_via_usdt(from_asset, to_asset, amount, is_max)
            elif 'insufficient balance' in error_msg:
                print("❌ Недостатньо коштів на балансі")
            elif 'minimum notional' in error_msg or 'min notional' in error_msg:
                print(f"❌ Сума занадто мала для CCXT ({amount} {from_asset})")
            else:
                print(f"❌ CCXT помилка: {e}")
            return False
    
    def _ccxt_convert_via_usdt(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        """Конвертація через USDT за допомогою CCXT"""
        try:
            if not self.ccxt_exchange:
                return False
                
            from_symbol = f"{from_asset}/USDT"
            if from_symbol not in self.ccxt_exchange.markets:
                print(f"❌ Ринок {from_symbol} недоступний")
                return False
                
            market_info = self.ccxt_exchange.markets[from_symbol]
            if not market_info.get('active', True):
                print(f"⚠️ Ринок {from_symbol} тимчасово недоступний")
                return False
                
            # Округлюємо кількість до step_size для зменшення залишку
            step_size = self._get_ccxt_step_size(from_symbol)
            if step_size > 0:
                if is_max:
                    amount = self.get_max_tradeable(amount, step_size, 0.001)
                    print(f"🔧 Макс. торгова сума з комісією: {amount}")
                else:
                    amount = self._round_to_step_size(amount, step_size)
                    print(f"🔧 Округлено до step_size: {amount}")
            
            order1 = self.ccxt_exchange.create_market_sell_order(from_symbol, amount)
            print(f"🔸 Крок 1: {from_asset} → USDT")
            
            time.sleep(2)
            
            # Отримуємо баланс USDT (з кількома спробами)
            usdt_balance = 0
            for attempt in range(3):
                try:
                    balance = self.ccxt_exchange.fetch_balance()
                    usdt_balance = balance['USDT']['free']
                    if usdt_balance > 0:
                        break
                    time.sleep(1)
                except:
                    if attempt == 2:
                        print("❌ Не вдалося отримати баланс USDT")
                        return False
                    time.sleep(1)
            
            if usdt_balance <= 0:
                print("❌ Недостатньо USDT після першого кроку")
                return False
            
            to_symbol = f"{to_asset}/USDT"
            if to_symbol not in self.ccxt_exchange.markets:
                print(f"❌ Ринок {to_symbol} недоступний")
                return False
                
            market_info = self.ccxt_exchange.markets[to_symbol]
            if not market_info.get('active', True):
                print(f"⚠️ Ринок {to_symbol} тимчасово недоступний")
                return False
                
            order2 = self.ccxt_exchange.create_order(
                symbol=to_symbol,
                type='market',
                side='buy',
                amount=None,
                price=None,
                params={'quoteOrderQty': usdt_balance}
            )
            print(f"🔸 Крок 2: USDT → {to_asset}")
            
            print(f"✅ CCXT конвертація через USDT: {order1['id']}, {order2['id']}")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'market is closed' in error_msg or 'market closed' in error_msg:
                print("⚠️ Один з ринків USDT тимчасово недоступний")
            elif 'insufficient balance' in error_msg:
                print("❌ Недостатньо коштів для конвертації через USDT")
            else:
                print(f"❌ CCXT конвертація через USDT помилка: {e}")
            return False

    def convert_dust_to_bnb(self):
        """Конвертує всі доступні дрібні залишки в BNB"""
        if not self.binance_client:
            print("❌ Binance клієнт не доступний")
            return False
            
        # Розширений список активів для конвертації пилу
        potential_dust_assets = ['PEPE', 'SHIB', 'USDC', 'DOGE', 'ADA', 'DOT', 'SOL', 'MATIC', 'LTC', 'LINK', 'UNI', 
                                'TRX', 'AVAX', 'ATOM', 'XRP', 'XLM', 'ALGO', 'VET', 'FTM', 'NEAR', 'SAND', 'MANA',
                                'GRT', 'ENJ', 'CHZ', 'BAT', 'ZIL', 'HBAR', 'THETA', 'ONE', 'IOTA', 'EOS']
        converted_assets = []
        failed_assets = []
        
        print("🔍 Перевірка балансів для конвертації пилу...")
        
        try:
            account = self.binance_client.get_account()
            account_balances = {balance['asset']: float(balance['free']) for balance in account['balances'] if float(balance['free']) > 0}
        except Exception as e:
            print(f"❌ Не вдалося отримати баланси: {e}")
            return False
        
        for asset in potential_dust_assets:
            if asset in account_balances and account_balances[asset] > 0:
                asset_usd_value = account_balances[asset] * self.rates.get(asset, 1.0)
                if asset_usd_value < 1.0:
                    print(f"🔄 Спроба конвертації {asset} (${asset_usd_value:.4f})...")
                    if self._convert_small_dust_to_bnb(asset):
                        converted_assets.append(asset)
                    else:
                        failed_assets.append(asset)
        
        if converted_assets:
            print(f"✅ Успішно конвертовано пил: {', '.join(converted_assets)}")
        if failed_assets:
            print(f"⚠️ Не вдалося конвертувати: {', '.join(failed_assets)}")
        if not converted_assets and not failed_assets:
            print("💡 Немає дрібних залишків для конвертації")
            
        return len(converted_assets) > 0

    def _convert_small_dust_to_bnb(self, asset: str) -> bool:
        """Конвертує дрібні залишки в BNB через Binance dust transfer"""
        try:
            if not self.binance_client:
                return False
            
            try:
                dust_log = self.binance_client.get_dust_log()
                eligible_assets = set()
                if 'results' in dust_log:
                    for result in dust_log['results']:
                        if 'details' in result:
                            for detail in result['details']:
                                if 'asset' in detail:
                                    eligible_assets.add(detail['asset'])
                
                if eligible_assets and asset not in eligible_assets:
                    print(f"⚠️ {asset} не доступний для конвертації пилу")
                    return False
            except:
                dust_eligible_assets = ['PEPE', 'SHIB', 'USDC', 'DOGE', 'ADA', 'DOT', 'SOL', 'MATIC']
                if asset not in dust_eligible_assets:
                    print(f"⚠️ {asset} може не підтримувати конвертацію пилу")
                
            dust_result = self.binance_client.transfer_dust(asset=[asset])
            print(f"✅ Пил {asset} конвертовано в BNB")
            return True
        except BinanceAPIException as e:
            error_code = str(e)
            if "-1102" in error_code:
                print(f"⚠️ {asset} не доступний для конвертації пилу на цьому акаунті")
            elif "-2010" in error_code:
                print(f"⚠️ Конвертація пилу {asset} тимчасово недоступна")
            else:
                print(f"❌ Помилка конвертації пилу {asset}: {e}")
            return False
        except Exception as e:
            print(f"❌ Не вдалося конвертувати пил {asset}: {e}")
            return False

    def _get_token_decimals(self, token_address: str) -> int:
        decimals_map = {
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': 18,  # WETH
            '0xdAC17F958D2ee523a2206206994597C13D831ec7': 6,   # USDT
            '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48': 6,   # USDC (правильна адреса)
            '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599': 8,   # WBTC
            '0x6B175474E89094C44Da98b954EedeAC495271d0F': 18,  # DAI
        }
        return decimals_map.get(token_address, 18)

    def _uniswap_convert(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        if self.testnet:
            return self._test_uniswap_convert(from_asset, to_asset, amount, is_max)
        try:
            if not self.uniswap or not self.web3:
                print("❌ Uniswap або Web3 не підключено")
                return False
            from_asset, to_asset = from_asset.upper(), to_asset.upper()
            
            # Перевірка балансу ETH для газу
            eth_balance = self.web3.eth.get_balance(self.address)
            eth_balance_ether = float(self.web3.from_wei(eth_balance, 'ether'))
            required_eth = 0.015  # Збільшуємо мінімальний баланс для надійності
            
            if eth_balance_ether < required_eth:
                print(f"❌ Недостатньо ETH для газу: {eth_balance_ether:.6f} ETH")
                print(f"💡 Потрібно мінімум {required_eth} ETH для Uniswap операцій")
                print(f"💰 Поповніть ETH баланс або використайте Binance")
                return False
            
            # Спеціальна обробка для BNB через BSC мережу
            if from_asset == 'BNB' or to_asset == 'BNB':
                print("⚠️ BNB через Uniswap не підтримується (потрібна BSC мережа)")
                return False
                
            if from_asset == 'ETH': from_asset = 'WETH'
            if to_asset == 'ETH': to_asset = 'WETH'
            from_token = self.token_addresses.get(from_asset)
            to_token = self.token_addresses.get(to_asset)
            
            if not from_token or not to_token:
                print(f"❌ Токени {from_asset}/{to_asset} не підтримуються")
                return False
                
            # Перевірка існування пари на Uniswap
            try:
                # Спроба отримати пул для перевірки існування пари
                if hasattr(self.uniswap, 'get_pool_info'):
                    pool_info = self.uniswap.get_pool_info(from_token, to_token, fee=3000)
                    if not pool_info:
                        print(f"❌ Пара {from_asset}/{to_asset} не знайдена на Uniswap")
                        return False
            except Exception as pool_error:
                print(f"⚠️ Неможливо перевірити пару {from_asset}/{to_asset}: {pool_error}")
                # Продовжуємо виконання, але з обережністю
                
            if is_max:
                if from_asset == 'WETH':
                    balance_wei = self.web3.eth.get_balance(self.address)
                    amount_wei = max(0, balance_wei - self.web3.to_wei(0.01, 'ether'))
                else:
                    try:
                        balance = self.uniswap.get_token_balance(from_token)
                        decimals = self._get_token_decimals(from_token)
                        amount_wei = int(balance * (10 ** decimals))
                    except Exception as balance_error:
                        print(f"❌ Помилка отримання балансу {from_asset}: {balance_error}")
                        return False
            else:
                if from_asset == 'WETH':
                    amount_wei = self.web3.to_wei(amount, 'ether')
                else:
                    decimals = self._get_token_decimals(from_token)
                    amount_wei = int(amount * (10 ** decimals))
                    
            if amount_wei <= 0:
                print(f"❌ Недостатня кількість для конвертації: {amount_wei}")
                return False
                
            print(f"🔄 Uniswap: {from_asset} → {to_asset}")
            
            # Approve токена якщо потрібно
            if from_asset != 'WETH':
                try:
                    print("🔓 Надання дозволу токена...")
                    # Використовуємо стандартний ERC-20 approve через Web3
                    from web3 import Web3
                    erc20_abi = [
                        {
                            "constant": False,
                            "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
                            "name": "approve",
                            "outputs": [{"name": "", "type": "bool"}],
                            "type": "function"
                        }
                    ]
                    
                    token_contract = self.web3.eth.contract(address=from_token, abi=erc20_abi)
                    uniswap_router = "0xE592427A0AEce92De3Edee1F18E0157C05861564"  # Uniswap V3 Router
                    
                    approve_tx = token_contract.functions.approve(
                        uniswap_router, amount_wei * 2
                    ).build_transaction({
                        'from': self.address,
                        'gas': 100000,
                        'gasPrice': self.web3.to_wei('20', 'gwei'),
                        'nonce': self.web3.eth.get_transaction_count(self.address)
                    })
                    
                    signed_tx = self.web3.eth.account.sign_transaction(approve_tx, private_key=os.getenv('ETH_PRIVATE_KEY'))
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print("✅ Дозвіл токена надано")
                    time.sleep(self.delays['approval_wait'])
                except Exception as e:
                    print(f"⚠️ Помилка approve токена: {e}")
                    # Продовжуємо без approve - можливо він вже є
                    
            print("🔄 Виконання swap...")
            try:
                # Використовуємо більш стабільний метод make_trade з додатковими параметрами
                tx_hash = self.uniswap.make_trade(
                    input_token=from_token, 
                    output_token=to_token, 
                    qty=amount_wei,
                    recipient=self.address, 
                    slippage=1.0,  # Збільшуємо slippage для більшої стабільності
                    fee=3000  # Стандартна комісія 0.3%
                )
                
                print("⏳ Очікування підтвердження транзакції...")
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
                
                if receipt and receipt.status == 1:
                    print("✅ Uniswap успішно")
                    return True
                else:
                    print("❌ Транзакція не вдалася")
                    return False
                    
            except (TypeError, IndexError) as te:
                if "tuple index out of range" in str(te):
                    print("❌ Помилка маршрутизації Uniswap - пара може не існувати або недостатня ліквідність")
                    print(f"💡 Порада: Перевірте чи є ліквідність для пари {from_asset}/{to_asset}")
                else:
                    print(f"❌ Помилка типу в Uniswap: {te}")
                return False
            except Exception as trade_error:
                print(f"❌ Помилка виконання swap: {trade_error}")
                return False
                
        except Exception as e:
            print(f"❌ Uniswap помилка: {e}")
            return False

    def _test_uniswap_convert(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        from_asset, to_asset = from_asset.upper(), to_asset.upper()
        supported_tokens = ['ETH', 'WETH', 'BNB', 'USDT', 'USDC', 'WBTC', 'DAI', 'UNI', 'LINK', 'PEPE', 'SHIB', 'DOGE', 'MATIC', 'CRO', 'LDO', 'AAVE', 'COMP']
        if from_asset not in supported_tokens or to_asset not in supported_tokens:
            print(f"❌ Токени {from_asset}/{to_asset} не підтримуються в Uniswap")
            return False
        if to_asset == 'SIGN':
            print(f"❌ Uniswap помилка: tuple index out of range")
            return False
        if from_asset == 'ETH': from_asset = 'WETH'
        if to_asset == 'ETH': to_asset = 'WETH'
        convert_amount = self.get_balance(from_asset) if is_max else amount
        if convert_amount <= 0:
            print(f"❌ Недостатньо {from_asset}")
            return False
        
        base_gas_fee = 15.0
        network_congestion = 1.0
        gas_fee_usd = base_gas_fee * (0.8 if from_asset == 'WETH' or to_asset == 'WETH' else 1.2) * network_congestion
        gas_fee_in_asset = gas_fee_usd / self.rates.get('ETH' if from_asset == 'WETH' else from_asset, 2500.0 if from_asset == 'WETH' else 1.0)
        
        if convert_amount <= gas_fee_in_asset:
            print(f"❌ Gas комісія (${gas_fee_usd:.0f}) перевищує суму конвертації")
            return False
        
        slippage_rates = {
            ('WETH', 'USDT'): 0.1, ('WETH', 'USDC'): 0.1, ('USDT', 'USDC'): 0.05,
            ('WETH', 'WBTC'): 0.2, ('PEPE', 'WETH'): 1.0, ('SHIB', 'WETH'): 0.8,
        }
        pair_key = (from_asset, to_asset)
        reverse_key = (to_asset, from_asset)
        slippage_percent = slippage_rates.get(pair_key, slippage_rates.get(reverse_key, 0.5))
        
        net_amount = convert_amount - gas_fee_in_asset
        after_trading_fee = net_amount * 0.997  # 0.3% trading fee
        rate = self.rates.get(from_asset, 1.0) / self.rates.get(to_asset, 1.0)
        before_slippage = after_trading_fee * rate
        final_amount = before_slippage * (1 - slippage_percent/100)
        
        print(f"🦄 Uniswap конвертація успішна: {final_amount:.8f} {to_asset}")
        print("⏳ Обробка транзакції...")
        time.sleep(1)
        
        import random
        if random.random() < 0.01:
            print("❌ Транзакція не вдалася")
            self.test_balance[from_asset] -= gas_fee_in_asset
            return False
        
        self.test_balance[from_asset] -= convert_amount
        self.test_balance[to_asset] = self.test_balance.get(to_asset, 0) + final_amount
        print("✅ Uniswap тестова конвертація виконана!")
        return True

    def convert(self, from_asset: str, to_asset: str, amount):
        from_asset = from_asset.upper()
        to_asset = to_asset.upper()
        if from_asset == to_asset:
            print("❌ Однакові токени")
            return False
        initial_balance = self.get_balance(from_asset)
        if initial_balance <= 0:
            print(f"❌ Немає {from_asset}")
            return False
        is_max = str(amount).lower() == 'max'
        convert_amount = initial_balance if is_max else float(amount)
        if convert_amount > initial_balance:
            print(f"❌ Недостатньо коштів")
            return False
        print(f"\n💱 Конвертація: {convert_amount:.8f} {from_asset} → {to_asset}")
        print(f"⏳ Підготовка до конвертації...")
        time.sleep(self.delays['pre_conversion'])
        if self.testnet:
            success = self._test_convert_realistic(from_asset, to_asset, convert_amount, is_max)
        else:
            success = False
            
            # Спеціальна логіка для BTC конвертації
            if from_asset == 'BTC' or to_asset == 'BTC':
                # Для BTC спочатку пробуємо Uniswap з WBTC мапінгом
                uniswap_from = 'WBTC' if from_asset == 'BTC' else from_asset
                uniswap_to = 'WBTC' if to_asset == 'BTC' else to_asset
                
                if (self.use_uniswap and uniswap_from in self.token_addresses 
                    and uniswap_to in self.token_addresses):
                    print("🦄 Спроба конвертації BTC через Uniswap (WBTC)...")
                    
                    # Перевіряємо ETH баланс заздалегідь
                    if self.web3:
                        eth_balance = self.web3.eth.get_balance(self.address)
                        eth_balance_ether = float(self.web3.from_wei(eth_balance, 'ether'))
                        if eth_balance_ether < 0.015:
                            print(f"⚠️ Недостатньо ETH для газу ({eth_balance_ether:.6f} ETH)")
                            print("🔄 Автоматичне переключення на Binance...")
                        else:
                            success = self._uniswap_convert(uniswap_from, uniswap_to, convert_amount, is_max)
                            if success:
                                print(f"✅ BTC успішно конвертовано через Uniswap як WBTC")
                
                # Якщо Uniswap не вдався, пробуємо Binance/CCXT
                if not success:
                    if self.ccxt_exchange:
                        print("🔧 Спроба конвертації BTC через CCXT...")
                        success = self._ccxt_convert(from_asset, to_asset, convert_amount, is_max)
                    
                    if not success and self.binance_client:
                        print("🔶 Спроба конвертації BTC через Binance...")
                        success = self._binance_convert(from_asset, to_asset, convert_amount, is_max)
            else:
                # Для інших токенів стандартна логіка
                uniswap_from = from_asset
                uniswap_to = to_asset
                
                if (self.use_uniswap and uniswap_from in self.token_addresses 
                    and uniswap_to in self.token_addresses):
                    print("🦄 Спроба конвертації через Uniswap...")
                    success = self._uniswap_convert(uniswap_from, uniswap_to, convert_amount, is_max)
                
                if not success and self.ccxt_exchange:
                    print("🔧 Спроба конвертації через CCXT...")
                    success = self._ccxt_convert(from_asset, to_asset, convert_amount, is_max)
                    
                if not success and self.binance_client:
                    print("🔶 Спроба конвертації через Binance...")
                    success = self._binance_convert(from_asset, to_asset, convert_amount, is_max)
                
        if success:
            self._show_conversion_remainder(from_asset, initial_balance, is_max)
        else:
            convert_value_usd = convert_amount * self.rates.get(from_asset, 1.0)
            print(f"\n❌ Конвертація не вдалася!")
            print(f"💰 Сума конвертації: {convert_amount:.8f} {from_asset} (~${convert_value_usd:.2f})")
            
            if convert_value_usd < 10.0:
                print(f"⚠️ Причина: Сума занадто мала (мінімум ~$10.00 для Binance)")
                print(f"💡 Рекомендації:")
                print(f"   • Накопичте більше {from_asset}")
                print(f"   • Або конвертуйте пил командою 'dust' (якщо доступно)")
            else:
                print(f"⚠️ Можливі причини:")
                print(f"   • Торгова пара недоступна")
                print(f"   • Проблеми з мережею")
                print(f"   • Тимчасові обмеження біржі")
                
        return success

    def _show_conversion_remainder(self, from_asset: str, initial_balance: float, is_max: bool):
        current_balance = self.get_balance(from_asset)
        if current_balance > 0.00000001:
            remainder_amount = current_balance
            remainder_percentage = (remainder_amount / initial_balance) * 100 if initial_balance > 0 else 0
            converted_amount = initial_balance - remainder_amount
            converted_percentage = (converted_amount / initial_balance) * 100 if initial_balance > 0 else 0
            print(f"\n📊 Результат конвертації:")
            print(f"🔸 Конвертовано: {converted_amount:.8f} {from_asset} ({converted_percentage:.2f}%)")
            print(f"⚠️ Залишок: {remainder_amount:.8f} {from_asset} ({remainder_percentage:.2f}%)")
            
            if is_max and remainder_percentage < 5 and from_asset not in ['BNB', 'USDT', 'USDC']:
                remainder_usd = remainder_amount * self.rates.get(from_asset, 1.0)
                if remainder_usd < 1.0:
                    print(f"💡 Залишок малий (${remainder_usd:.3f}). Хочете конвертувати пил в BNB? (y/n): ", end="")
                    try:
                        dust_choice = input().lower()
                        if dust_choice in ['y', 'yes', 'так', 'да']:
                            if self._convert_small_dust_to_bnb(from_asset):
                                print("✅ Пил успішно конвертовано в BNB")
                    except:
                        pass
        else:
            print(f"\n📊 Результат конвертації:")
            print(f"✅ Конвертовано повністю: {initial_balance:.8f} {from_asset} (100.00%)")

    def _test_convert_realistic(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        success = False
        if self.use_uniswap:
            uniswap_tokens = ['ETH', 'WETH', 'USDT', 'USDC', 'WBTC', 'DAI', 'UNI', 'LINK', 'PEPE', 'SHIB']
            if from_asset in uniswap_tokens and to_asset in uniswap_tokens:
                success = self._test_uniswap_convert(from_asset, to_asset, amount, is_max)
        if not success:
            if self.use_uniswap:
                print("⏳ Перехід на Binance...")
                time.sleep(self.delays['test_simulation'])
            success = self._test_binance_convert(from_asset, to_asset, amount, is_max)
        return success

    def _test_binance_convert(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        min_notional_usd = 10.0
        order_value_usd = amount * self.rates.get(from_asset, 1.0)
        if order_value_usd < min_notional_usd:
            print(f"❌ Binance помилка: APIError(code=-1013): Filter failure: NOTIONAL")
            return False
        fee = amount * 0.001
        net_amount = amount - fee
        rate = self.rates.get(from_asset, 1.0) / self.rates.get(to_asset, 1.0)
        receive_amount = net_amount * rate
        print(f"🔸 Binance конвертація успішна: {receive_amount:.8f} {to_asset}")
        time.sleep(self.delays['test_simulation'])
        self.test_balance[from_asset] -= amount
        self.test_balance[to_asset] = self.test_balance.get(to_asset, 0) + receive_amount
        print(f"✅ Binance тестова конвертація виконана!")
        return True

    def _test_convert(self, from_asset: str, to_asset: str, amount: float) -> bool:
        fee = amount * 0.001
        net_amount = amount - fee
        rate = self.rates.get(from_asset, 1.0) / self.rates.get(to_asset, 1.0)
        receive_amount = net_amount * rate
        self.test_balance[from_asset] -= amount
        self.test_balance[to_asset] = self.test_balance.get(to_asset, 0) + receive_amount
        print("✅ Тестова конвертація виконана!")
        return True

    def show_balance(self):
        print(f"\n💰 Баланс ({'ТЕСТ' if self.testnet else 'РЕАЛ'})")
        if self.testnet:
            balances = {k: v for k, v in self.test_balance.items() if v > 0}
        else:
            balances = {}
            if self.binance_client:
                try:
                    account = self.binance_client.get_account()
                    for balance in account['balances']:
                        free_balance = float(balance['free'])
                        if free_balance > 0:
                            balances[balance['asset']] = free_balance
                except Exception as e:
                    print(f"❌ Помилка Binance балансу: {e}")
            if self.uniswap:
                try:
                    eth_balance = self.get_balance('ETH')
                    if eth_balance > 0:
                        balances['ETH'] = eth_balance
                    else:
                        # Показуємо ETH баланс навіть якщо він 0 для Uniswap користувачів
                        balances['ETH'] = 0.0
                except Exception as e:
                    print(f"❌ Помилка ETH балансу: {e}")
        total_usd = 0
        for asset, amount in balances.items():
            usd_value = amount * self.rates.get(asset, 1.0)
            total_usd += usd_value
            status_indicator = ""
            if asset == 'ETH' and self.use_uniswap and not self.testnet:
                if amount < 0.015:
                    status_indicator = " ⚠️ (недостатньо для Uniswap газу)"
                else:
                    status_indicator = " ✅ (достатньо для Uniswap)"
            print(f"  {asset}: {amount:,.8f} (~${usd_value:,.2f}){status_indicator}")
        print(f"💵 Загалом: ${total_usd:,.2f}")
        
        # Додаткова інформація для Uniswap користувачів
        if self.use_uniswap and not self.testnet:
            eth_balance = balances.get('ETH', 0)
            if eth_balance < 0.015:
                print(f"\n💡 Рекомендації для Uniswap:")
                print(f"   • Поповніть ETH баланс до мінімум 0.015 ETH")
                print(f"   • Потрібно ~${0.015 * self.rates.get('ETH', 2500):.2f} USD для газу")
                print(f"   • Або використовуйте Binance для конвертації BTC")

    def add_test_balance(self, asset: str, amount: float):
        if not self.testnet:
            print("❌ Тільки для тестового режиму")
            return
        asset = asset.upper()
        self.test_balance[asset] = self.test_balance.get(asset, 0) + amount
        print(f"💰 Додано {amount} {asset}")

    def update_rates(self):
        if self.binance_client:
            self._update_binance_info()
        else:
            print("✅ Курси оновлено (тестовий режим)")


def interactive_converter():
    print("🔄 === УНІВЕРСАЛЬНИЙ КРИПТО КОНВЕРТЕР ===")
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
    platform = input("Платформа (binance/uniswap/both): ").strip().lower()
    use_uniswap = platform in ['uniswap', 'both']
    try:
        trader = UnifiedCryptoTrader(testnet=testnet, use_uniswap=use_uniswap)
    except Exception as e:
        print(f"❌ Помилка ініціалізації: {e}")
        return
    print("\n📋 Команди:")
    print("  balance - показати баланс")
    print("  convert - конвертувати токени")
    print("  dust - конвертувати пил в BNB")
    print("  add - додати тестовий баланс")
    print("  update - оновити курси")
    print("  exit - вихід")
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
            elif command == 'dust':
                if testnet:
                    print("❌ Функція не доступна в тестовому режимі")
                else:
                    print("🗑️ Конвертація пилу в BNB...")
                    trader.convert_dust_to_bnb()
            elif command == 'add':
                if not testnet:
                    print("❌ Тільки для тестового режиму")
                    continue
                asset = input("Токен: ").strip().upper()
                try:
                    amount = float(input("Кількість: "))
                    trader.add_test_balance(asset, amount)
                except ValueError:
                    print("❌ Неправильна кількість")
            elif command == 'convert':
                from_token = input("З токена: ").strip().upper()
                to_token = input("В токен: ").strip().upper()
                if not from_token or not to_token or from_token == to_token:
                    print("❌ Неправильні токени")
                    continue
                balance = trader.get_balance(from_token)
                if balance <= 0:
                    print(f"❌ Немає {from_token}")
                    continue
                print(f"💰 Доступно {from_token}: {balance:,.8f}")
                amount_input = input("Кількість (або 'max' для всього): ").strip()
                if not amount_input:
                    continue
                print(f"\n⚠️  Конвертація {from_token} → {to_token}")
                confirm = input("Підтвердити? (y/n): ")
                if confirm.lower() in ['y', 'yes', 'так', 'да']:
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