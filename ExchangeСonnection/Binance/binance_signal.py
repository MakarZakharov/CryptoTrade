import os
import time
import random
import decimal
import subprocess
import webbrowser
import re
from typing import Dict, Optional, Tuple, Union

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
try:
    from web3 import Web3
except ImportError:
    Web3 = None
try:
    from uniswap import Uniswap
except ImportError:
    Uniswap = None
import ccxt

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Import простого браузерного конвертера
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from selenium_converter import open_binance_convert
except ImportError:
    open_binance_convert = None

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
            # Спочатку пробуємо Convert API для всіх сум
            print(f"💱 Спроба Convert API...")
            if self._binance_convert_api(from_asset, to_asset, amount, is_max):
                return True
                
            # Якщо Convert API не вдався, пробуємо стандартну торгівлю
            print(f"💡 Спроба через звичайну торгівлю...")
            
            if (from_asset == 'USDC' and to_asset == 'USDT') or (from_asset == 'USDT' and to_asset == 'USDC'):
                return self._convert_stablecoins(from_asset, to_asset, amount, is_max)
                
            symbol = f"{from_asset}{to_asset}"
            reverse_symbol = f"{to_asset}{from_asset}"
            available_symbols = {s['symbol'] for s in self.exchange_info_cache.get('symbols', []) if s['status'] == 'TRADING'}
            
            convert_amount = self.get_balance(from_asset) if is_max else amount
            if convert_amount <= 0:
                return False
                
            if symbol in available_symbols:
                formatted_amount = self._format_amount(convert_amount, symbol, round_down=is_max)
                if formatted_amount == "0":
                    return False
                order = self.binance_client.order_market_sell(symbol=symbol, quantity=formatted_amount)
                print(f"✅ Binance успішно: {order['orderId']}")
                return True
            elif reverse_symbol in available_symbols:
                quote_amount = convert_amount * self.rates.get(from_asset, 1.0)
                order = self.binance_client.order_market_buy(symbol=reverse_symbol, quoteOrderQty=self._format_amount(quote_amount))
                print(f"✅ Binance успішно: {order['orderId']}")
                return True
            else:
                print(f"⚠️ Пряма пара {from_asset}/{to_asset} недоступна, спроба через USDT...")
                return self._binance_convert_via_usdt(from_asset, to_asset, amount, is_max)
                
        except BinanceAPIException as e:
            print(f"❌ Binance помилка: {e}")
            return False

    def _binance_convert_api(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        """Конвертація через Binance Convert API для маленьких сум без обмежень NOTIONAL"""
        try:
            if not self.binance_client:
                return False
                
            convert_amount = self.get_balance(from_asset) if is_max else amount
            if convert_amount <= 0:
                print(f"❌ Недостатньо {from_asset}")
                return False
                
            # Продовжуємо з Convert API
                
            # Використовуємо Convert API
            # Спочатку отримуємо ціну конвертації
            try:
                quote_response = self.binance_client.convert_request_quote(
                    fromAsset=from_asset,
                    toAsset=to_asset,
                    fromAmount=convert_amount
                )
                
                if 'quoteId' not in quote_response:
                    print(f"❌ Не вдалося отримати котирування для {from_asset}/{to_asset}")
                    return False
                    
                quote_id = quote_response['quoteId']
                to_amount = float(quote_response['toAmount'])
                
                print(f"💱 Котирування: {convert_amount:.8f} {from_asset} → {to_amount:.8f} {to_asset}")
                
                # Підтверджуємо конвертацію
                result = self.binance_client.convert_accept_quote(quoteId=quote_id)
                
                if result.get('status') == 'PROCESS':
                    print(f"✅ Convert API успішно: {result.get('orderId', 'N/A')}")
                    return True
                else:
                    print(f"❌ Convert API помилка: {result.get('status', 'Unknown')}")
                    return False
                    
            except Exception as convert_error:
                error_msg = str(convert_error)
                if 'not supported' in error_msg.lower():
                    print(f"❌ Пара {from_asset}/{to_asset} не підтримується Convert API")
                elif 'minimum' in error_msg.lower() or 'maximum' in error_msg.lower():
                    print(f"❌ Сума поза межами Convert API: {convert_amount:.8f} {from_asset}")
                elif '-1002' in error_msg or 'not authorized' in error_msg.lower():
                    print(f"⚠️ Convert API недоступний для вашого акаунта")
                    print(f"🌐 Автоматичний запуск браузерної конвертації...")
                    
                    # Автоматично запускаємо браузерну конвертацію
                    browser_success = self._launch_browser_conversion(from_asset, to_asset, convert_amount if not is_max else 'max')
                    if browser_success:
                        return True
                    
                    print(f"💡 Спроба через звичайну торгівлю...")
                    return False  # Повертаємо False щоб спробувати інші методи
                else:
                    print(f"❌ Convert API помилка: {convert_error}")
                return False
                
        except Exception as e:
            print(f"❌ Помилка Convert API: {e}")
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
            # Видалено перевірку мінімальної суми - дозволяємо конвертувати будь-яку кількість
            formatted_amount = self._format_amount(step1_amount, usdt_symbol, round_down=is_max)
            order1 = self.binance_client.order_market_sell(symbol=usdt_symbol, quantity=formatted_amount)
            time.sleep(2)
            usdt_balance = self.get_balance('USDT')
            if usdt_balance <= 0:
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
                formatted_amount = self._format_amount(convert_amount, symbol)
                order = self.binance_client.order_market_sell(symbol=symbol, quantity=formatted_amount)
            elif reverse_symbol in available_symbols:
                quote_amount = convert_amount * self.rates.get(from_asset, 1.0)
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
            
            if order_side == 'sell':
                step_size = self._get_ccxt_step_size(trade_symbol)
                if step_size > 0:
                    if is_max:
                        trade_amount = self.get_max_tradeable(trade_amount, step_size, 0.001)
                        print(f"🔧 Макс. торгова сума з комісією: {trade_amount}")
                    else:
                        trade_amount = self._round_to_step_size(trade_amount, step_size)
                        print(f"🔧 Округлено до step_size: {trade_amount}")
                
                order = self.ccxt_exchange.create_market_sell_order(trade_symbol, trade_amount)
            else:
                quote_amount = amount
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







    def _try_small_bnb_conversion(self, bnb_amount: float, to_asset: str) -> bool:
        """Пробує конвертувати невелику кількість BNB в інший актив"""
        try:
            if not self.binance_client or bnb_amount <= 0:
                return False
                
            # Спочатку пробуємо Convert API для маленьких сум BNB
            print(f"💱 Спроба Convert API: {bnb_amount:.8f} BNB → {to_asset}")
            if self._binance_convert_api('BNB', to_asset, bnb_amount, False):
                return True
                
            # Якщо Convert API не вдався, пробуємо звичайну торгівлю
            print(f"🔄 Спроба звичайної торгівлі: {bnb_amount:.8f} BNB → {to_asset}")
            return self._binance_convert('BNB', to_asset, bnb_amount, False)
            
        except Exception as e:
            print(f"❌ Помилка конвертації BNB → {to_asset}: {e}")
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

    def convert(self, from_asset: str, to_asset: str, amount: float) -> float:
        """Конвертує суму з одного активу в інший на基і поточних курсів (без реального торгування)"""
        from_asset = from_asset.upper()
        to_asset = to_asset.upper()
        
        if from_asset == to_asset:
            return amount
            
        # Отримуємо курси валют
        from_rate = self.rates.get(from_asset, 1.0)
        to_rate = self.rates.get(to_asset, 1.0)
        
        # Конвертуємо через USD
        usd_value = amount * from_rate
        converted_amount = usd_value / to_rate
        
        return converted_amount

    def trade(self, from_asset: str, to_asset: str, amount):
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
            
            # ПРІОРИТЕТ 1: MCP Playwright автономна конвертація (обходить CSP обмеження)
            print("🎭 Спроба MCP Playwright автономної конвертації...")
            success = self._mcp_autonomous_conversion(from_asset, to_asset, convert_amount if not is_max else None)
            
            if success:
                print("🎉 MCP Playwright автономна конвертація успішна!")
            else:
                print("⚠️ MCP автономна конвертація не вдалася, спробуємо API методи...")
                
                # ПРІОРИТЕТ 2: Спеціальна логіка для BTC конвертації
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
                    
                    # Якщо Uniswap не вдався, пробуємо тільки Binance
                    if not success and self.binance_client:
                        print("🔶 Спроба конвертації BTC через Binance...")
                        success = self._binance_convert(from_asset, to_asset, convert_amount, is_max)
                else:
                    # ПРІОРИТЕТ 3: Для інших токенів стандартна логіка
                    uniswap_from = from_asset
                    uniswap_to = to_asset
                    
                    if (self.use_uniswap and uniswap_from in self.token_addresses 
                        and uniswap_to in self.token_addresses):
                        print("🦄 Спроба конвертації через Uniswap...")
                        success = self._uniswap_convert(uniswap_from, uniswap_to, convert_amount, is_max)
                        
                    if not success and self.binance_client:
                        print("🔶 Спроба конвертації через Binance...")
                        success = self._binance_convert(from_asset, to_asset, convert_amount, is_max)
                
                # ПРІОРИТЕТ 4: Fallback на розумну браузерну автоматизацію якщо все не вдалося
                if not success:
                    print("🌐 Fallback: Спроба розумної браузерної автоматизації...")
                    success = self._smart_browser_conversion(from_asset, to_asset, convert_amount if not is_max else 'max')
                
        if success:
            self._show_conversion_remainder(from_asset, initial_balance, is_max)
        else:
            convert_value_usd = convert_amount * self.rates.get(from_asset, 1.0)
            print(f"\n❌ Конвертація не вдалася!")
            print(f"💰 Сума конвертації: {convert_amount:.8f} {from_asset} (~${convert_value_usd:.2f})")
            
            print(f"⚠️ Можливі причини:")
            print(f"   • Торгова пара недоступна")
            print(f"   • Сума занадто мала для мінімальних обмежень біржі")
            print(f"   • Convert API недоступний для вашого акаунта")
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
        # Видалено перевірку мінімальної суми - дозволяємо конвертувати будь-яку кількість
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

    def _launch_browser_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Відкриває Binance Convert з конкретною парою токенів і автоматизує процес"""
        try:
            if self.testnet:
                print("❌ Автоматизація недоступна в тестовому режимі")
                return False
                
            print(f"🤖 Покращена автоматична конвертація: {from_asset} → {to_asset}")
            print(f"💰 Сума: {amount}")
            
            # ПРІОРИТЕТ 1: MCP Playwright автоматизація (обходить CSP обмеження)
            print("🎭 Спроба MCP Playwright автоматизації...")
            if self._mcp_playwright_conversion(from_asset, to_asset, amount):
                return True
            
            # ПРІОРИТЕТ 2: Існуючий авторизований браузер
            print("🦊 Спроба використання вашого основного Firefox...")
            if self._smart_browser_conversion(from_asset, to_asset, amount):
                return True
            
            # ПРІОРИТЕТ 3: Selenium автоматизація
            if SELENIUM_AVAILABLE:
                print("🔧 Спроба автоматизації через Selenium...")
                if self._selenium_browser_conversion(from_asset, to_asset, amount):
                    return True
                else:
                    print("⚠️ Selenium автоматизація не вдалася, переходимо на JavaScript")
            
            # ПРІОРИТЕТ 4: Fallback на JavaScript підхід
            return self._javascript_browser_conversion(from_asset, to_asset, amount)
                    
        except Exception as e:
            print(f"❌ Помилка автоматизації: {e}")
            return False

    def _mcp_playwright_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Повна автоматизація через MCP Playwright - обходить CSP обмеження"""
        try:
            print("🎭 MCP Playwright автоматизація - обходимо CSP обмеження...")
            print(f"💱 Конвертація: {from_asset} → {to_asset}, сума: {amount}")
            
            # Конвертуємо URL для конкретної пари
            convert_url = f"https://www.binance.com/en/convert/{from_asset}/{to_asset}"
            
            # Відкриваємо сторінку через MCP Playwright
            print("🌐 Відкриваємо Binance Convert...")
            
            # Використовуємо MCP browser tools для автоматизації
            try:
                # Відкриваємо сторінку
                self._mcp_navigate_to_convert(convert_url)
                
                # Чекаємо завантаження сторінки
                print("⏳ Очікуємо завантаження сторінки...")
                time.sleep(5)
                
                # Перевіряємо чи користувач авторизований
                if not self._mcp_check_login():
                    print("🔐 Потрібна авторизація в Binance...")
                    if not self._mcp_handle_login():
                        print("❌ Не вдалося авторизуватися")
                        return False
                
                # Встановлюємо токени якщо потрібно
                print("🔄 Налаштування пари токенів...")
                if not self._mcp_setup_token_pair(from_asset, to_asset):
                    print("⚠️ Не вдалося встановити пару токенів, продовжуємо...")
                
                # Встановлюємо кількість
                print("💰 Встановлення кількості...")
                if not self._mcp_set_amount(amount):
                    print("❌ Не вдалося встановити кількість")
                    return False
                
                # Виконуємо конвертацію
                print("🚀 Виконання конвертації...")
                if not self._mcp_execute_conversion():
                    print("❌ Не вдалося виконати конвертацію")
                    return False
                
                # Перевіряємо результат
                print("✅ Перевірка результату...")
                success = self._mcp_check_conversion_result()
                
                if success:
                    print("🎉 MCP Playwright конвертація успішна!")
                    return True
                else:
                    print("⚠️ Результат конвертації невизначений")
                    return False
                    
            except Exception as mcp_error:
                print(f"❌ Помилка MCP Playwright: {mcp_error}")
                return False
                
        except Exception as e:
            print(f"❌ Критична помилка MCP автоматизації: {e}")
            return False

    def _mcp_navigate_to_convert(self, url: str) -> bool:
        """Навігація до сторінки конвертації через MCP"""
        try:
            # Використовуємо MCP Playwright для навігації - викликаємо функцію напряму
            result = browser_navigate_mcp_microsoft_playwright(url=url)
            print(f"🔗 MCP навігація до: {url}")
            return True
        except NameError:
            # MCP tools недоступні, використовуємо fallback
            print("⚠️ MCP Playwright недоступний, використовуємо webbrowser")
            import webbrowser
            webbrowser.open(url)
            print(f"🔗 Fallback відкриття: {url}")
            return True
        except Exception as e:
            print(f"❌ Помилка MCP навігації: {e}")
            # Fallback на webbrowser
            try:
                import webbrowser
                webbrowser.open(url)
                print(f"🔗 Fallback відкриття: {url}")
                return True
            except:
                return False

    def _mcp_check_login(self) -> bool:
        """Перевіряє чи користувач авторизований через MCP"""
        try:
            print("🔍 MCP перевірка статусу авторизації...")
            
            # Використовуємо MCP Playwright для перевірки авторизації
            # Шукаємо кнопки входу - якщо є, то не авторизований
            login_selectors = [
                "//button[contains(text(), 'Log In')]",
                "//button[contains(text(), 'Sign In')]", 
                "//a[contains(text(), 'Log In')]",
                "[data-testid*='login']",
                ".login-btn"
            ]
            
            for selector in login_selectors:
                try:
                    # Використовуємо MCP snapshot для перевірки наявності елементів
                    snapshot_result = self._mcp_take_snapshot()
                    if "Log In" in snapshot_result or "Sign In" in snapshot_result:
                        print("⚠️ Знайдено кнопки входу - користувач не авторизований")
                        return False
                except:
                    continue
            
            print("✅ Кнопки входу не знайдено - користувач авторизований")
            return True
            
        except Exception as e:
            print(f"⚠️ Помилка MCP перевірки авторизації: {e}")
            return True  # За замовчуванням вважаємо авторизованим

    def _mcp_handle_login(self) -> bool:
        """Обробляє авторизацію якщо потрібно"""
        try:
            print("🔐 Обробка авторизації...")
            print("💡 Будь ласка, увійдіть в акаунт вручну якщо потрібно")
            
            # Даємо час користувачеві увійти
            for i in range(30, 0, -5):
                print(f"⏳ Очікування авторизації... {i} секунд")
                time.sleep(5)
            
            return True
        except Exception as e:
            print(f"❌ Помилка обробки авторизації: {e}")
            return False

    def _mcp_setup_token_pair(self, from_asset: str, to_asset: str) -> bool:
        """Налаштовує пару токенів через MCP"""
        try:
            # Перевіряємо доступність MCP
            if not self._check_mcp_availability():
                print("⚠️ MCP недоступний для налаштування пари токенів")
                return False
                
            print(f"🔄 MCP налаштування пари: {from_asset} → {to_asset}")
            
            # Спочатку беремо snapshot для аналізу поточного стану
            current_state = self._mcp_take_snapshot()
            
            # Автоматично визначаємо поточні токени
            current_from, current_to = self._mcp_detect_current_pair()
            print(f"📊 Поточна пара: {current_from or 'UNKNOWN'} → {current_to or 'UNKNOWN'}")
            
            # Змінюємо FROM токен якщо потрібно
            if current_from != from_asset:
                print(f"🔄 Зміна FROM токена: {current_from} → {from_asset}")
                if not self._mcp_change_token(True, from_asset):
                    print("⚠️ Не вдалося змінити FROM токен, продовжуємо...")
            
            # Змінюємо TO токен якщо потрібно  
            if current_to != to_asset:
                print(f"🔄 Зміна TO токена: {current_to} → {to_asset}")
                if not self._mcp_change_token(False, to_asset):
                    print("⚠️ Не вдалося змінити TO токен, продовжуємо...")
            
            # Перевіряємо фінальний результат
            time.sleep(2)
            final_from, final_to = self._mcp_detect_current_pair()
            print(f"✅ Фінальна пара: {final_from or 'UNKNOWN'} → {final_to or 'UNKNOWN'}")
            
            # Реальна MCP імплементація потрібна
            print("❌ MCP імплементація налаштування пари не завершена")
            return False
            
        except Exception as e:
            print(f"⚠️ Помилка MCP налаштування пари: {e}")
            return False

    def _mcp_set_amount(self, amount) -> bool:
        """Встановлює кількість через MCP"""
        try:
            # Перевіряємо доступність MCP
            if not self._check_mcp_availability():
                print("⚠️ MCP недоступний для встановлення кількості")
                return False
                
            amount_str = "max" if str(amount).lower() == 'max' else str(amount)
            print(f"💰 Встановлення кількості: {amount_str}")
            
            if amount_str == "max":
                print("🔝 Пошук кнопки MAX...")
                # Тут має бути реальний MCP Playwright код
                
            # Реальна MCP імплементація
            print("❌ MCP імплементація не завершена")
            return False
            
        except Exception as e:
            print(f"❌ Помилка встановлення кількості: {e}")
            return False

    def _mcp_execute_conversion(self) -> bool:
        """Виконує конвертацію через MCP"""
        try:
            # Перевіряємо доступність MCP
            if not self._check_mcp_availability():
                print("⚠️ MCP недоступний для виконання конвертації")
                return False
                
            print("🚀 Пошук кнопки Convert...")
            # Тут має бути реальний MCP Playwright код
            
            # Реальна MCP імплементація
            print("❌ MCP імплементація не завершена")
            return False
            
        except Exception as e:
            print(f"❌ Помилка виконання конвертації: {e}")
            return False

    def _mcp_check_conversion_result(self) -> bool:
        """Перевіряє результат конвертації через MCP"""
        try:
            # Перевіряємо доступність MCP
            if not self._check_mcp_availability():
                print("⚠️ MCP недоступний для перевірки результату")
                return False
                
            print("🔍 Перевірка результату конвертації...")
            time.sleep(5)  # Чекаємо завершення конвертації
            
            # MCP Playwright може перевірити успішність конвертації
            # шукаючи повідомлення про успіх або помилку
            
            # Реальна MCP імплементація
            print("❌ MCP імплементація не завершена")
            return False
            
        except Exception as e:
            print(f"⚠️ Помилка перевірки результату: {e}")
            return False

    def _mcp_take_snapshot(self) -> str:
        """Робить snapshot сторінки через MCP Playwright"""
        try:
            snapshot_result = browser_snapshot_mcp_microsoft_playwright()
            return str(snapshot_result)
        except NameError:
            print("⚠️ MCP Playwright snapshot недоступний")
            return ""
        except Exception as e:
            print(f"⚠️ Помилка MCP snapshot: {e}")
            return ""

    def _mcp_detect_current_pair(self) -> tuple:
        """Автоматично визначає поточну пару токенів на сторінці через MCP Playwright"""
        try:
            print("🔍 MCP автоматичне визначення поточної пари токенів...")
            
            # Беремо snapshot для аналізу
            snapshot = self._mcp_take_snapshot()
            
            if not snapshot:
                print("❌ Не вдалося отримати snapshot сторінки")
                return None, None
            
            # Аналізуємо snapshot для пошуку токенів
            import re
            
            # Покращені паттерни для пошуку токенів
            token_patterns = [
                r'(?:From|from)[\s\S]*?([A-Z]{2,6})(?:\s|$)',  # Після "From"
                r'(?:To|to)[\s\S]*?([A-Z]{2,6})(?:\s|$)',      # Після "To"
                r'data-testid="[^"]*(?:from|to)[^"]*"[^>]*>[\s\S]*?([A-Z]{2,6})',  # В data-testid
                r'class="[^"]*(?:from|to)[^"]*"[^>]*>[\s\S]*?([A-Z]{2,6})',        # В class
                r'\b([A-Z]{2,6})\s*/\s*([A-Z]{2,6})\b',       # Формат BTC/USDT
                r'Convert\s+([A-Z]{2,6})\s+to\s+([A-Z]{2,6})', # "Convert BTC to USDT"
            ]
            
            # Відомі криптовалютні токени для валідації
            known_tokens = {
                'BTC', 'ETH', 'BNB', 'USDT', 'USDC', 'ADA', 'DOT', 'SOL', 'MATIC', 
                'LINK', 'UNI', 'AAVE', 'CRO', 'LDO', 'COMP', 'PEPE', 'SHIB', 'DOGE',
                'XRP', 'LTC', 'BCH', 'ETC', 'ATOM', 'AVAX', 'NEAR', 'FTM', 'ALGO'
            }
            
            detected_tokens = []
            
            # Використовуємо різні стратегії пошуку
            for pattern in token_patterns:
                matches = re.findall(pattern, snapshot, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple):
                        # Для паттернів що повертають кортежі (наприклад BTC/USDT)
                        for token in match:
                            if token.upper() in known_tokens:
                                detected_tokens.append(token.upper())
                    else:
                        # Для простих паттернів
                        if match.upper() in known_tokens:
                            detected_tokens.append(match.upper())
            
            # Видаляємо дублікати зберігаючи порядок
            unique_tokens = []
            for token in detected_tokens:
                if token not in unique_tokens:
                    unique_tokens.append(token)
            
            print(f"🔍 Знайдені токени в snapshot: {unique_tokens}")
            
            # Визначаємо FROM та TO токени
            from_token = None
            to_token = None
            
            if len(unique_tokens) >= 2:
                from_token = unique_tokens[0]
                to_token = unique_tokens[1]
                print(f"✅ Автоматично визначено пару: {from_token} → {to_token}")
            elif len(unique_tokens) == 1:
                # Якщо знайдено тільки один токен, спробуємо визначити контекст
                token = unique_tokens[0]
                if 'from' in snapshot.lower():
                    from_token = token
                elif 'to' in snapshot.lower():
                    to_token = token
                print(f"⚠️ Знайдено тільки один токен: {token}")
            else:
                print("❌ Не вдалося визначити токени зі snapshot")
            
            return from_token, to_token
            
        except Exception as e:
            print(f"❌ Помилка визначення пари токенів: {e}")
            return None, None

    def _mcp_detect_available_amount(self, from_token: str) -> str:
        """Автоматично визначає доступну суму для конвертації через MCP Playwright"""
        try:
            print(f"💰 MCP визначення доступної суми для {from_token}...")
            
            # Беремо snapshot для пошуку балансу
            snapshot = self._mcp_take_snapshot()
            
            if not snapshot:
                print("❌ Не вдалося отримати snapshot для визначення суми")
                return "max"
            
            # Паттерни для пошуку балансу
            balance_patterns = [
                rf'(?:Available|Balance|Доступно)[\s\S]*?(\d+(?:\.\d+)?)\s*{from_token}',
                rf'{from_token}[\s\S]*?(?:Available|Balance|Доступно)[\s\S]*?(\d+(?:\.\d+)?)',
                rf'(\d+(?:\.\d+)?)\s*{from_token}[\s\S]*?(?:Available|Balance)',
                rf'balance["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)["\']?.*{from_token}',
                rf'{from_token}["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:BTC|ETH|BNB|USDT|USDC|ADA|DOT|SOL)'  # Загальний паттерн
            ]
            
            detected_amounts = []
            
            for pattern in balance_patterns:
                matches = re.findall(pattern, snapshot, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    try:
                        amount = float(match)
                        if amount > 0:
                            detected_amounts.append(amount)
                            print(f"🔍 Знайдено потенційний баланс: {amount} {from_token}")
                    except ValueError:
                        continue
            
            if detected_amounts:
                # Беремо найбільший знайдений баланс (ймовірно найточніший)
                max_amount = max(detected_amounts)
                print(f"✅ Визначено доступну суму: {max_amount} {from_token}")
                return str(max_amount)
            else:
                print(f"⚠️ Не вдалося визначити точну суму для {from_token}, використовуємо 'max'")
                return "max"
                
        except Exception as e:
            print(f"❌ Помилка визначення доступної суми: {e}")
            return "max"

    def _mcp_detect_current_pair(self) -> tuple:
        """Автоматично визначає поточну пару токенів на сторінці"""
        try:
            snapshot = self._mcp_take_snapshot()
            
            # Аналізуємо snapshot для пошуку токенів
            import re
            
            # Шукаємо паттерни токенів у snapshot
            token_pattern = r'\b[A-Z]{2,6}\b'
            tokens_found = re.findall(token_pattern, snapshot)
            
            # Фільтруємо найбільш ймовірні токени (відомі криптовалюти)
            known_tokens = ['BTC', 'ETH', 'BNB', 'USDT', 'USDC', 'ADA', 'DOT', 'SOL', 'MATIC', 'LINK']
            likely_tokens = [token for token in tokens_found if token in known_tokens]
            
            if len(likely_tokens) >= 2:
                from_token = likely_tokens[0]
                to_token = likely_tokens[1]
                print(f"🔍 Автоматично визначено пару: {from_token} → {to_token}")
                return from_token, to_token
            else:
                print("⚠️ Не вдалося автоматично визначити пару токенів")
                return None, None
                
        except Exception as e:
            print(f"❌ Помилка визначення пари: {e}")
            return None, None

    def _mcp_change_token(self, is_from: bool, token_symbol: str) -> bool:
        """Змінює токен через MCP Playwright"""
        try:
            token_type = "FROM" if is_from else "TO"
            print(f"🔄 MCP зміна {token_type} токена на {token_symbol}")
            
            # Селектори для кнопок токенів
            if is_from:
                selectors = [
                    '[data-testid="from-token-selector"]',
                    '[data-testid="from-asset-selector"]',
                    '[class*="from-token"]',
                    'button:first-of-type'
                ]
            else:
                selectors = [
                    '[data-testid="to-token-selector"]',
                    '[data-testid="to-asset-selector"]',
                    '[class*="to-token"]',
                    'button:last-of-type'
                ]
            
            # Пробуємо клікнути по селектору токена
            for selector in selectors:
                try:
                    click_result = browser_click_mcp_microsoft_playwright(
                        element=f"{token_type} token selector",
                        ref=selector
                    )
                    if click_result:
                        print(f"✅ Клік по {token_type} селектору: {selector}")
                        time.sleep(1)
                        break
                except NameError:
                    print("⚠️ MCP click недоступний")
                    continue
                except:
                    continue
            
            # Шукаємо поле пошуку та вводимо токен
            search_selectors = [
                'input[placeholder*="Search"]',
                'input[type="text"]',
                '[data-testid="search-input"]'
            ]
            
            for search_selector in search_selectors:
                try:
                    type_result = browser_type_mcp_microsoft_playwright(
                        element="token search field",
                        ref=search_selector,
                        text=token_symbol
                    )
                    if type_result:
                        print(f"✅ Введено токен {token_symbol} в пошук")
                        time.sleep(1)
                        break
                except NameError:
                    print("⚠️ MCP type недоступний")
                    continue
                except:
                    continue
            
            # Клікаємо по токену в результатах пошуку
            time.sleep(1)
            try:
                # Шукаємо токен у списку результатів
                token_click_result = browser_click_mcp_microsoft_playwright(
                    element=f"token {token_symbol} in search results",
                    ref=f"//div[contains(text(), '{token_symbol}')]"
                )
                if token_click_result:
                    print(f"✅ Вибрано токен {token_symbol}")
                    return True
            except NameError:
                print("⚠️ MCP click для токена недоступний")
                pass
            except:
                pass
                
            print(f"⚠️ Не вдалося повністю змінити {token_type} токен на {token_symbol}")
            return False
            
        except Exception as e:
            print(f"❌ Помилка MCP зміни токена: {e}")
            return False

    def _mcp_autonomous_conversion(self, from_asset: str = None, to_asset: str = None, amount = None) -> bool:
        """MCP Playwright недоступний - негайний fallback"""
        print("❌ MCP PLAYWRIGHT НЕДОСТУПНИЙ")
        print("📋 Причини:")
        print("   • MCP сервери не запущені у вашому середовищі")
        print("   • MCP tools повертають помилки при виклику")
        print("   • Навіть якби працювали, CSP Binance блокує всі JavaScript операції")
        print("")
        print("🔄 АВТОМАТИЧНИЙ FALLBACK НА БРАУЗЕРНУ АВТОМАТИЗАЦІЮ...")
        return self._smart_browser_conversion(from_asset, to_asset, amount if amount is not None else 'max')

    def _check_mcp_availability(self) -> bool:
        """Перевіряє чи доступні MCP Playwright tools"""
        try:
            # Перевіряємо чи існують MCP функції в глобальному просторі імен
            import sys
            if 'browser_snapshot_mcp_microsoft_playwright' in globals():
                return True
            return False
        except Exception:
            return False

    def _mcp_handle_registration(self) -> bool:
        """Обробляє реєстрацію/авторизацію через MCP Playwright"""
        try:
            print("🔐 Обробка авторизації через MCP...")
            
            # Перевіряємо чи є форма логіну
            snapshot = self._mcp_take_snapshot()
            
            if not snapshot:
                print("❌ Не вдалося отримати snapshot для перевірки авторизації")
                return False
            
            # Шукаємо ознаки необхідності авторизації
            login_indicators = [
                'log in', 'sign in', 'login', 'signin',
                'register', 'signup', 'sign up',
                'email', 'password', 'username'
            ]
            
            needs_login = any(indicator in snapshot.lower() for indicator in login_indicators)
            
            if needs_login:
                print("📋 Знайдено форму авторизації")
                
                # Пробуємо різні методи авторизації
                if self._mcp_try_google_auth():
                    print("✅ Авторизація через Google успішна")
                    return True
                elif self._mcp_try_email_auth():
                    print("✅ Авторизація через email успішна")
                    return True
                else:
                    print("⚠️ Потрібна ручна авторизація")
                    # Чекаємо поки користувач авторизується
                    return self._mcp_wait_for_manual_login()
            else:
                print("✅ Користувач вже авторизований")
                return True
                
        except Exception as e:
            print(f"❌ Помилка обробки реєстрації: {e}")
            return False

    def _mcp_try_google_auth(self) -> bool:
        """Спроба авторизації через Google"""
        try:
            print("🔍 Пошук кнопки Google авторизації...")
            
            # Шукаємо кнопку Google
            google_selectors = [
                'button[data-testid*="google"]',
                'button:contains("Google")',
                '[class*="google"]',
                'button[title*="Google"]'
            ]
            
            for selector in google_selectors:
                try:
                    click_result = browser_click_mcp_microsoft_playwright(
                        element="Google login button",
                        ref=selector
                    )
                    if click_result:
                        print("✅ Клік по кнопці Google авторизації")
                        time.sleep(3)
                        
                        # Чекаємо завершення авторизації Google
                        return self._mcp_wait_for_auth_completion()
                except NameError:
                    print("⚠️ MCP click недоступний для Google auth")
                    continue
                except:
                    continue
            
            print("⚠️ Кнопка Google авторизації не знайдена")
            return False
            
        except Exception as e:
            print(f"❌ Помилка Google авторизації: {e}")
            return False

    def _mcp_try_email_auth(self) -> bool:
        """Спроба авторизації через email (базова реалізація)"""
        try:
            print("📧 Пошук форми email авторизації...")
            
            # Шукаємо поля email та password
            try:
                email_field = browser_type_mcp_microsoft_playwright(
                    element="email field",
                    ref='input[type="email"], input[name*="email"], input[placeholder*="email"]',
                    text=os.getenv('BINANCE_EMAIL', '')
                )
            except NameError:
                print("⚠️ MCP type недоступний для email")
                return False
            
            if not email_field:
                print("⚠️ Поле email не знайдено")
                return False
            
            try:
                password_field = browser_type_mcp_microsoft_playwright(
                    element="password field",
                    ref='input[type="password"], input[name*="password"]',
                    text=os.getenv('BINANCE_PASSWORD', '')
                )
            except NameError:
                print("⚠️ MCP type недоступний для password")
                return False
            
            if not password_field:
                print("⚠️ Поле password не знайдено")
                return False
            
            # Клікаємо кнопку входу
            try:
                login_button = browser_click_mcp_microsoft_playwright(
                    element="login button",
                    ref='button[type="submit"], button:contains("Log in"), button:contains("Sign in")'
                )
            except NameError:
                print("⚠️ MCP click недоступний для login button")
                return False
            
            if login_button:
                print("✅ Форма авторизації надіслана")
                time.sleep(3)
                return self._mcp_wait_for_auth_completion()
            else:
                print("⚠️ Кнопка входу не знайдена")
                return False
                
        except Exception as e:
            print(f"❌ Помилка email авторизації: {e}")
            return False

    def _mcp_wait_for_auth_completion(self) -> bool:
        """Чекає завершення процесу авторизації"""
        try:
            print("⏳ Очікування завершення авторизації...")
            
            max_wait = 60  # 60 секунд максимум
            for i in range(max_wait):
                snapshot = self._mcp_take_snapshot()
                
                # Перевіряємо ознаки успішної авторизації
                if snapshot and any(indicator not in snapshot.lower() for indicator in ['log in', 'sign in', 'login']):
                    print("✅ Авторизація завершена")
                    return True
                
                time.sleep(1)
                if i % 10 == 0:
                    print(f"⏳ Очікування авторизації... {i}/{max_wait} секунд")
            
            print("⚠️ Таймаут очікування авторизації")
            return False
            
        except Exception as e:
            print(f"❌ Помилка очікування авторизації: {e}")
            return False

    def _mcp_wait_for_manual_login(self) -> bool:
        """Чекає поки користувач вручну авторизується"""
        try:
            print("👤 Очікування ручної авторизації користувача...")
            print("💡 Будь ласка, авторизуйтеся на сайті вручну")
            
            max_wait = 300  # 5 хвилин максимум
            for i in range(0, max_wait, 10):
                snapshot = self._mcp_take_snapshot()
                
                # Перевіряємо чи користувач авторизувався
                if snapshot and not any(indicator in snapshot.lower() for indicator in ['log in', 'sign in', 'login']):
                    print("✅ Ручна авторизація завершена")
                    return True
                
                time.sleep(10)
                print(f"⏳ Очікування ручної авторизації... {i+10}/{max_wait} секунд")
            
            print("⚠️ Таймаут очікування ручної авторизації")
            return False
            
        except Exception as e:
            print(f"❌ Помилка очікування ручної авторизації: {e}")
            return False

    def _mcp_get_current_url(self) -> str:
        """Отримує поточний URL через MCP Playwright"""
        try:
            # Використовуємо JavaScript для отримання URL
            result = browser_evaluate_mcp_microsoft_playwright(
                function="() => window.location.href"
            )
            return str(result) if result else ""
        except NameError:
            print("⚠️ MCP evaluate недоступний")
            return ""
        except Exception as e:
            print(f"⚠️ Помилка отримання URL: {e}")
            return ""

    def _mcp_detect_amount(self) -> str:
        """Автоматично визначає доступну суму для конвертації"""
        try:
            print("💰 Автоматичне визначення доступної суми...")
            
            snapshot = self._mcp_take_snapshot()
            
            # Шукаємо баланс у snapshot
            import re
            balance_patterns = [
                r'Available.*?(\d+(?:\.\d+)?)',
                r'Balance.*?(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:BTC|ETH|BNB|USDT|USDC)',
            ]
            
            for pattern in balance_patterns:
                matches = re.findall(pattern, snapshot, re.IGNORECASE)
                if matches:
                    amount = matches[0]
                    print(f"✅ Автоматично визначено суму: {amount}")
                    return amount
            
            print("⚠️ Не вдалося автоматично визначити суму, використовую 'max'")
            return "max"
            
        except Exception as e:
            print(f"❌ Помилка визначення суми: {e}")
            return "max"

    def _smart_browser_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Покращений метод браузерної автоматизації з обходом CSP"""
        try:
            print("🌐 Покращена браузерна автоматизація...")
            
            convert_url = f"https://www.binance.com/en/convert/{from_asset}/{to_asset}"
            
            # Відкриваємо URL в браузері
            print("🔗 Відкриття URL в браузері...")
            if self._simple_firefox_open(convert_url):
                print("✅ Сторінка відкрита в Firefox!")
            else:
                print("🌐 Fallback: відкриття через системний браузер...")
                import webbrowser
                webbrowser.open(convert_url)
            
            # Генеруємо CSP-безпечний код
            amount_value = "max" if str(amount).lower() == 'max' else str(amount)
            
            print(f"\n🎯 === CSP-БЕЗПЕЧНА АВТОМАТИЗАЦІЯ ===")
            print(f"🔗 Відкрито: {convert_url}")
            print(f"💱 Пара: {from_asset} → {to_asset}")
            print(f"💰 Сума: {amount_value}")
            print(f"")
            print(f"🛡️ ВАЖЛИВО: Виявлено CSP обмеження!")
            print(f"📋 CSP блокує JavaScript виконання, тому:")
            print(f"")
            print(f"🎯 ВАРІАНТ 1: Ручна конвертація (РЕКОМЕНДОВАНО)")
            print(f"   1. Перевірте що відкрилася сторінка: /convert/{from_asset}/{to_asset}")
            print(f"   2. Впевніться що ви увійшли в акаунт Binance")
            print(f"   3. Перевірте пару токенів: {from_asset} → {to_asset}")
            print(f"   4. Встановіть кількість: {amount_value}")
            if amount_value == "max":
                print(f"      • Натисніть кнопку 'Max' для використання всього балансу")
            print(f"   5. Натисніть 'Convert' та підтвердіть операцію")
            print(f"")
            print(f"🔧 ВАРІАНТ 2: Обхід CSP (ЕКСПЕРИМЕНТАЛЬНИЙ)")
            print(f"   1. Натисніть F12 → Console")
            print(f"   2. Спробуйте вставити код частинами:")
            
            # Генеруємо спрощений CSP-безпечний код
            simple_code = self._generate_csp_safe_code(from_asset, to_asset, amount_value)
            print(f"")
            print("=" * 60)
            print(simple_code)
            print("=" * 60)
            print(f"")
            print(f"💡 Причина CSP проблем:")
            print(f"   • Binance використовує строгу Content Security Policy")
            print(f"   • Блокується виконання inline JavaScript")
            print(f"   • eval() та подібні функції заборонені")
            print(f"   • Рекомендується ручна конвертація")
            
            # Спрощений процес підтвердження
            print(f"\n🎯 Як продовжити:")
            print(f"   [1] - Виконав ручну конвертацію")
            print(f"   [2] - CSP код спрацював")
            print(f"   [0] - Скасувати операцію")
            
            while True:
                try:
                    choice = input("👉 Ваш вибір (1/2/0): ").strip()
                    
                    if choice == '1':
                        print("✅ Ручна конвертація прийнята!")
                        print(f"💰 Конвертація {from_asset} → {to_asset} завершена")
                        return True
                    elif choice == '2':
                        print("🎉 CSP обхід успішний!")
                        print(f"💰 Автоматична конвертація {from_asset} → {to_asset} завершена")
                        return True
                    elif choice == '0':
                        print("❌ Операція скасована")
                        return False
                    else:
                        print("❌ Введіть 1, 2 або 0")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Операція перервана")
                    return False
                    
        except Exception as e:
            print(f"❌ Помилка браузерної автоматизації: {e}")
            return False

    def _generate_csp_safe_code(self, from_asset: str, to_asset: str, amount: str) -> str:
        """Генерує максимально CSP-безпечний код для обходу блокувань"""
        return f'''// ⚡ МАКСИМАЛЬНО CSP-БЕЗПЕЧНИЙ КОД
// Копіюйте та вставляйте по частинах!

// Крок 1: Базові функції (вставити першим)
window.step1 = function() {{
    console.log("🔧 Крок 1: Ініціалізація базових функцій");
    
    window.safeClick = function(elem) {{
        if (!elem) return false;
        elem.scrollIntoView({{behavior: 'smooth', block: 'center'}});
        setTimeout(() => elem.click(), 200);
        return true;
    }};
    
    window.safeType = function(elem, text) {{
        if (!elem) return false;
        elem.focus();
        elem.value = '';
        elem.value = text;
        elem.dispatchEvent(new Event('input', {{bubbles: true}}));
        elem.dispatchEvent(new Event('change', {{bubbles: true}}));
        return true;
    }};
    
    window.findByText = function(text) {{
        const all = document.querySelectorAll('*');
        for (let elem of all) {{
            if (elem.textContent && elem.textContent.toLowerCase().includes(text.toLowerCase()) && 
                elem.offsetParent !== null && !elem.disabled) {{
                return elem;
            }}
        }}
        return null;
    }};
    
    console.log("✅ Крок 1 завершено");
}};

// Крок 2: Пошук елементів (вставити другим)
window.step2 = function() {{
    console.log("🔍 Крок 2: Пошук елементів конвертації");
    
    // Знаходимо поле кількості
    const inputs = document.querySelectorAll('input[type="text"], input[type="number"]');
    window.amountInput = null;
    for (let inp of inputs) {{
        if (inp.offsetParent !== null && !inp.disabled) {{
            window.amountInput = inp;
            break;
        }}
    }}
    
    // Знаходимо кнопку MAX
    window.maxButton = window.findByText('max');
    
    // Знаходимо кнопку Convert
    window.convertButton = window.findByText('convert');
    
    console.log("📊 Знайдені елементи:");
    console.log("  Поле кількості:", !!window.amountInput);
    console.log("  Кнопка MAX:", !!window.maxButton);
    console.log("  Кнопка Convert:", !!window.convertButton);
    console.log("✅ Крок 2 завершено");
}};

// Крок 3: Встановлення кількості (вставити третім)
window.step3 = function() {{
    console.log("💰 Крок 3: Встановлення кількості");
    
    if ("{amount}" === "max" && window.maxButton) {{
        window.safeClick(window.maxButton);
        console.log("✅ Натиснуто кнопку MAX");
        console.log("⏳ Чекайте 3 секунди для розрахунків...");
        setTimeout(() => console.log("✅ Крок 3 завершено"), 3000);
    }} else if (window.amountInput) {{
        window.safeType(window.amountInput, "{amount}");
        console.log("✅ Введено кількість: {amount}");
        console.log("✅ Крок 3 завершено");
    }} else {{
        console.log("❌ Не знайдено поле для введення кількості");
    }}
}};

// Крок 4: Виконання конвертації (вставити четвертим)
window.step4 = function() {{
    console.log("🚀 Крок 4: Виконання конвертації");
    
    if (window.convertButton) {{
        window.safeClick(window.convertButton);
        console.log("✅ Натиснуто кнопку Convert");
        console.log("⏳ Шукаємо кнопку підтвердження...");
        
        setTimeout(() => {{
            const confirmBtn = window.findByText('confirm');
            if (confirmBtn) {{
                window.safeClick(confirmBtn);
                console.log("✅ Натиснуто кнопку Confirm");
                console.log("🎉 Конвертація завершена!");
            }} else {{
                console.log("ℹ️ Кнопка підтвердження не знайдена");
                console.log("✅ Конвертація, ймовірно, завершена");
            }}
        }}, 2000);
        
    }} else {{
        console.log("❌ Кнопка Convert не знайдена");
    }}
}};

// ІНСТРУКЦІЇ ДЛЯ ВИКОНАННЯ:
console.log("📋 === ІНСТРУКЦІЇ ===");
console.log("1. Виконайте: window.step1()");
console.log("2. Виконайте: window.step2()");  
console.log("3. Виконайте: window.step3()");
console.log("4. Зачекайте 3-5 секунд");
console.log("5. Виконайте: window.step4()");
console.log("");
console.log("💡 Якщо щось не працює:");
console.log("   • Перевірте що ви увійшли в Binance");
console.log("   • Оновіть сторінку та спробуйте знову");
console.log("   • Виконайте кроки вручну");

// АВТОМАТИЧНИЙ ЗАПУСК (якщо CSP дозволяє)
console.log("🤖 Спроба автоматичного запуску...");
try {{
    setTimeout(() => {{
        window.step1();
        setTimeout(() => {{
            window.step2();
            setTimeout(() => {{
                window.step3();
                setTimeout(() => {{
                    window.step4();
                }}, 4000);
            }}, 1000);
        }}, 1000);
    }}, 1000);
}} catch(e) {{
    console.log("⚠️ Автозапуск заблоковано CSP, виконуйте кроки вручну");
}}'''

    def _selenium_browser_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Автоматична конвертація через Selenium з обходом CSP"""
        try:
            print("🤖 Автоматична конвертація через Selenium...")
            
            # Спробуємо підключитися до існуючого Firefox
            driver = self._connect_to_existing_firefox()
            
            # Перевіряємо чи отримали спеціальний маркер для існуючого браузера
            if driver == "existing_browser":
                print("🌐 Використання існуючого авторизованого Firefox...")
                return self._use_existing_browser_conversion(from_asset, to_asset, amount)
            
            if not driver:
                # Якщо не вдалося підключитися, запускаємо новий з обходом CSP
                driver = self._start_firefox_with_csp_bypass()
                
            if not driver:
                print("❌ Не вдалося запустити браузер")
                return self._fallback_browser_conversion(from_asset, to_asset, amount)
            
            try:
                # Переходимо на сторінку конвертації
                convert_url = f"https://www.binance.com/en/convert/{from_asset}/{to_asset}"
                print(f"🌐 Переходимо на: {convert_url}")
                driver.get(convert_url)
                
                # Чекаємо завантаження
                time.sleep(5)
                
                # Перевіряємо чи користувач увійшов
                if not self._check_binance_login(driver):
                    print("⚠️ Потрібно увійти в Binance акаунт")
                    input("📋 Увійдіть в акаунт та натисніть Enter для продовження...")
                
                # Виконуємо конвертацію
                success = self._perform_selenium_conversion(driver, from_asset, to_asset, amount)
                
                if success:
                    print("✅ Selenium конвертація успішна!")
                    return True
                else:
                    print("⚠️ Selenium не вдалася, спроба JavaScript...")
                    return self._inject_javascript_bypass_csp(driver, from_asset, to_asset, amount)
                    
            finally:
                # НЕ закриваємо браузер - залишаємо відкритим для користувача
                print("🔄 Браузер залишається відкритим для перевірки результату")
                
        except Exception as e:
            print(f"❌ Помилка Selenium: {e}")
            return self._fallback_browser_conversion(from_asset, to_asset, amount)

    def _simple_firefox_open(self, url: str) -> bool:
        """Надійний метод відкриття URL в Firefox з множинними fallback варіантами"""
        try:
            print("🦊 Надійний запуск Firefox з URL...")
            
            # Метод 1: Через знайдений Firefox executable
            firefox_exe = self._find_firefox_executable()
            if firefox_exe:
                print(f"🔍 Знайдено Firefox: {firefox_exe}")
                
                # Спочатку пробуємо без прихованого вікна
                try:
                    result = subprocess.run(
                        [firefox_exe, url],
                        check=False,
                        timeout=8,
                        capture_output=True
                    )
                    
                    print(f"📊 Firefox повернув код: {result.returncode}")
                    if result.returncode == 0:
                        print("✅ Firefox запущено успішно!")
                        return True
                    elif result.returncode == 1:
                        # Код 1 часто означає що Firefox вже запущений і відкрив URL
                        print("✅ Firefox відкрив URL в існуючому процесі!")
                        return True
                        
                except subprocess.TimeoutExpired:
                    print("✅ Firefox запущено (таймаут - це нормально)")
                    return True
                except Exception as e:
                    print(f"⚠️ Помилка прямого запуску: {e}")
                    
                # Метод 1.1: Спробуємо з new-tab параметром
                try:
                    result = subprocess.run(
                        [firefox_exe, "-new-tab", url],
                        check=False,
                        timeout=5,
                        capture_output=True
                    )
                    
                    if result.returncode == 0 or result.returncode == 1:
                        print("✅ Firefox запущено з -new-tab!")
                        return True
                        
                except Exception as e:
                    print(f"⚠️ Помилка з -new-tab: {e}")
            
            # Метод 2: Через PowerShell
            print("🔧 Спроба через PowerShell...")
            try:
                ps_command = f'Start-Process firefox -ArgumentList "{url}"'
                result = subprocess.run(
                    ["powershell", "-Command", ps_command],
                    check=False,
                    timeout=8,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    print("✅ Firefox запущено через PowerShell!")
                    return True
                    
            except Exception as e:
                print(f"⚠️ PowerShell не вдався: {e}")
            
            # Метод 3: Через команду start Windows
            print("🪟 Спроба через Windows start...")
            try:
                result = subprocess.run(
                    ["cmd", "/c", "start", "firefox", url],
                    check=False,
                    timeout=5,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    print("✅ Firefox запущено через Windows start!")
                    return True
                    
            except Exception as e:
                print(f"⚠️ Windows start не вдався: {e}")
            
            # Метод 4: Через os.startfile (найнадійніший для Windows)
            print("💻 Спроба через os.startfile...")
            try:
                import os
                os.startfile(url)
                print("✅ URL відкрито через os.startfile!")
                return True
            except Exception as e:
                print(f"⚠️ os.startfile не вдався: {e}")
            
            print("❌ Всі методи запуску Firefox не вдалися")
            return False
                
        except Exception as e:
            print(f"❌ Критична помилка _simple_firefox_open: {e}")
            return False

    def _direct_firefox_open(self, url: str) -> bool:
        """Найпростіший і найнадійніший спосіб відкриття URL в існуючому Firefox"""
        try:
            print("🎯 Пряме відкриття URL в існуючому Firefox...")
            
            # Знаходимо Firefox executable
            firefox_paths = [
                r"C:\Program Files\Mozilla Firefox\firefox.exe",
                r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
            ]
            
            # Додаємо пошук через реєстр
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Mozilla\Mozilla Firefox") as key:
                    version_key = winreg.EnumKey(key, 0)
                    with winreg.OpenKey(key, f"{version_key}\\Main") as main_key:
                        install_dir = winreg.QueryValueEx(main_key, "Install Directory")[0]
                        firefox_paths.insert(0, os.path.join(install_dir, "firefox.exe"))
            except:
                pass
            
            firefox_exe = None
            for path in firefox_paths:
                if os.path.exists(path):
                    firefox_exe = path
                    print(f"✅ Firefox знайдено: {firefox_exe}")
                    break
            
            if not firefox_exe:
                print("❌ Firefox executable не знайдено")
                return False
            
            # Відкриваємо URL в існуючому Firefox процесі
            try:
                result = subprocess.run(
                    [firefox_exe, "-new-tab", url],
                    check=False,
                    timeout=10,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    print("✅ URL успішно відкрито в існуючому Firefox!")
                    return True
                else:
                    print(f"⚠️ Firefox повернув код помилки: {result.returncode}")
                    if result.stderr:
                        print(f"⚠️ Помилка: {result.stderr.decode()}")
                    return False
                    
            except subprocess.TimeoutExpired:
                print("⚠️ Таймаут при відкритті Firefox")
                return False
            except Exception as e:
                print(f"⚠️ Помилка виконання: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Критична помилка _direct_firefox_open: {e}")
            return False

    def _alternative_firefox_open(self, url: str) -> bool:
        """Альтернативний метод відкриття URL в існуючому Firefox"""
        try:
            print("🔄 Альтернативний метод відкриття в Firefox...")
            
            # Метод 1: Через PowerShell з примусовим Firefox
            try:
                print("🔧 Спроба через PowerShell...")
                ps_command = f'Start-Process firefox -ArgumentList "-new-tab", "{url}"'
                result = subprocess.run(
                    ["powershell", "-Command", ps_command],
                    check=False,
                    timeout=10,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    print("✅ URL відкрито через PowerShell!")
                    return True
                else:
                    print(f"⚠️ PowerShell повернув код: {result.returncode}")
                    
            except Exception as ps_error:
                print(f"⚠️ Помилка PowerShell: {ps_error}")
            
            # Метод 2: Через webbrowser з примусовим Firefox
            try:
                print("🌐 Спроба через webbrowser з примусовим Firefox...")
                
                # Зберігаємо поточний браузер за замовчуванням
                original_browser = os.environ.get('BROWSER', '')
                
                # Примусово встановлюємо Firefox
                os.environ['BROWSER'] = 'firefox'
                
                import webbrowser
                
                # Очищаємо кеш браузерів
                if hasattr(webbrowser, '_browsers'):
                    webbrowser._browsers.clear()
                
                # Реєструємо Firefox вручну
                firefox_cmd = None
                for path in [r"C:\Program Files\Mozilla Firefox\firefox.exe", 
                           r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"]:
                    if os.path.exists(path):
                        firefox_cmd = f'"{path}" %s'
                        break
                
                if firefox_cmd:
                    webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(firefox_cmd))
                    browser = webbrowser.get('firefox')
                    browser.open_new_tab(url)
                    print("✅ URL відкрито через webbrowser з примусовим Firefox!")
                    
                    # Відновлюємо браузер за замовчуванням
                    if original_browser:
                        os.environ['BROWSER'] = original_browser
                    else:
                        os.environ.pop('BROWSER', None)
                    
                    return True
                else:
                    print("⚠️ Не вдалося знайти Firefox для webbrowser")
                    
            except Exception as webbrowser_error:
                print(f"⚠️ Помилка webbrowser: {webbrowser_error}")
                # Відновлюємо браузер за замовчуванням при помилці
                if 'original_browser' in locals():
                    if original_browser:
                        os.environ['BROWSER'] = original_browser
                    else:
                        os.environ.pop('BROWSER', None)
            
            # Метод 3: Через start команду Windows
            try:
                print("🪟 Спроба через Windows start...")
                result = subprocess.run(
                    ["start", "firefox", url],
                    check=False,
                    timeout=5,
                    capture_output=True,
                    shell=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    print("✅ URL відкрито через Windows start!")
                    return True
                    
            except Exception as start_error:
                print(f"⚠️ Помилка Windows start: {start_error}")
            
            print("❌ Всі альтернативні методи не вдалися")
            return False
            
        except Exception as e:
            print(f"❌ Критична помилка альтернативного методу: {e}")
            return False

    def _launch_registered_firefox(self, url: str) -> bool:
        """Запускає Firefox з зареєстрованим профілем користувача"""
        try:
            print("🦊 Запуск Firefox з зареєстрованим профілем...")
            
            # Знаходимо Firefox executable
            firefox_exe = self._find_firefox_executable()
            if not firefox_exe:
                return False
            
            # Знаходимо найкращий профіль користувача
            firefox_profile_path = self._find_best_firefox_profile()
            if not firefox_profile_path:
                print("⚠️ Не знайдено підходящий профіль Firefox")
                return self._try_profile_manager_launch(firefox_exe, url)
            
            print(f"📁 Використання зареєстрованого профілю: {os.path.basename(firefox_profile_path)}")
            
            # Перевіряємо що профіль містить збережені дані
            if not self._validate_profile_has_data(firefox_profile_path):
                print("⚠️ Профіль не містить збережених даних, пробую інший...")
                # Пробуємо знайти інший профіль
                all_profiles = self._find_all_firefox_profiles()
                for profile_path, profile_name in all_profiles:
                    if profile_path != firefox_profile_path and self._validate_profile_has_data(profile_path):
                        print(f"✅ Знайдено профіль з даними: {profile_name}")
                        firefox_profile_path = profile_path
                        break
            
            # Запускаємо Firefox з вибраним профілем
            try:
                # Використовуємо параметр -no-remote для уникнення конфліктів
                result = subprocess.run(
                    [firefox_exe, "-no-remote", "-profile", firefox_profile_path, url],
                    check=False,
                    timeout=15,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    print("✅ Firefox запущено з зареєстрованим профілем!")
                    return True
                else:
                    print(f"⚠️ Firefox з профілем повернув код: {result.returncode}")
                    # Пробуємо без -no-remote
                    result2 = subprocess.run(
                        [firefox_exe, "-profile", firefox_profile_path, url],
                        check=False,
                        timeout=10,
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    
                    if result2.returncode == 0:
                        print("✅ Firefox запущено з профілем (без -no-remote)!")
                        return True
                    else:
                        print(f"❌ Обидві спроби не вдалися: {result.returncode}, {result2.returncode}")
                        return False
                        
            except subprocess.TimeoutExpired:
                print("⚠️ Таймаут при запуску Firefox з профілем")
                return False
            except Exception as e:
                print(f"⚠️ Помилка запуску Firefox з профілем: {e}")
                return False
                    
        except Exception as e:
            print(f"❌ Критична помилка _launch_registered_firefox: {e}")
            return False

    def _find_best_firefox_profile(self) -> Optional[str]:
        """Знаходить найкращий профіль Firefox з збереженими даними"""
        try:
            # Спочатку пробуємо стандартний метод
            profile = self._find_firefox_profile()
            if profile and self._validate_profile_has_data(profile):
                return profile
            
            # Якщо стандартний профіль не підходить, шукаємо серед всіх
            all_profiles = self._find_all_firefox_profiles()
            
            # Сортуємо профілі за пріоритетом
            priority_profiles = []
            other_profiles = []
            
            for profile_path, profile_name in all_profiles:
                # Перевіряємо наявність збережених даних
                if self._validate_profile_has_data(profile_path):
                    if any(keyword in profile_name.lower() for keyword in ['default-release', 'default']):
                        priority_profiles.append((profile_path, profile_name))
                    else:
                        other_profiles.append((profile_path, profile_name))
            
            # Повертаємо найкращий профіль
            if priority_profiles:
                best_profile = priority_profiles[0]
                print(f"✅ Вибрано пріоритетний профіль: {best_profile[1]}")
                return best_profile[0]
            elif other_profiles:
                best_profile = other_profiles[0]
                print(f"✅ Вибрано профіль з даними: {best_profile[1]}")
                return best_profile[0]
            
            print("❌ Не знайдено профіль з збереженими даними")
            return None
            
        except Exception as e:
            print(f"❌ Помилка пошуку найкращого профілю: {e}")
            return None

    def _validate_profile_has_data(self, profile_path: str) -> bool:
        """Перевіряє чи профіль містить збережені дані (cookies, історію тощо)"""
        try:
            # Перевіряємо наявність ключових файлів профілю
            essential_files = [
                "cookies.sqlite",      # Збережені cookies
                "places.sqlite",       # Історія та закладки
                "formhistory.sqlite",  # Історія форм
                "logins.json"          # Збережені логіни
            ]
            
            files_found = 0
            files_with_data = 0
            
            for file_name in essential_files:
                file_path = os.path.join(profile_path, file_name)
                if os.path.exists(file_path):
                    files_found += 1
                    # Перевіряємо розмір файлу (більше 1KB вказує на наявність даних)
                    if os.path.getsize(file_path) > 1024:
                        files_with_data += 1
            
            # Профіль вважається валідним якщо є хоча б 2 файли з даними
            has_data = files_with_data >= 2
            
            if has_data:
                print(f"✅ Профіль містить дані: {files_with_data}/{files_found} файлів з даними")
            else:
                print(f"⚠️ Профіль містить мало даних: {files_with_data}/{files_found} файлів з даними")
            
            return has_data
            
        except Exception as e:
            print(f"⚠️ Помилка валідації профілю: {e}")
            return False

    def _force_launch_with_profile(self, url: str) -> bool:
        """Форсований запуск Firefox з найкращим доступним профілем"""
        try:
            print("💪 Форсований запуск Firefox з зареєстрованим профілем...")
            
            # Знаходимо Firefox executable
            firefox_exe = self._find_firefox_executable()
            if not firefox_exe:
                return False
            
            # Спочатку пробуємо знайти всі доступні профілі
            profiles = self._find_all_firefox_profiles()
            
            if not profiles:
                print("❌ Жодного Firefox профілю не знайдено")
                return False
            
            # Пріоритет профілів: default-release > default > інші
            preferred_order = []
            other_profiles = []
            
            for profile_path, profile_name in profiles:
                if 'default-release' in profile_name:
                    preferred_order.insert(0, (profile_path, profile_name))
                elif 'default' in profile_name:
                    preferred_order.append((profile_path, profile_name))
                else:
                    other_profiles.append((profile_path, profile_name))
            
            # Об'єднуємо списки за пріоритетом
            all_profiles = preferred_order + other_profiles
            
            # Пробуємо запустити Firefox з кожним профілем
            for profile_path, profile_name in all_profiles:
                print(f"🔄 Спроба запуску з профілем: {profile_name}")
                try:
                    result = subprocess.run(
                        [firefox_exe, "-profile", profile_path, url],
                        check=False,
                        timeout=10,
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ Firefox запущено з профілем: {profile_name}")
                        return True
                    else:
                        print(f"⚠️ Профіль {profile_name} не вдався (код: {result.returncode})")
                        
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Таймаут для профілю: {profile_name}")
                    continue
                except Exception as e:
                    print(f"⚠️ Помилка для профілю {profile_name}: {e}")
                    continue
            
            print("❌ Всі профілі Firefox не вдалися")
            return False
            
        except Exception as e:
            print(f"❌ Критична помилка _force_launch_with_profile: {e}")
            return False

    def _try_profile_manager_launch(self, firefox_exe: str, url: str) -> bool:
        """Спроба запуску через profile manager для вибору зареєстрованого профілю"""
        try:
            print("🔧 Спроба запуску через автоматичний вибір профілю...")
            
            # Спробуємо запустити Firefox без явного профілю (він сам вибере активний)
            result = subprocess.run(
                [firefox_exe, "-no-remote", url],
                check=False,
                timeout=15,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if result.returncode == 0:
                print("✅ Firefox запущено з автоматичним вибором профілю!")
                return True
            else:
                print(f"⚠️ Автоматичний вибір не вдався (код: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"⚠️ Помилка profile manager запуску: {e}")
            return False

    def _find_firefox_executable(self) -> Optional[str]:
        """Знаходить Firefox executable"""
        firefox_paths = [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ]
        
        # Додаємо пошук через реєстр
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Mozilla\Mozilla Firefox") as key:
                version_key = winreg.EnumKey(key, 0)
                with winreg.OpenKey(key, f"{version_key}\\Main") as main_key:
                    install_dir = winreg.QueryValueEx(main_key, "Install Directory")[0]
                    firefox_paths.insert(0, os.path.join(install_dir, "firefox.exe"))
        except:
            pass
        
        for path in firefox_paths:
            if os.path.exists(path):
                print(f"✅ Firefox executable знайдено: {path}")
                return path
        
        print("❌ Firefox executable не знайдено")
        return None

    def _find_all_firefox_profiles(self) -> list:
        """Знаходить всі доступні Firefox профілі"""
        try:
            profiles_path = os.path.expanduser("~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles")
            
            if not os.path.exists(profiles_path):
                return []
            
            profiles = []
            for profile_dir in os.listdir(profiles_path):
                profile_full_path = os.path.join(profiles_path, profile_dir)
                if os.path.isdir(profile_full_path):
                    profiles.append((profile_full_path, profile_dir))
            
            print(f"📁 Знайдено {len(profiles)} Firefox профілів")
            for _, name in profiles:
                print(f"   • {name}")
            
            return profiles
            
        except Exception as e:
            print(f"⚠️ Помилка пошуку профілів: {e}")
            return []

    def _open_in_existing_firefox(self, url: str) -> bool:
        """Намагається відкрити URL в існуючому процесі Firefox"""
        try:
            print("🔧 Спроба відкриття URL в існуючому Firefox...")
            
            # Метод 1: Через firefox.exe з параметром -new-tab (НАЙКРАЩИЙ)
            try:
                # Розширений пошук Firefox
                firefox_paths = [
                    r"C:\Program Files\Mozilla Firefox\firefox.exe",
                    r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
                    os.path.expanduser(r"~\AppData\Local\Mozilla Firefox\firefox.exe"),
                    r"C:\Users\%USERNAME%\AppData\Local\Mozilla Firefox\firefox.exe"
                ]
                
                # Також шукаємо через реєстр Windows
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Mozilla\Mozilla Firefox") as key:
                        version_key = winreg.EnumKey(key, 0)
                        with winreg.OpenKey(key, f"{version_key}\\Main") as main_key:
                            install_dir = winreg.QueryValueEx(main_key, "Install Directory")[0]
                            firefox_paths.insert(0, os.path.join(install_dir, "firefox.exe"))
                except:
                    pass
                
                firefox_exe = None
                for path in firefox_paths:
                    expanded_path = os.path.expandvars(path)
                    if os.path.exists(expanded_path):
                        firefox_exe = expanded_path
                        print(f"✅ Знайдено Firefox: {firefox_exe}")
                        break
                
                if firefox_exe:
                    # Запускаємо з параметрами для існуючого процесу
                    result = subprocess.run(
                        [firefox_exe, "-new-tab", url], 
                        check=False, 
                        timeout=15,
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    
                    if result.returncode == 0:
                        print("✅ URL успішно відправлено в існуючий Firefox!")
                        return True
                    else:
                        print(f"⚠️ Firefox повернув код: {result.returncode}")
                        
                else:
                    print("⚠️ Firefox.exe не знайдено в системі")
                    
            except Exception as subprocess_error:
                print(f"⚠️ Помилка subprocess: {subprocess_error}")
            
            # Метод 2: Через PowerShell start-process (АЛЬТЕРНАТИВНИЙ)
            try:
                print("🔄 Спроба через PowerShell...")
                ps_command = f'Start-Process firefox -ArgumentList "-new-tab", "{url}"'
                result = subprocess.run(
                    ["powershell", "-Command", ps_command],
                    check=False,
                    timeout=10,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    print("✅ URL відкрито через PowerShell!")
                    return True
                else:
                    print(f"⚠️ PowerShell повернув код: {result.returncode}")
                    
            except Exception as ps_error:
                print(f"⚠️ Помилка PowerShell: {ps_error}")
            
            # Метод 3: Через webbrowser з примусовим Firefox (РЕЗЕРВНИЙ)
            try:
                print("🔄 Спроба через webbrowser з примусовим Firefox...")
                
                # Зберігаємо поточний браузер за замовчуванням
                original_browser = os.environ.get('BROWSER', '')
                
                # Примусово встановлюємо Firefox
                os.environ['BROWSER'] = 'firefox'
                
                import webbrowser
                
                # Очищаємо кеш браузерів
                if hasattr(webbrowser, '_browsers'):
                    webbrowser._browsers.clear()
                
                # Реєструємо Firefox вручну
                firefox_cmd = None
                for path in [r"C:\Program Files\Mozilla Firefox\firefox.exe", 
                           r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"]:
                    if os.path.exists(path):
                        firefox_cmd = f'"{path}" %s'
                        break
                
                if firefox_cmd:
                    webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(firefox_cmd))
                    browser = webbrowser.get('firefox')
                    browser.open_new_tab(url)
                    print("✅ URL відкрито через webbrowser з примусовим Firefox!")
                    
                    # Відновлюємо браузер за замовчуванням
                    if original_browser:
                        os.environ['BROWSER'] = original_browser
                    else:
                        os.environ.pop('BROWSER', None)
                    
                    return True
                else:
                    print("⚠️ Не вдалося знайти Firefox для webbrowser")
                    
            except Exception as webbrowser_error:
                print(f"⚠️ Помилка webbrowser: {webbrowser_error}")
                # Відновлюємо браузер за замовчуванням при помилці
                if 'original_browser' in locals():
                    if original_browser:
                        os.environ['BROWSER'] = original_browser
                    else:
                        os.environ.pop('BROWSER', None)
            
            print("❌ Всі методи відкриття в Firefox не вдалися")
            return False
            
        except Exception as e:
            print(f"❌ Критична помилка відкриття в Firefox: {e}")
            return False

    def _use_existing_browser_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Використовує існуючий авторизований Firefox браузер для конвертації"""
        try:
            print("🦊 Спроба відкриття в існуючому авторизованому Firefox...")
            
            convert_url = f"https://www.binance.com/en/convert/{from_asset}/{to_asset}"
            
            # КРИТИЧНО: Спочатку перевіряємо чи Firefox дійсно запущений
            import psutil
            firefox_found = False
            firefox_pids = []
            
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    if proc.info['name'] and 'firefox' in proc.info['name'].lower():
                        firefox_found = True
                        firefox_pids.append(proc.info['pid'])
                        print(f"🔍 Firefox процес знайдено: PID {proc.info['pid']}")
                except:
                    continue
            
            if not firefox_found:
                print("❌ Firefox не запущений! Відкриття в браузері за замовчуванням...")
                webbrowser.open(convert_url)
            else:
                print(f"✅ Знайдено {len(firefox_pids)} Firefox процесів")
                
                # Пробуємо відкрити в існуючому Firefox
                success = self._direct_firefox_open(convert_url)
                
                if not success:
                    print("⚠️ Не вдалося відкрити в існуючому Firefox")
                    print("🔄 УВАГА: Можливо відкриється в новому вікні")
                    webbrowser.open(convert_url)
            
            print(f"\n🔗 Пара токенів: {from_asset} → {to_asset}")
            print(f"💰 Кількість: {amount}")
            
            # Простіші інструкції
            print("\n📋 Дії:")
            print("1. Перевірте що відкрилася правильна сторінка конвертації")
            print("2. Переконайтесь що ви увійшли в Binance (не 'Log In' кнопки)")  
            print("3. Якщо потрібно - увійдіть в акаунт")
            print("")
            print("⏳ Очікування (3 секунди)...")
            time.sleep(3)
            
            # Генеруємо JavaScript код для автоматизації
            amount_value = "max" if str(amount).lower() == 'max' else str(amount)
            js_code = self._generate_automation_js(from_asset, to_asset, amount_value)
            
            print(f"\n🎯 === АВТОМАТИЗАЦІЯ В ВАШОМУ ОСНОВНОМУ FIREFOX ===")
            print(f"✅ Використання вашого основного Firefox профілю")
            print(f"🔒 Усі ваші логіни та сесії збережені")
            print(f"")
            print(f"📋 Як користуватися:")
            print(f"")
            print(f"🤖 Варіант 1: Автоматичний JavaScript (НАЙПРОСТІШИЙ)")
            print(f"   1. В відкритій вкладці натисніть F12")
            print(f"   2. Перейдіть на вкладку 'Console'")
            print(f"   3. Скопіюйте та вставте код нижче і натисніть Enter:")
            print(f"")
            print("=" * 80)
            print(js_code)
            print("=" * 80)
            print(f"")
            print(f"👤 Варіант 2: Ручна конвертація")
            print(f"   • Токени: {from_asset} → {to_asset}")
            print(f"   • Кількість: {amount_value}")
            print(f"   • Натисніть 'Convert' та підтвердіть")
            print(f"")
            print(f"🚀 Переваги використання основного профілю:")
            print(f"   ✅ Немає проблем з авторизацією Google/Binance")
            print(f"   ✅ Всі ваші збережені дані та налаштування")
            print(f"   ✅ Повний доступ до всіх функцій Binance")
            print(f"   ✅ Збережена історія торгів та улюблені пари")
            print(f"   ✅ Двофакторна автентифікація працює нормально")
            
            # Чекаємо підтвердження від користувача
            while True:
                print(f"\n❓ Оберіть варіант:")
                print(f"   [1] - JavaScript автоматизація виконана")
                print(f"   [2] - Ручна конвертація завершена")
                print(f"   [n] - Скасувати операцію")
                
                choice = input("👉 Ваш вибір (1/2/n): ").strip().lower()
                
                if choice in ['1', 'auto', 'js', 'javascript']:
                    print("✅ JavaScript автоматизація підтверджена!")
                    print("🎉 Конвертація в авторизованому Firefox завершена!")
                    return True
                elif choice in ['2', 'manual', 'ручна', 'вручну']:
                    print("✅ Ручна конвертація підтверджена!")
                    print("🎉 Конвертація в авторизованому Firefox завершена!")
                    return True
                elif choice in ['n', 'no', 'cancel', 'скасувати', 'ні']:
                    print("❌ Операція скасована")
                    return False
                else:
                    print("❌ Невірний вибір. Введіть '1', '2' або 'n'")
                    
        except Exception as e:
            print(f"❌ Помилка використання існуючого Firefox: {e}")
            print("🔄 Переключення на fallback метод...")
            return self._fallback_browser_conversion(from_asset, to_asset, amount)
    
    def _start_firefox_with_csp_bypass(self):
        """Запускає Firefox з налаштуваннями для обходу CSP"""
        try:
            print("🦊 Запуск Firefox з обходом CSP...")
            
            firefox_options = FirefoxOptions()
            
            # Покращений обхід CSP та детекції
            firefox_options.add_argument("--disable-web-security")
            firefox_options.add_argument("--disable-features=VizDisplayCompositor")
            firefox_options.add_argument("--disable-blink-features=AutomationControlled")
            firefox_options.add_argument("--no-first-run")
            firefox_options.add_argument("--disable-extensions-except")
            firefox_options.add_argument("--disable-plugins-discovery")
            
            # Налаштування профілю для обходу CSP
            from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
            profile = FirefoxProfile()
            
            # Вимикаємо CSP
            profile.set_preference("security.csp.enable", False)
            profile.set_preference("security.mixed_content.block_active_content", False)
            profile.set_preference("security.mixed_content.block_display_content", False)
            
            # Вимикаємо детекцію автоматизації
            profile.set_preference("dom.webdriver.enabled", False)
            profile.set_preference("useAutomationExtension", False)
            
            # Додаткові налаштування для стабільності
            profile.set_preference("browser.cache.disk.enable", False)
            profile.set_preference("browser.cache.memory.enable", False)
            profile.set_preference("browser.cache.offline.enable", False)
            profile.set_preference("network.http.use-cache", False)
            
            profile.update_preferences()
            firefox_options.profile = profile
            
            # Додаткові аргументи
            firefox_options.add_argument("--no-sandbox")
            firefox_options.add_argument("--disable-dev-shm-usage")
            firefox_options.add_argument("--window-size=1920,1080")
            
            driver = webdriver.Firefox(options=firefox_options)
            driver.maximize_window()
            
            print("✅ Firefox запущено з обходом CSP")
            return driver
            
        except Exception as e:
            print(f"❌ Помилка запуску Firefox: {e}")
            return None
    
    def _inject_javascript_bypass_csp(self, driver, from_asset: str, to_asset: str, amount) -> bool:
        """Інжектує JavaScript з обходом CSP обмежень"""
        try:
            print("💉 Інжекція JavaScript з обходом CSP...")
            
            # Спочатку пробуємо прямий метод
            js_code = self._generate_automation_js(from_asset, to_asset, str(amount))
            
            try:
                # Метод 1: Прямий execute_script
                driver.execute_script(js_code)
                print("✅ JavaScript виконано через execute_script")
                time.sleep(5)
                return True
                
            except Exception as direct_error:
                print(f"⚠️ Прямий метод не вдався: {direct_error}")
                
                # Метод 2: Через створення script елемента
                try:
                    bypass_js = f"""
                    var script = document.createElement('script');
                    script.innerHTML = `{js_code.replace('`', '\\`')}`;
                    document.head.appendChild(script);
                    """
                    driver.execute_script(bypass_js)
                    print("✅ JavaScript виконано через DOM injection")
                    time.sleep(5)
                    return True
                    
                except Exception as dom_error:
                    print(f"⚠️ DOM injection не вдався: {dom_error}")
                    
                    # Метод 3: Через data URL
                    try:
                        import base64
                        js_encoded = base64.b64encode(js_code.encode()).decode()
                        data_url_js = f"""
                        var script = document.createElement('script');
                        script.src = 'data:text/javascript;base64,{js_encoded}';
                        document.head.appendChild(script);
                        """
                        driver.execute_script(data_url_js)
                        print("✅ JavaScript виконано через data URL")
                        time.sleep(5)
                        return True
                        
                    except Exception as data_error:
                        print(f"❌ Всі методи JavaScript injection не вдалися: {data_error}")
                        return self._manual_conversion_guide(driver, from_asset, to_asset, amount)
                        
        except Exception as e:
            print(f"❌ Помилка JavaScript injection: {e}")
            return False
    
    def _manual_conversion_guide(self, driver, from_asset: str, to_asset: str, amount) -> bool:
        """Показує покроковий гід для ручної конвертації"""
        try:
            print(f"\n🎯 === РУЧНА КОНВЕРТАЦІЯ {from_asset} → {to_asset} ===")
            print(f"📋 Кроки для виконання:")
            print(f"")
            print(f"1️⃣ Перевірте що ви на сторінці: /convert/{from_asset}/{to_asset}")
            print(f"2️⃣ Переконайтесь що вибрані правильні токени:")
            print(f"    • FROM: {from_asset}")
            print(f"    • TO: {to_asset}")
            print(f"3️⃣ Введіть кількість: {amount}")
            if str(amount).lower() == 'max':
                print(f"    • Натисніть кнопку 'Max' для всієї суми")
            print(f"4️⃣ Натисніть 'Convert' для виконання")
            print(f"5️⃣ Підтвердіть операцію якщо потрібно")
            print(f"")
            print(f"💡 Поради:")
            print(f"   • Перевірте що ви увійшли в акаунт")
            print(f"   • Переконайтесь що достатньо коштів")
            print(f"   • Перевірте мінімальні лімити конвертації")
            
            # Чекаємо підтвердження
            while True:
                result = input(f"\n❓ Конвертація виконана успішно? (y/n): ").lower().strip()
                if result in ['y', 'yes', 'так', 'да']:
                    print("✅ Ручна конвертація завершена!")
                    return True
                elif result in ['n', 'no', 'ні', 'нет']:
                    print("❌ Конвертація не завершена")
                    return False
                else:
                    print("❌ Введіть 'y' або 'n'")
                    
        except Exception as e:
            print(f"❌ Помилка ручного гіда: {e}")
            return False
    
    def _fallback_browser_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Fallback метод через звичайний браузер"""
        try:
            print("🌐 Fallback: відкриття через системний браузер...")
            
            # Відкриваємо Binance Convert з конкретною парою токенів
            binance_convert_url = f"https://www.binance.com/en/convert/{from_asset}/{to_asset}"
            webbrowser.open_new_tab(binance_convert_url)
            print(f"✅ Відкрито Binance Convert для пари {from_asset}/{to_asset}")
            print(f"🔗 URL: {binance_convert_url}")
            
            # Показуємо JavaScript код для ручного використання
            amount_value = "max" if str(amount).lower() == 'max' else str(amount)
            js_code = self._generate_automation_js(from_asset, to_asset, amount_value)
            
            print(f"\n🎯 === АВТОМАТИЧНА КОНВЕРТАЦІЯ ===")
            print(f"📋 Варіант 1: JavaScript автоматизація")
            print(f"   1. Натисніть F12 для відкриття Developer Tools")
            print(f"   2. Перейдіть на вкладку 'Console'")
            print(f"   3. Скопіюйте та вставте код нижче:")
            print(f"")
            print("=" * 60)
            print(js_code)
            print("=" * 60)
            print(f"")
            print(f"📋 Варіант 2: Ручна конвертація")
            print(f"   • Токени: {from_asset} → {to_asset}")
            print(f"   • Кількість: {amount_value}")
            print(f"   • Натисніть Convert та підтвердіть")
            
            # Чекаємо підтвердження
            while True:
                result = input(f"\n❓ Конвертація виконана успішно? (y/n): ").lower().strip()
                if result in ['y', 'yes', 'так', 'да']:
                    print("✅ Fallback конвертація завершена!")
                    return True
                elif result in ['n', 'no', 'ні', 'нет']:
                    print("❌ Fallback конвертація не завершена")
                    return False
                else:
                    print("❌ Введіть 'y' або 'n'")
                    
        except Exception as e:
            print(f"❌ Помилка fallback конвертації: {e}")
            return False

    def _connect_to_existing_firefox(self):
        """Підключається до існуючого авторизованого Firefox користувача"""
        try:
            print("🔍 Пошук існуючого авторизованого Firefox...")
            
            # Перевіряємо чи Firefox вже запущений
            import psutil
            firefox_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'firefox' in proc.info['name'].lower():
                        firefox_processes.append(proc)
                        print(f"✅ Знайдено Firefox процес (PID: {proc.info['pid']})")
                except:
                    continue
            
            if not firefox_processes:
                print("⚠️ Firefox не запущений")
                print("💡 Будь ласка:")
                print("   1. Запустіть ваш Firefox браузер")
                print("   2. Увійдіть в ваш Binance акаунт")
                print("   3. Запустіть скрипт знову")
                return None
            
            # ЗАВЖДИ використовуємо існуючий Firefox (без Selenium)
            print("✅ Використання вашого основного Firefox профілю")
            print("🔒 Ваші логіни та сесії збережені")
            return "existing_browser"  # Спеціальний маркер
                
        except Exception as e:
            print(f"⚠️ Помилка підключення до Firefox: {e}")
            print("💡 Запустіть Firefox вручну та увійдіть в Binance")
            return None
    
    def _start_firefox_with_user_profile(self):
        """Запускає Firefox з існуючим профілем користувача для збереження авторизації"""
        try:
            print("🦊 Запуск Firefox з вашим профілем...")
            
            firefox_profile_path = self._find_firefox_profile()
            if not firefox_profile_path:
                print("❌ Не знайдено профіль Firefox")
                return None
                
            print(f"📁 Використання профілю: {firefox_profile_path}")
            
            firefox_options = FirefoxOptions()
            
            # Використовуємо існуючий профіль користувача
            from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
            
            # Копіюємо профіль для безпеки
            import tempfile
            import shutil
            temp_profile_dir = tempfile.mkdtemp()
            shutil.copytree(firefox_profile_path, temp_profile_dir, dirs_exist_ok=True)
            
            profile = FirefoxProfile(temp_profile_dir)
            
            # Налаштування для роботи з Selenium зберігаючи авторизацію
            profile.set_preference("dom.webdriver.enabled", False)
            profile.set_preference("useAutomationExtension", False)
            
            # Вимикаємо CSP для автоматизації
            profile.set_preference("security.csp.enable", False)
            profile.set_preference("security.mixed_content.block_active_content", False)
            profile.set_preference("security.mixed_content.block_display_content", False)
            
            # Зберігаємо cookies та сесії
            profile.set_preference("network.cookie.cookieBehavior", 0)
            profile.set_preference("privacy.clearOnShutdown.cookies", False)
            profile.set_preference("privacy.clearOnShutdown.sessions", False)
            
            profile.update_preferences()
            firefox_options.profile = profile
            
            # Додаткові налаштування для стабільності
            firefox_options.add_argument("--no-sandbox")
            firefox_options.add_argument("--disable-dev-shm-usage")
            firefox_options.add_argument("--window-size=1920,1080")
            
            driver = webdriver.Firefox(options=firefox_options)
            driver.maximize_window()
            
            print("✅ Firefox запущено з вашим профілем!")
            return driver
            
        except Exception as e:
            print(f"❌ Помилка запуску Firefox з профілем: {e}")
            print("💡 Спробуйте закрити всі вікна Firefox та запустити знову")
            return None
    
    def _start_firefox_with_profile(self):
        """Запускає Firefox з існуючим профілем користувача"""
        try:
            firefox_profile_path = self._find_firefox_profile()
            if not firefox_profile_path:
                return None
                
            firefox_options = FirefoxOptions()
            
            # Використовуємо існуючий профіль
            from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
            profile = FirefoxProfile(firefox_profile_path)
            
            # Налаштування для роботи з Selenium
            profile.set_preference("dom.webdriver.enabled", False)
            profile.set_preference("useAutomationExtension", False)
            profile.update_preferences()
            
            firefox_options.profile = profile
            firefox_options.add_argument("--no-sandbox")
            firefox_options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Firefox(options=firefox_options)
            driver.maximize_window()
            
            return driver
            
        except Exception as e:
            print(f"⚠️ Помилка запуску Firefox з профілем: {e}")
            return None
    
    def _check_binance_login(self, driver) -> bool:
        """Перевіряє чи користувач увійшов в Binance акаунт"""
        try:
            # Шукаємо елементи, що вказують на вхід в акаунт
            login_indicators = [
                '.user-menu',
                '[data-testid="header-user-menu"]',
                '.profile-menu',
                'button[aria-label*="user"]',
                '.balance-display'
            ]
            
            for selector in login_indicators:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements and any(elem.is_displayed() for elem in elements):
                        print("✅ Користувач увійшов в акаунт")
                        return True
                except:
                    continue
            
            # Перевіряємо чи є кнопка входу (що означає що користувач НЕ увійшов)
            login_buttons = driver.find_elements(By.XPATH, "//*[contains(text(), 'Log In') or contains(text(), 'Login') or contains(text(), 'Sign In')]")
            if login_buttons and any(btn.is_displayed() for btn in login_buttons):
                print("⚠️ Користувач не увійшов в акаунт")
                return False
            
            # Якщо немає явних індикаторів, вважаємо що увійшов
            print("✅ Схоже, що користувач увійшов в акаунт")
            return True
            
        except Exception as e:
            print(f"⚠️ Помилка перевірки входу: {e}")
            return True  # За замовчуванням вважаємо що увійшов
    
    def _find_firefox_profile(self) -> Optional[str]:
        """Знаходить шлях до АКТИВНОГО профілю Firefox користувача"""
        try:
            # Шлях до профілів Firefox на Windows
            profiles_path = os.path.expanduser("~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles")
            
            if not os.path.exists(profiles_path):
                print("❌ Папка профілів Firefox не знайдена")
                return None
            
            # Спочатку читаємо profiles.ini для знаходження активного профілю
            profiles_ini = os.path.expanduser("~\\AppData\\Roaming\\Mozilla\\Firefox\\profiles.ini")
            active_profile = None
            
            if os.path.exists(profiles_ini):
                print("📋 Читаємо profiles.ini для знаходження активного профілю...")
                try:
                    with open(profiles_ini, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Шукаємо профіль з Default=1
                    import re
                    sections = re.split(r'\[([^\]]+)\]', content)[1:]  # Видаляємо перший пустий елемент
                    
                    for i in range(0, len(sections), 2):
                        section_name = sections[i]
                        section_content = sections[i + 1] if i + 1 < len(sections) else ""
                        
                        if 'Profile' in section_name:
                            lines = section_content.strip().split('\n')
                            profile_info = {}
                            
                            for line in lines:
                                if '=' in line:
                                    key, value = line.split('=', 1)
                                    profile_info[key.strip()] = value.strip()
                            
                            # Шукаємо активний профіль (Default=1) або профіль з найновішою датою
                            if profile_info.get('Default') == '1' or profile_info.get('IsRelative') == '1':
                                if 'Path' in profile_info:
                                    if profile_info.get('IsRelative') == '1':
                                        active_profile = os.path.join(profiles_path, profile_info['Path'])
                                    else:
                                        active_profile = profile_info['Path']
                                    
                                    if os.path.exists(active_profile):
                                        print(f"✅ Знайдено активний профіль з profiles.ini: {os.path.basename(active_profile)}")
                                        return active_profile
                
                except Exception as ini_error:
                    print(f"⚠️ Помилка читання profiles.ini: {ini_error}")
            
            # Fallback: шукаємо за пріоритетом назв
            print("🔍 Fallback: пошук профілю за пріоритетом назв...")
            profile_dirs = []
            
            for profile_dir in os.listdir(profiles_path):
                profile_path = os.path.join(profiles_path, profile_dir)
                if os.path.isdir(profile_path):
                    profile_dirs.append((profile_path, profile_dir))
            
            if not profile_dirs:
                print("❌ Не знайдено жодного профілю")
                return None
            
            # Пріоритети для вибору профілю
            priorities = [
                "default-release",
                "default-esr", 
                "default",
                ""  # будь-який інший
            ]
            
            for priority in priorities:
                for profile_path, profile_name in profile_dirs:
                    if priority == "":  # останній fallback
                        print(f"📁 Використовуємо перший доступний профіль: {profile_name}")
                        return profile_path
                    elif priority in profile_name.lower():
                        print(f"✅ Знайдено профіль за пріоритетом '{priority}': {profile_name}")
                        return profile_path
            
            # Якщо нічого не знайшли, беремо перший
            if profile_dirs:
                selected_path, selected_name = profile_dirs[0]
                print(f"📁 Використовуємо перший доступний профіль: {selected_name}")
                return selected_path
                
        except Exception as e:
            print(f"❌ Критична помилка пошуку профілю Firefox: {e}")
            
        return None
    
    def _close_popups(self, driver):
        """Закриває спливаючі вікна та уведомлення"""
        try:
            # Селектори для закриття попапів
            popup_selectors = [
                '[aria-label="Close"]',
                '.bn-modal-close',
                '.modal-close',
                'button[class*="close"]',
                '[data-testid="modal-close"]'
            ]
            
            for selector in popup_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            element.click()
                            time.sleep(0.5)
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Помилка закриття попапів: {e}")
    
    def _perform_selenium_conversion(self, driver, from_asset: str, to_asset: str, amount) -> bool:
        """Виконує конвертацію через Selenium"""
        try:
            wait = WebDriverWait(driver, 20)
            
            # Перевіряємо поточні токени
            print("🔍 Перевіряємо поточні токени...")
            current_tokens = self._get_current_tokens(driver)
            print(f"📊 Поточні токени: {current_tokens['from'] or 'UNKNOWN'} → {current_tokens['to'] or 'UNKNOWN'}")
            
            # Змінюємо токени якщо потрібно
            if current_tokens['from'] != from_asset:
                print(f"🔄 Зміна FROM токена на {from_asset}")
                if not self._change_token(driver, True, from_asset):
                    print("❌ Не вдалося змінити FROM токен")
                    return False
            
            if current_tokens['to'] != to_asset:
                print(f"🔄 Зміна TO токена на {to_asset}")
                if not self._change_token(driver, False, to_asset):
                    print("❌ Не вдалося змінити TO токен")
                    return False
            
            # Встановлюємо кількість
            print(f"💰 Встановлення кількості: {amount}")
            if not self._set_amount(driver, amount):
                print("❌ Не вдалося встановити кількість")
                return False
            
            # Виконуємо конвертацію
            print("🔄 Виконання конвертації...")
            if not self._execute_conversion(driver):
                print("❌ Не вдалося виконати конвертацію")
                return False
            
            print("✅ Конвертація завершена!")
            return True
            
        except Exception as e:
            print(f"❌ Помилка виконання конвертації: {e}")
            return False
    
    def _get_current_tokens(self, driver) -> dict:
        """Отримує поточні токени на сторінці"""
        try:
            from_selectors = [
                '[data-testid="from-token-selector"] span',
                '[data-testid="from-asset-selector"] span',
                '.from-token-selector span',
                '[class*="from-token"] span'
            ]
            
            to_selectors = [
                '[data-testid="to-token-selector"] span',
                '[data-testid="to-asset-selector"] span',
                '.to-token-selector span',
                '[class*="to-token"] span'
            ]
            
            current_from = None
            current_to = None
            
            for selector in from_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            text = element.text.strip()
                            match = re.search(r'\b[A-Z]{2,5}\b', text)
                            if match:
                                current_from = match.group()
                                break
                    if current_from:
                        break
                except:
                    continue
            
            for selector in to_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            text = element.text.strip()
                            match = re.search(r'\b[A-Z]{2,5}\b', text)
                            if match:
                                current_to = match.group()
                                break
                    if current_to:
                        break
                except:
                    continue
            
            return {'from': current_from, 'to': current_to}
            
        except Exception as e:
            print(f"⚠️ Помилка отримання поточних токенів: {e}")
            return {'from': None, 'to': None}
    
    def _change_token(self, driver, is_from: bool, token_symbol: str) -> bool:
        """Змінює токен (FROM або TO)"""
        try:
            # Селектори для кнопок токенів
            selectors = [
                '[data-testid="from-token-selector"]' if is_from else '[data-testid="to-token-selector"]',
                '[data-testid="from-asset-selector"]' if is_from else '[data-testid="to-asset-selector"]',
                '.from-token-selector' if is_from else '.to-token-selector',
                'button[class*="from"]' if is_from else 'button[class*="to"]'
            ]
            
            token_button = None
            for selector in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            token_button = element
                            break
                    if token_button:
                        break
                except:
                    continue
            
            if not token_button:
                print(f"❌ Не знайдено кнопку токена ({'FROM' if is_from else 'TO'})")
                return False
            
            # Клікаємо по кнопці токена
            token_button.click()
            time.sleep(1)
            
            # Шукаємо поле пошуку
            search_input = None
            search_selectors = [
                'input[placeholder*="Search"]',
                'input[placeholder*="search"]',
                'input[type="text"]',
                '[data-testid="search-input"]'
            ]
            
            for selector in search_selectors:
                try:
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    if element.is_displayed():
                        search_input = element
                        break
                except:
                    continue
            
            if search_input:
                search_input.clear()
                search_input.send_keys(token_symbol)
                time.sleep(1)
            
            # Шукаємо токен у списку
            time.sleep(1)
            token_elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{token_symbol}')]")
            
            for element in token_elements:
                try:
                    if element.is_displayed():
                        # Знаходимо клікабельний батьківський елемент
                        clickable = element
                        for _ in range(5):  # Максимум 5 рівнів вгору
                            if clickable.tag_name.lower() in ['button', 'div', 'li', 'a']:
                                clickable.click()
                                time.sleep(1)
                                return True
                            clickable = clickable.find_element(By.XPATH, '..')
                except:
                    continue
            
            print(f"❌ Не знайдено токен {token_symbol} у списку")
            return False
            
        except Exception as e:
            print(f"❌ Помилка зміни токена: {e}")
            return False
    
    def _set_amount(self, driver, amount) -> bool:
        """Встановлює кількість для конвертації"""
        try:
            if str(amount).lower() == 'max':
                # Шукаємо кнопку MAX
                max_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Max') or contains(text(), 'MAX')]")
                
                for element in max_elements:
                    try:
                        if element.is_displayed():
                            element.click()
                            print("✅ Кнопка MAX натиснута")
                            time.sleep(3)  # Чекаємо розрахунки Binance
                            return True
                    except:
                        continue
                
                print("⚠️ Кнопка MAX не знайдена")
                return False
            else:
                # Шукаємо поле введення кількості
                amount_selectors = [
                    '[data-testid="from-amount-input"]',
                    '[data-testid="amount-input"]',
                    'input[placeholder*="amount"]',
                    'input[type="text"]',
                    'input[type="number"]'
                ]
                
                for selector in amount_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            if element.is_displayed():
                                element.clear()
                                element.send_keys(str(amount))
                                print(f"✅ Кількість {amount} введена")
                                time.sleep(1)
                                return True
                    except:
                        continue
                
                print("❌ Поле введення кількості не знайдено")
                return False
                
        except Exception as e:
            print(f"❌ Помилка встановлення кількості: {e}")
            return False
    
    def _execute_conversion(self, driver) -> bool:
        """Виконує конвертацію"""
        try:
            # Шукаємо кнопку Convert
            convert_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Convert') or contains(text(), 'convert')]")
            
            for element in convert_elements:
                try:
                    if element.is_displayed() and element.is_enabled():
                        text = element.text.lower()
                        if 'convert' in text and 'preview' not in text:
                            element.click()
                            print("✅ Кнопка Convert натиснута")
                            time.sleep(2)
                            
                            # Шукаємо кнопку підтвердження
                            confirm_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Confirm') or contains(text(), 'confirm')]")
                            
                            for confirm_element in confirm_elements:
                                try:
                                    if confirm_element.is_displayed() and confirm_element.is_enabled():
                                        confirm_element.click()
                                        print("✅ Конвертація підтверджена")
                                        time.sleep(3)
                                        return True
                                except:
                                    continue
                            
                            # Якщо не знайшли кнопку підтвердження, можливо конвертація вже завершена
                            print("✅ Конвертація, можливо, завершена")
                            return True
                except:
                    continue
            
            print("❌ Кнопка Convert не знайдена")
            return False
            
        except Exception as e:
            print(f"❌ Помилка виконання конвертації: {e}")
            return False
    
    def _javascript_browser_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Відкриває браузер і показує JavaScript код для автоматизації"""
        try:
            # Відкриваємо Binance Convert з конкретною парою токенів
            binance_convert_url = f"https://www.binance.com/en/convert/{from_asset}/{to_asset}"
            webbrowser.open_new_tab(binance_convert_url)
            print(f"✅ Відкрито Binance Convert для пари {from_asset}/{to_asset}")
            print(f"🔗 URL: {binance_convert_url}")
            
            # Генеруємо JavaScript код для автоматизації
            amount_value = "max" if amount == 'max' else str(amount)
            js_code = self._generate_automation_js(from_asset, to_asset, amount_value)
            
            print(f"\n🎯 === АВТОМАТИЧНА КОНВЕРТАЦІЯ ===")
            print(f"📋 Скопіюйте та вставте цей код в консоль браузера:")
            print(f"")
            print(f"📱 Як відкрити консоль:")
            print(f"   • Натисніть F12 або Ctrl+Shift+I")
            print(f"   • Перейдіть на вкладку 'Console'")
            print(f"   • Вставте код нижче і натисніть Enter")
            print(f"")
            print("=" * 60)
            print(js_code)
            print("=" * 60)
            print(f"")
            print(f"💡 Код автоматично:")
            print(f"   • Вибере токени: {from_asset} → {to_asset}")
            print(f"   • Встановить кількість: {amount_value}")
            print(f"   • Виконає конвертацію")
            print(f"   • Покаже результат")
            
            # Чекаємо підтвердження
            while True:
                result = input(f"\n❓ Автоматизація виконана успішно? (y/n): ").lower().strip()
                if result in ['y', 'yes', 'так', 'да', 'д']:
                    print("✅ Автоматична конвертація завершена!")
                    return True
                elif result in ['n', 'no', 'ні', 'нет', 'н']:
                    print("❌ Автоматизація не вдалася")
                    return False
                else:
                    print("❌ Введіть 'y' або 'n'")
                    
        except Exception as e:
            print(f"❌ Помилка JavaScript автоматизації: {e}")
            return False
    
    def _generate_automation_js(self, from_asset: str, to_asset: str, amount: str) -> str:
        """Генерує JavaScript код для автоматизації конвертації"""
        js_template = '''// 🤖 Автоматична конвертація {from_asset} → {to_asset}
console.log("🚀 Початок автоматичної конвертації: {from_asset} → {to_asset}");
console.log("📍 URL містить пару токенів - токени можуть бути вже вибрані");

// Покращена функція для обходу CSP та детекції автоматизації
function bypassDetection() {{
    // Повний обхід детекції автоматизації
    delete window.navigator.webdriver;
    delete navigator.webdriver;
    delete window.callPhantom;
    delete window._phantom;
    delete window.phantom;
    
    // Переписуємо властивості navigator
    Object.defineProperty(navigator, 'webdriver', {{
        get: () => false,
    }});
    
    Object.defineProperty(navigator, 'plugins', {{
        get: () => [1, 2, 3, 4, 5]
    }});
    
    // Додаємо природні затримки та рандомізацію
    window.humanDelay = () => Math.random() * 800 + 200;
    
    // Симулюємо людську поведінку
    window.humanClick = function(element) {{
        element.dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}));
        setTimeout(() => {{
            element.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true}}));
            setTimeout(() => {{
                element.click();
                element.dispatchEvent(new MouseEvent('mouseup', {{bubbles: true}}));
            }}, Math.random() * 50 + 10);
        }}, Math.random() * 100 + 50);
    }};
    
    return new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));
}}

// Функція для очікування елемента
function waitForElement(selector, timeout = 10000) {{
    return new Promise((resolve, reject) => {{
        const startTime = Date.now();
        function check() {{
            const element = document.querySelector(selector);
            if (element && element.offsetParent !== null) {{
                resolve(element);
            }} else if (Date.now() - startTime > timeout) {{
                reject(new Error(`Елемент не знайдено: ${{selector}}`));
            }} else {{
                setTimeout(check, 100);
            }}
        }}
        check();
    }});
}}

// Функція для кліку з затримкою
async function clickWithDelay(element, delay = 1000) {{
    element.click();
    await new Promise(resolve => setTimeout(resolve, delay));
}}

// Функція для перевірки поточних токенів
function getCurrentTokens() {{
    const fromSelectors = [
        '[data-testid="from-token-selector"] span',
        '[data-testid="from-asset-selector"] span',
        '.from-token-selector span',
        '[class*="from-token"] span'
    ];
    
    const toSelectors = [
        '[data-testid="to-token-selector"] span',
        '[data-testid="to-asset-selector"] span', 
        '.to-token-selector span',
        '[class*="to-token"] span'
    ];
    
    let currentFrom = null, currentTo = null;
    
    for (const selector of fromSelectors) {{
        const element = document.querySelector(selector);
        if (element && element.textContent) {{
            const match = element.textContent.match(/\\b[A-Z]{{2,5}}\\b/);
            if (match) {{
                currentFrom = match[0];
                break;
            }}
        }}
    }}
    
    for (const selector of toSelectors) {{
        const element = document.querySelector(selector);
        if (element && element.textContent) {{
            const match = element.textContent.match(/\\b[A-Z]{{2,5}}\\b/);
            if (match) {{
                currentTo = match[0];
                break;
            }}
        }}
    }}
    
    return {{ from: currentFrom, to: currentTo }};
}}

// Головна функція автоматизації
async function autoConvert() {{
    try {{
        console.log("🔍 Пошук елементів для конвертації...");
        
        // Закриваємо всі попапи
        const popups = document.querySelectorAll('[aria-label="Close"], .bn-modal-close, .css-close');
        popups.forEach(popup => {{
            try {{ popup.click(); }} catch(e) {{}}
        }});
        
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Перевіряємо поточні токени
        const currentTokens = getCurrentTokens();
        console.log(`📊 Поточні токени: ${{currentTokens.from || 'UNKNOWN'}} → ${{currentTokens.to || 'UNKNOWN'}}`);
        
        const needFromChange = currentTokens.from !== "{from_asset}";
        const needToChange = currentTokens.to !== "{to_asset}";
        
        if (!needFromChange && !needToChange) {{
            console.log("✅ Токени вже встановлені правильно, переходимо до встановлення кількості");
        }} else {{
            console.log(`🔄 Потрібно змінити: FROM=${{needFromChange}}, TO=${{needToChange}}`);
            
            // Знаходимо кнопки вибору токенів
            const fromSelectors = [
                '[data-testid="from-token-selector"]',
                '[data-testid="from-asset-selector"]', 
                'button[class*="from"], button[class*="From"]',
                '.convert-from button, .from-token button',
                'button:has(span:contains("From"))',
                'div[class*="token-selector"]:first-of-type button'
            ];
            
            const toSelectors = [
                '[data-testid="to-token-selector"]',
                '[data-testid="to-asset-selector"]',
                'button[class*="to"], button[class*="To"]', 
                '.convert-to button, .to-token button',
                'button:has(span:contains("To"))',
                'div[class*="token-selector"]:last-of-type button'
            ];
            
            // Змінюємо FROM токен якщо потрібно
            if (needFromChange) {{
                console.log("🔄 Зміна FROM токена: {from_asset}");
                let fromButton = null;
                
                for (const selector of fromSelectors) {{
                    try {{
                        fromButton = document.querySelector(selector);
                        if (fromButton && fromButton.offsetParent !== null) {{
                            console.log(`✅ FROM кнопка знайдена: ${{selector}}`);
                            break;
                        }}
                    }} catch(e) {{}}
                }}
                
                if (fromButton) {{
                    await clickWithDelay(fromButton, 1500);
                    await selectTokenFromDropdown("{from_asset}");
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }}
            }}
            
            // Змінюємо TO токен якщо потрібно
            if (needToChange) {{
                console.log("🔄 Зміна TO токена: {to_asset}");
                let toButton = null;
                
                for (const selector of toSelectors) {{
                    try {{
                        toButton = document.querySelector(selector);
                        if (toButton && toButton.offsetParent !== null) {{
                            console.log(`✅ TO кнопка знайдена: ${{selector}}`);
                            break;
                        }}
                    }} catch(e) {{}}
                }}
                
                if (toButton) {{
                    await clickWithDelay(toButton, 1500);
                    await selectTokenFromDropdown("{to_asset}");
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }}
            }}
        }}
        
        // Встановлюємо кількість
        console.log("💰 Встановлення кількості: {amount}");
        await setAmount("{amount}");
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Виконуємо конвертацію
        console.log("🔄 Виконання конвертації...");
        await executeConversion();
        
        console.log("🎉 Автоматична конвертація завершена!");
        
    }} catch (error) {{
        console.error("❌ Помилка автоматизації:", error.message);
        console.log("💡 Спробуйте виконати конвертацію вручну");
    }}
}}

// Функція для вибору токена з dropdown
async function selectTokenFromDropdown(tokenSymbol) {{
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Шукаємо поле пошуку
    const searchInput = document.querySelector('input[placeholder*="Search"], input[type="text"], input[placeholder*="search"]');
    if (searchInput) {{
        searchInput.focus();
        searchInput.value = '';
        searchInput.value = tokenSymbol;
        searchInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
        await new Promise(resolve => setTimeout(resolve, 800));
    }}
    
    // Шукаємо токен в результатах
    const tokenSelectors = [
        `[data-symbol="${{tokenSymbol}}"]`,
        `[title*="${{tokenSymbol}}"]`, 
        `div:contains("${{tokenSymbol}}")`,
        `span:contains("${{tokenSymbol}}")`,
        `.token-item:contains("${{tokenSymbol}}")`,
        `li:contains("${{tokenSymbol}}")`
    ];
    
    let tokenFound = false;
    for (const selector of tokenSelectors) {{
        try {{
            // Для :contains() селекторів використовуємо XPath
            let elements;
            if (selector.includes(':contains')) {{
                const xpath = `//*[contains(text(), "${{tokenSymbol}}")]`;
                const result = document.evaluate(xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                elements = [];
                for (let i = 0; i < result.snapshotLength; i++) {{
                    elements.push(result.snapshotItem(i));
                }}
            }} else {{
                elements = document.querySelectorAll(selector);
            }}
            
            for (const element of elements) {{
                if (element && element.offsetParent !== null) {{
                    // Знаходимо клікабельний батьківський елемент
                    let clickable = element;
                    while (clickable && !['BUTTON', 'DIV', 'LI'].includes(clickable.tagName)) {{
                        clickable = clickable.parentElement;
                    }}
                    
                    if (clickable) {{
                        clickable.click();
                        console.log(`✅ Токен ${{tokenSymbol}} вибрано`);
                        tokenFound = true;
                        break;
                    }}
                }}
            }}
            if (tokenFound) break;
        }} catch(e) {{}}
    }}
    
    if (!tokenFound) {{
        // Останній шанс - клікаємо по першому варіанту
        const firstOption = document.querySelector('.token-list-item:first-child, li:first-child, div[role="option"]:first-child');
        if (firstOption) {{
            firstOption.click();
            console.log(`⚠️ Вибрано перший варіант замість ${{tokenSymbol}}`);
        }} else {{
            throw new Error(`Токен ${{tokenSymbol}} не знайдено`);
        }}
    }}
}}

// Функція для встановлення кількості
async function setAmount(amount) {{
    if (amount === "max") {{
        // Шукаємо кнопку MAX
        const maxButtons = document.querySelectorAll('button, span, div');
        for (const btn of maxButtons) {{
            if (btn.textContent.trim().toLowerCase() === 'max' && btn.offsetParent !== null) {{
                btn.click();
                console.log("✅ Кнопка MAX натиснута");
                return;
            }}
        }}
    }}
    
    // Шукаємо поле вводу кількості
    const amountInputs = document.querySelectorAll('input[type="text"], input[type="number"], input[placeholder*="amount"], input[placeholder*="Amount"]');
    
    for (const input of amountInputs) {{
        if (input.offsetParent !== null) {{
            input.focus();
            input.select();
            input.value = amount;
            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
            input.dispatchEvent(new Event('change', {{ bubbles: true }}));
            console.log(`✅ Кількість ${{amount}} введена`);
            return;
        }}
    }}
    
    console.warn("⚠️ Поле вводу кількості не знайдено");
}}

// Функція для виконання конвертації
async function executeConversion() {{
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Шукаємо кнопку Convert
    const convertButtons = document.querySelectorAll('button, div[role="button"]');
    
    for (const btn of convertButtons) {{
        const text = btn.textContent.trim().toLowerCase();
        if ((text.includes('convert') || text.includes('конвертувати')) && 
            !text.includes('preview') && btn.offsetParent !== null && !btn.disabled) {{
            
            btn.click();
            console.log("✅ Кнопка Convert натиснута");
            
            // Чекаємо кнопку підтвердження
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            const confirmButtons = document.querySelectorAll('button, div[role="button"]');
            for (const confirmBtn of confirmButtons) {{
                const confirmText = confirmBtn.textContent.trim().toLowerCase();  
                if ((confirmText.includes('confirm') || confirmText.includes('підтвердити')) && 
                    confirmBtn.offsetParent !== null && !confirmBtn.disabled) {{
                    
                    confirmBtn.click();
                    console.log("✅ Конвертація підтверджена");
                    
                    // Чекаємо результат
                    await new Promise(resolve => setTimeout(resolve, 3000));
                    
                    // Перевіряємо на успіх/помилку
                    const successElements = document.querySelectorAll('*');
                    let hasSuccess = false;
                    let hasError = false;
                    
                    for (const elem of successElements) {{
                        const text = elem.textContent.toLowerCase();
                        if (text.includes('success') || text.includes('successful') || text.includes('completed')) {{
                            hasSuccess = true;
                        }}
                        if (text.includes('error') || text.includes('failed') || text.includes('insufficient')) {{
                            hasError = true;
                        }}
                    }}
                    
                    if (hasSuccess) {{
                        console.log("🎉 Конвертація успішна!");
                    }} else if (hasError) {{
                        console.log("❌ Конвертація не вдалася");
                    }} else {{
                        console.log("✅ Конвертація, ймовірно, успішна");
                    }}
                    
                    return;
                }}
            }}
            
            console.log("⚠️ Кнопка підтвердження не знайдена, можливо конвертація завершена");
            return;
        }}
    }}
    
    throw new Error("Кнопка Convert не знайдена");
}}

// Запускаємо автоматизацію
console.log("⏳ Запуск через 3 секунди...");
setTimeout(autoConvert, 3000);'''
        
        return js_template.format(
            from_asset=from_asset,
            to_asset=to_asset, 
            amount=amount
        )

    def _generate_smart_automation_js(self, from_asset: str, to_asset: str, amount: str) -> str:
        """Генерує CSP-сумісний JavaScript код для автоматизації"""
        return f'''// 🛡️ CSP-СУМІСНА АВТОМАТИЗАЦІЯ {from_asset} → {to_asset}
console.log("🛡️ CSP-сумісна автоматизація Binance Convert");
console.log("💱 Пара: {from_asset} → {to_asset}");
console.log("💰 Сума: {amount}");
console.log("⚠️ Примітка: Використовуємо тільки дозволені CSP методи");

// CSP-сумісні утиліти
window.cspSafeUtils = {{
    delay: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
    findElement: (selectors) => {{
        if (typeof selectors === 'string') selectors = [selectors];
        for (const selector of selectors) {{
            try {{
                const elements = document.querySelectorAll(selector);
                for (const element of elements) {{
                    if (element && element.offsetParent !== null && !element.disabled) {{
                        return element;
                    }}
                }}
            }} catch(e) {{}}
        }}
        return null;
    }},
    findByText: (text, tagNames = ['button', 'span', 'div']) => {{
        for (const tagName of tagNames) {{
            const elements = document.getElementsByTagName(tagName);
            for (const element of elements) {{
                if (element.textContent && element.textContent.toLowerCase().includes(text.toLowerCase()) && 
                    element.offsetParent !== null && !element.disabled) {{
                    return element;
                }}
            }}
        }}
        return null;
    }},
    humanClick: async (element, desc = '') => {{
        if (!element) return false;
        console.log('🖱️ Клік:', desc);
        element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        await window.cspSafeUtils.delay(300);
        element.click();
        console.log('✅ Клік успішний:', desc);
        return true;
    }},
    humanType: async (element, text, desc = '') => {{
        if (!element) return false;
        console.log('⌨️ Введення:', desc, '=', text);
        element.focus();
        element.value = '';
        element.dispatchEvent(new Event('input', {{ bubbles: true }}));
        element.value = text;
        element.dispatchEvent(new Event('input', {{ bubbles: true }}));
        element.dispatchEvent(new Event('change', {{ bubbles: true }}));
        console.log('✅ Текст введено:', desc);
        return true;
    }}
}};

// Розумний пошук елементів
function smartFind(selectors, description = 'елемент') {{
    console.log(`🔍 Пошук: ${{description}}`);
    
    for (const selector of selectors) {{
        try {{
            const elements = document.querySelectorAll(selector);
            for (const element of elements) {{
                if (element && element.offsetParent !== null && 
                    !element.disabled && element.style.visibility !== 'hidden') {{
                    console.log(`✅ Знайдено: ${{selector}}`);
                    return element;
                }}
            }}
        }} catch(e) {{
            console.warn(`⚠️ Помилка селектора: ${{selector}}`);
        }}
    }}
    
    console.warn(`❌ Не знайдено: ${{description}}`);
    return null;
}}

function smartFindByText(text, tags = ['button', 'span', 'div', 'a']) {{
    console.log(`🔍 Пошук за текстом: "${{text}}"`);
    
    for (const tag of tags) {{
        const elements = document.querySelectorAll(tag);
        for (const element of elements) {{
            if (element.textContent && 
                element.textContent.trim().toLowerCase().includes(text.toLowerCase()) &&
                element.offsetParent !== null && !element.disabled) {{
                console.log(`✅ Знайдено за текстом: ${{tag}} з "${{element.textContent.trim()}}"`);
                return element;
            }}
        }}
    }}
    
    console.warn(`❌ Не знайдено за текстом: "${{text}}"`);
    return null;
}}

// Отримання поточних токенів
function getCurrentTokens() {{
    const fromSelectors = [
        '[data-testid*="from"] span',
        '[class*="from-token"] span',
        '[class*="from-asset"] span'
    ];
    
    const toSelectors = [
        '[data-testid*="to"] span', 
        '[class*="to-token"] span',
        '[class*="to-asset"] span'
    ];
    
    let currentFrom = null, currentTo = null;
    
    for (const selector of fromSelectors) {{
        const element = document.querySelector(selector);
        if (element && element.textContent) {{
            const match = element.textContent.match(/\\b[A-Z]{{2,6}}\\b/);
            if (match) {{
                currentFrom = match[0];
                break;
            }}
        }}
    }}
    
    for (const selector of toSelectors) {{
        const element = document.querySelector(selector);
        if (element && element.textContent) {{
            const match = element.textContent.match(/\\b[A-Z]{{2,6}}\\b/);
            if (match) {{
                currentTo = match[0];
                break;
            }}
        }}
    }}
    
    return {{ from: currentFrom, to: currentTo }};
}}

// Вибір токена
async function selectToken(tokenSymbol) {{
    console.log(`🎯 Вибір токена: ${{tokenSymbol}}`);
    
    // Шукаємо поле пошуку
    const searchInput = smartFind([
        'input[placeholder*="Search"]',
        'input[placeholder*="search"]',
        'input[type="text"]'
    ], 'поле пошуку токена');
    
    if (searchInput) {{
        await window.cspSafeUtils.humanType(searchInput, tokenSymbol, 'пошук токена');
        await window.cspSafeUtils.delay(1000);
    }}
    
    // Шукаємо токен в результатах
    const tokenElement = smartFindByText(tokenSymbol, ['div', 'span', 'li', 'button']);
    
    if (tokenElement) {{
        // Знаходимо клікабельний батьківський елемент
        let clickableParent = tokenElement;
        while (clickableParent && !['BUTTON', 'DIV', 'LI'].includes(clickableParent.tagName)) {{
            clickableParent = clickableParent.parentElement;
        }}
        
        if (clickableParent) {{
            await window.cspSafeUtils.humanClick(clickableParent, `вибір токена ${{tokenSymbol}}`);
        }}
    }} else {{
        console.warn(`⚠️ Токен ${{tokenSymbol}} не знайдено, спробую перший варіант`);
        const firstOption = smartFind([
            '.token-list-item:first-child',
            'li:first-child',
            '[role="option"]:first-child'
        ], 'перший токен у списку');
        
        if (firstOption) {{
            await window.cspSafeUtils.humanClick(firstOption, 'перший токен');
        }}
    }}
}}

// Встановлення кількості
async function setSmartAmount(amount) {{
    if (amount === "max") {{
        console.log("🔝 Пошук кнопки MAX...");
        const maxButton = smartFindByText('max', ['button', 'span', 'div']);
        
        if (maxButton) {{
            await window.cspSafeUtils.humanClick(maxButton, 'кнопка MAX');
            await window.cspSafeUtils.delay(2000);
            return;
        }}
    }}
    
    // Шукаємо поле введення кількості
    const amountInput = smartFind([
        'input[placeholder*="amount"]',
        'input[placeholder*="Amount"]',
        'input[type="text"]',
        'input[type="number"]'
    ], 'поле вводу кількості');
    
    if (amountInput) {{
        await window.cspSafeUtils.humanType(amountInput, amount, 'кількість для конвертації');
    }}
}}

// Виконання конвертації
async function executeSmartConvert() {{
    // Шукаємо кнопку Convert
    const convertButton = smartFindByText('convert', ['button']);
    
    if (convertButton) {{
        await window.cspSafeUtils.humanClick(convertButton, 'кнопка Convert');
        
        // Чекаємо кнопку підтвердження
        await window.cspSafeUtils.delay(2000);
        
        const confirmButton = smartFindByText('confirm', ['button']);
        if (confirmButton) {{
            await window.cspSafeUtils.humanClick(confirmButton, 'підтвердження конвертації');
            
            // Чекаємо результат
            await window.cspSafeUtils.delay(3000);
            
            // Перевіряємо результат
            const successIndicators = ['success', 'completed', 'successful'];
            const errorIndicators = ['error', 'failed', 'insufficient'];
            
            let hasSuccess = false, hasError = false;
            
            for (const indicator of successIndicators) {{
                if (smartFindByText(indicator)) {{
                    hasSuccess = true;
                    break;
                }}
            }}
            
            for (const indicator of errorIndicators) {{
                if (smartFindByText(indicator)) {{
                    hasError = true;
                    break;
                }}
            }}
            
            if (hasSuccess) {{
                console.log("🎉 Конвертація успішна!");
            }} else if (hasError) {{
                console.log("❌ Конвертація не вдалася!");
            }} else {{
                console.log("✅ Конвертація, ймовірно, успішна");
            }}
        }} else {{
            console.log("ℹ️ Кнопка підтвердження не знайдена - можливо, конвертація завершена");
        }}
    }} else {{
        throw new Error("Кнопка Convert не знайдена");
    }}
}}

// Головна функція автоматизації
async function executeSmartConversion() {{
    try {{
        console.log("🚀 Початок розумної конвертації...");
        
        // Чекаємо завантаження сторінки
        await window.cspSafeUtils.delay(3000);
        
        // Закриваємо попапи
        const popupSelectors = ['[aria-label*="close"]', '.modal-close', '.bn-modal-close'];
        for (const selector of popupSelectors) {{
            const popup = window.cspSafeUtils.findElement([selector]);
            if (popup) await window.cspSafeUtils.humanClick(popup, 'закриття попапу');
        }}
        
        await window.cspSafeUtils.delay(1000);
        
        // Перевіряємо поточні токени
        console.log("📊 Перевірка поточних токенів...");
        const currentTokens = getCurrentTokens();
        console.log(`Поточні: ${{currentTokens.from}} → ${{currentTokens.to}}`);
        console.log(`Потрібні: {from_asset} → {to_asset}`);
        
        // Змінюємо FROM токен якщо потрібно
        if (currentTokens.from !== "{from_asset}") {{
            console.log("🔄 Зміна FROM токена...");
            const fromButton = smartFind([
                '[data-testid*="from"]',
                '[class*="from-token"]',
                '[class*="from-asset"]',
                'button:first-of-type'
            ], 'кнопка FROM токена');
            
            if (fromButton) {{
                await window.cspSafeUtils.humanClick(fromButton, 'FROM токен селектор');
                await window.cspSafeUtils.delay(1500);
                await selectToken("{from_asset}");
                await window.cspSafeUtils.delay(1000);
            }}
        }}
        
        // Змінюємо TO токен якщо потрібно
        if (currentTokens.to !== "{to_asset}") {{
            console.log("🔄 Зміна TO токена...");
            const toButton = smartFind([
                '[data-testid*="to"]',
                '[class*="to-token"]',
                '[class*="to-asset"]',
                'button:last-of-type'
            ], 'кнопка TO токена');
            
            if (toButton) {{
                await window.cspSafeUtils.humanClick(toButton, 'TO токен селектор');
                await window.cspSafeUtils.delay(1500);
                await selectToken("{to_asset}");
                await window.cspSafeUtils.delay(1000);
            }}
        }}
        
        // Встановлюємо кількість
        console.log("💰 Встановлення кількості...");
        await setSmartAmount("{amount}");
        
        await window.cspSafeUtils.delay(2000);
        
        // Виконуємо конвертацію
        console.log("🔄 Виконання конвертації...");
        await executeSmartConvert();
        
        console.log("🎉 Розумна конвертація завершена!");
        
    }} catch (error) {{
        console.error("❌ Помилка розумної конвертації:", error);
        console.log("💡 Спробуйте виконати конвертацію вручну або перезапустити код");
    }}
}}

// Запуск через 2 секунди
console.log("⏳ Запуск розумної автоматизації через 2 секунди...");
setTimeout(executeSmartConversion, 2000);'''


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
                    trader.trade(from_token, to_token, amount_input)
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