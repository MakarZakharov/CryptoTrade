import os
import time
import hashlib
import hmac
import urllib.parse
import requests
from typing import Dict, Optional, List, Union
import json
import logging
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
import ccxt

load_dotenv()


class MexcOrderType(Enum):
    """Типи ордерів MEXC"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class MexcOrderSide(Enum):
    """Сторони ордера"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class MexcConfig:
    """Конфігурація для MEXC клієнта"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: bool = True
    user_agent: str = "MexcTrader/1.0"


@dataclass
class OrderRequest:
    """Запит на створення ордера"""
    symbol: str
    side: MexcOrderSide
    order_type: MexcOrderType
    quantity: Optional[str] = None
    quote_order_qty: Optional[str] = None
    price: Optional[str] = None
    new_client_order_id: Optional[str] = None


@dataclass
class Balance:
    """Інформація про баланс"""
    asset: str
    free: float
    locked: float
    total: float


class MexcAPIException(Exception):
    """Виняток для помилок MEXC API"""
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class MexcClient:
    """Клієнт для роботи з MEXC API"""

    # MEXC API endpoints
    BASE_URL = "https://api.mexc.com"
    TESTNET_URL = "https://contract.mexc.com"  # MEXC testnet URL
    API_VERSION = "v3"

    def __init__(self, config: Optional[MexcConfig] = None):
        """
        Ініціалізація MEXC клієнта
        
        Args:
            config: Конфігурація клієнта
        """
        self.config = config or MexcConfig()
        
        # Отримання API ключів з .env файлу або конфігурації
        self.api_key = self.config.api_key or os.getenv('MEXC_API_KEY')
        self.api_secret = self.config.api_secret or os.getenv('MEXC_API_SECRET')
        
        # Вибір URL залежно від режиму
        self.base_url = self.TESTNET_URL if self.config.testnet else self.BASE_URL
        
        # Ініціалізація logger
        self.logger = self._setup_logger()
        
        # Кеш для торгових пар та курсів
        self._symbols_cache: Dict = {}
        self._ticker_cache: Dict = {}
        self._cache_timestamp = 0
        self.cache_ttl = 60  # Кеш на 1 хвилину
        
        # CCXT клієнт для резервного використання
        self.ccxt_client: Optional[ccxt.mexc] = None
        
        # Перевірка підключення
        self._validate_connection()

    def _setup_logger(self) -> logging.Logger:
        """Налаштування логера"""
        logger = logging.getLogger(f'MexcClient_{id(self)}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _validate_connection(self) -> None:
        """Перевірка підключення до API"""
        if self.config.testnet:
            self.logger.info("🧪 MEXC клієнт в тестовому режимі")
            return
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("⚠️ API ключі не налаштовані - доступні тільки публічні методи")
            return
        
        try:
            # Ініціалізація CCXT клієнта
            self._init_ccxt_client()
            
            # Тест API підключення
            account_response = self._private_request('GET', '/api/v3/account')
            if 'balances' not in account_response:
                raise MexcAPIException("Неочікувана відповідь API")
            
            self.logger.info("✅ MEXC API успішно підключено")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка підключення до MEXC API: {e}")
            if not self.config.testnet:
                raise MexcAPIException(f"Неможливо підключитися до MEXC API: {e}")

    def _init_ccxt_client(self) -> None:
        """Ініціалізація CCXT клієнта"""
        if not self.api_key or not self.api_secret:
            return
        
        try:
            self.ccxt_client = ccxt.mexc({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.config.testnet,
                'enableRateLimit': self.config.rate_limit,
                'timeout': self.config.timeout * 1000,
                'options': {'defaultType': 'spot'}
            })
            
            if not self.config.testnet:
                self.ccxt_client.load_markets()
                self.logger.info("✅ CCXT MEXC клієнт ініціалізовано")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Не вдалося ініціалізувати CCXT: {e}")
            self.ccxt_client = None

    def _generate_signature(self, query_string: str, timestamp: str) -> str:
        """Генерація підпису для приватних запитів"""
        string_to_sign = timestamp + 'GET' + '/api/v3/account' + query_string
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _public_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Виконання публічного запиту до MEXC API"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, params=params, timeout=self.config.timeout)
                else:
                    response = requests.post(url, json=params, timeout=self.config.timeout)
                
                response.raise_for_status()
                data = response.json()
                
                # MEXC повертає різні формати помилок
                if isinstance(data, dict) and 'code' in data and data['code'] != 200:
                    raise MexcAPIException(f"API помилка: {data.get('msg', 'Невідома помилка')}", data.get('code'))
                
                return data
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise MexcAPIException(f"Помилка мережі: {e}")
                
                self.logger.warning(f"Повтор запиту через {self.config.retry_delay}с...")
                time.sleep(self.config.retry_delay)
        
        raise MexcAPIException("Максимальна кількість спроб вичерпана")

    def _private_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Виконання приватного запиту до MEXC API"""
        if not self.api_key or not self.api_secret:
            raise MexcAPIException("API ключі не налаштовані")
        
        if params is None:
            params = {}
        
        timestamp = str(int(time.time() * 1000))
        params['timestamp'] = timestamp
        
        # Створення query string
        query_string = urllib.parse.urlencode(params)
        
        # Генерація підпису (MEXC використовує специфічний формат)
        string_to_sign = f"timestamp={timestamp}"
        if len(params) > 1:  # якщо є інші параметри окрім timestamp
            string_to_sign = query_string
            
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {
            'X-MEXC-APIKEY': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': self.config.user_agent
        }
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, params=params, headers=headers, timeout=self.config.timeout)
                elif method.upper() == 'POST':
                    response = requests.post(url, params=params, headers=headers, timeout=self.config.timeout)
                elif method.upper() == 'DELETE':
                    response = requests.delete(url, params=params, headers=headers, timeout=self.config.timeout)
                else:
                    raise MexcAPIException(f"Непідтримуваний HTTP метод: {method}")
                
                response.raise_for_status()
                data = response.json()
                
                # MEXC повертає різні формати помилок
                if isinstance(data, dict) and 'code' in data and data['code'] != 200:
                    raise MexcAPIException(f"API помилка: {data.get('msg', 'Невідома помилка')}", data.get('code'))
                
                return data
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise MexcAPIException(f"Помилка мережі: {e}")
                
                self.logger.warning(f"Повтор запиту через {self.config.retry_delay}с...")
                time.sleep(self.config.retry_delay)
        
        raise MexcAPIException("Максимальна кількість спроб вичерпана")

    def _update_cache(self) -> None:
        """Оновлення кешу торгових пар та курсів"""
        current_time = time.time()
        if current_time - self._cache_timestamp < self.cache_ttl:
            return
        
        try:
            # Оновлення торгових пар
            exchange_info = self._public_request('GET', '/api/v3/exchangeInfo')
            if 'symbols' in exchange_info:
                self._symbols_cache = {symbol['symbol']: symbol for symbol in exchange_info['symbols']}
            
            # Оновлення тікерів
            ticker_data = self._public_request('GET', '/api/v3/ticker/24hr')
            if isinstance(ticker_data, list):
                self._ticker_cache = {ticker['symbol']: ticker for ticker in ticker_data}
            
            self._cache_timestamp = current_time
            self.logger.info("✅ Кеш MEXC даних оновлено")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка оновлення кешу: {e}")

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Отримання інформації про торгову пару"""
        self._update_cache()
        return self._symbols_cache.get(symbol)

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Отримання інформації про тікер"""
        self._update_cache()
        return self._ticker_cache.get(symbol)

    def get_balance(self, asset: str) -> Union[Balance, float]:
        """Отримання балансу активу"""
        if self.config.testnet:
            # Тестовий баланс для демонстрації
            test_balances = {
                'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 
                'MX': 1000.0, 'ADA': 5000.0, 'DOT': 100.0
            }
            amount = test_balances.get(asset.upper(), 0.0)
            return Balance(asset=asset.upper(), free=amount, locked=0.0, total=amount)
        
        try:
            account_response = self._private_request('GET', '/api/v3/account')
            if 'balances' in account_response:
                for balance in account_response['balances']:
                    if balance['asset'] == asset.upper():
                        return Balance(
                            asset=asset.upper(),
                            free=float(balance['free']),
                            locked=float(balance['locked']),
                            total=float(balance['free']) + float(balance['locked'])
                        )
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання балансу {asset}: {e}")
            
        return Balance(asset=asset.upper(), free=0.0, locked=0.0, total=0.0)

    def get_all_balances(self) -> List[Balance]:
        """Отримання всіх балансів"""
        balances = []
        
        if self.config.testnet:
            test_balances = {
                'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 
                'MX': 1000.0, 'ADA': 5000.0, 'DOT': 100.0
            }
            for asset, amount in test_balances.items():
                balances.append(Balance(asset=asset, free=amount, locked=0.0, total=amount))
            return balances
        
        try:
            account_response = self._private_request('GET', '/api/v3/account')
            if 'balances' in account_response:
                for balance in account_response['balances']:
                    total = float(balance['free']) + float(balance['locked'])
                    if total > 0:
                        balances.append(Balance(
                            asset=balance['asset'],
                            free=float(balance['free']),
                            locked=float(balance['locked']),
                            total=total
                        ))
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання балансів: {e}")
        
        return balances

    def place_order(self, order_request: OrderRequest) -> Optional[str]:
        """Створення ордера"""
        if self.config.testnet:
            self.logger.info(f"🧪 Тестовий ордер: {order_request}")
            return f"test_order_{int(time.time())}"
        
        try:
            params = {
                'symbol': order_request.symbol,
                'side': order_request.side.value,
                'type': order_request.order_type.value,
            }
            
            if order_request.quantity:
                params['quantity'] = order_request.quantity
            if order_request.quote_order_qty:
                params['quoteOrderQty'] = order_request.quote_order_qty
            if order_request.price:
                params['price'] = order_request.price
            if order_request.new_client_order_id:
                params['newClientOrderId'] = order_request.new_client_order_id
            
            response = self._private_request('POST', '/api/v3/order', params)
            
            if 'orderId' in response:
                order_id = str(response['orderId'])
                self.logger.info(f"✅ Ордер створено: {order_id}")
                return order_id
                
        except Exception as e:
            self.logger.error(f"❌ Помилка створення ордера: {e}")
            
        return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Скасування ордера"""
        if self.config.testnet:
            self.logger.info(f"🧪 Тестове скасування ордера: {order_id}")
            return True
            
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            response = self._private_request('DELETE', '/api/v3/order', params)
            
            if 'orderId' in response:
                self.logger.info(f"✅ Ордер скасовано: {order_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Помилка скасування ордера: {e}")
            
        return False

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Отримання статусу ордера"""
        if self.config.testnet:
            return {
                'orderId': order_id,
                'symbol': symbol,
                'status': 'FILLED',
                'type': 'MARKET',
                'side': 'BUY',
                'executedQty': '0.1'
            }
            
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            response = self._private_request('GET', '/api/v3/order', params)
            
            return response
                
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання статусу ордера: {e}")
            
        return None

    def convert(self, from_asset: str, to_asset: str, amount: Union[str, float]) -> bool:
        """Конвертація активів через MEXC"""
        from_asset = from_asset.upper()
        to_asset = to_asset.upper()
        
        if from_asset == to_asset:
            self.logger.warning("❌ Однакові активи для конвертації")
            return False
        
        # Отримання балансу
        balance = self.get_balance(from_asset)
        if isinstance(balance, Balance):
            available_amount = balance.free
        else:
            available_amount = float(balance)
            
        if available_amount <= 0:
            self.logger.warning(f"❌ Недостатньо {from_asset} для конвертації")
            return False
        
        # Обробка суми
        is_max = str(amount).lower() == 'max'
        convert_amount = available_amount if is_max else float(amount)
        
        if convert_amount > available_amount:
            self.logger.warning(f"❌ Недостатньо коштів: потрібно {convert_amount}, доступно {available_amount}")
            return False
        
        # Знаходження торгової пари
        symbol = f"{from_asset}{to_asset}"
        reverse_symbol = f"{to_asset}{from_asset}"
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            symbol_info = self.get_symbol_info(reverse_symbol)
            if symbol_info:
                symbol = reverse_symbol
        
        if not symbol_info:
            self.logger.error(f"❌ Торгова пара {from_asset}/{to_asset} не знайдена")
            return False
        
        # Створення ордера
        order_request = OrderRequest(
            symbol=symbol,
            side=MexcOrderSide.SELL if symbol.startswith(from_asset) else MexcOrderSide.BUY,
            order_type=MexcOrderType.MARKET,
            quantity=str(convert_amount) if symbol.startswith(from_asset) else None,
            quote_order_qty=str(convert_amount) if not symbol.startswith(from_asset) else None
        )
        
        order_id = self.place_order(order_request)
        
        if order_id:
            self.logger.info(f"✅ Конвертація успішна: {convert_amount} {from_asset} → {to_asset}")
            return True
        else:
            self.logger.error("❌ Не вдалося виконати конвертацію")
            return False

    def show_balances(self) -> None:
        """Відображення всіх балансів"""
        balances = self.get_all_balances()
        
        if not balances:
            self.logger.info("💰 Немає активних балансів")
            return
        
        self.logger.info(f"💰 Баланси ({'ТЕСТ' if self.config.testnet else 'РЕАЛ'}):")
        total_usd = 0.0
        
        for balance in balances:
            # Простий курс для демонстрації
            rates = {'BTC': 45000, 'ETH': 2500, 'USDT': 1, 'MX': 3, 'ADA': 0.5, 'DOT': 8}
            rate = rates.get(balance.asset, 1.0)
            usd_value = balance.total * rate
            total_usd += usd_value
            
            print(f"  {balance.asset}: {balance.total:,.8f} (~${usd_value:,.2f})")
        
        print(f"💵 Загалом: ~${total_usd:,.2f}")

    def get_trading_symbols(self) -> List[str]:
        """Отримання списку доступних торгових пар"""
        self._update_cache()
        return list(self._symbols_cache.keys())

    def get_market_price(self, symbol: str) -> Optional[float]:
        """Отримання ринкової ціни для символу"""
        ticker = self.get_ticker(symbol)
        if ticker and 'lastPrice' in ticker:
            return float(ticker['lastPrice'])
        return None

# Функція демонстрації
def demo_mexc_client():
    """Демонстрація роботи з MEXC клієнтом"""
    print("🏛️ === MEXC API КЛІЄНТ ДЕМО ===")
    
    # Створення конфігурації
    config = MexcConfig(testnet=True)
    
    # Ініціалізація клієнта
    client = MexcClient(config)
    
    # Показ балансів
    print("\n1. Показ балансів:")
    client.show_balances()
    
    # Отримання інформації про торгові пари
    print("\n2. Доступні торгові пари:")
    symbols = client.get_trading_symbols()
    if client.config.testnet:
        print("Пари в тестовому режимі: BTCUSDT, ETHUSDT, MXUSDT та інші")
    else:
        print(f"Знайдено {len(symbols)} торгових пар")
    
    # Демонстрація отримання ціни
    print("\n3. Ринкова ціна BTCUSDT:")
    price = client.get_market_price('BTCUSDT')
    if price:
        print(f"Ціна: ${price:,.2f}")
    else:
        print("Ціна не доступна (тестовий режим)")
    
    # Демонстрація конвертації
    print("\n4. Тестова конвертація:")
    success = client.convert('BTC', 'ETH', 0.1)
    print(f"Результат конвертації: {'✅ Успішно' if success else '❌ Помилка'}")
    
    # Показ оновлених балансів
    print("\n5. Баланси після конвертації:")
    client.show_balances()


if __name__ == "__main__":
    demo_mexc_client()