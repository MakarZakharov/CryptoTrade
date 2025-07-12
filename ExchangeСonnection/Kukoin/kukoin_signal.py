import os
import time
import hashlib
import hmac
import base64
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


class KucoinOrderType(Enum):
    """Типи ордерів Kucoin"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class KucoinOrderSide(Enum):
    """Сторони ордера"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class KucoinConfig:
    """Конфігурація для Kucoin клієнта"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    testnet: bool = True
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: bool = True
    user_agent: str = "KucoinTrader/1.0"


@dataclass
class OrderRequest:
    """Запит на створення ордера"""
    symbol: str
    side: KucoinOrderSide
    order_type: KucoinOrderType
    size: Optional[str] = None
    funds: Optional[str] = None
    price: Optional[str] = None
    client_oid: Optional[str] = None


@dataclass
class Balance:
    """Інформація про баланс"""
    currency: str
    available: float
    hold: float
    balance: float


class KucoinAPIException(Exception):
    """Виняток для помилок Kucoin API"""
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


class KucoinClient:
    """Клієнт для роботи з Kucoin API"""

    # Kucoin API endpoints
    BASE_URL = "https://api.kucoin.com"
    SANDBOX_URL = "https://openapi-sandbox.kucoin.com"
    API_VERSION = "v1"

    def __init__(self, config: Optional[KucoinConfig] = None):
        """
        Ініціалізація Kucoin клієнта
        
        Args:
            config: Конфігурація клієнта
        """
        self.config = config or KucoinConfig()
        
        # Отримання API ключів з .env файлу або конфігурації
        self.api_key = self.config.api_key or os.getenv('KUCOIN_API_KEY')
        self.api_secret = self.config.api_secret or os.getenv('KUCOIN_API_SECRET')
        self.api_passphrase = self.config.api_passphrase or os.getenv('KUCOIN_API_PASSPHRASE')
        
        # Вибір URL залежно від режиму
        self.base_url = self.SANDBOX_URL if self.config.testnet else self.BASE_URL
        
        # Ініціалізація logger
        self.logger = self._setup_logger()
        
        # Кеш для торгових пар та курсів
        self._symbols_cache: Dict = {}
        self._ticker_cache: Dict = {}
        self._cache_timestamp = 0
        self.cache_ttl = 60  # Кеш на 1 хвилину
        
        # CCXT клієнт для резервного використання
        self.ccxt_client: Optional[ccxt.kucoin] = None
        
        # Перевірка підключення
        self._validate_connection()

    def _setup_logger(self) -> logging.Logger:
        """Налаштування логера"""
        logger = logging.getLogger(f'KucoinClient_{id(self)}')
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
            self.logger.info("🧪 Kucoin клієнт в тестовому режимі")
            return
        
        if not self.api_key or not self.api_secret or not self.api_passphrase:
            self.logger.warning("⚠️ API ключі не налаштовані - доступні тільки публічні методи")
            return
        
        try:
            # Ініціалізація CCXT клієнта
            self._init_ccxt_client()
            
            # Тест API підключення
            accounts_response = self._private_request('GET', '/api/v1/accounts')
            if accounts_response.get('code') != '200000':
                raise KucoinAPIException(f"API помилка: {accounts_response.get('msg')}")
            
            self.logger.info("✅ Kucoin API успішно підключено")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка підключення до Kucoin API: {e}")
            if not self.config.testnet:
                raise KucoinAPIException(f"Неможливо підключитися до Kucoin API: {e}")

    def _init_ccxt_client(self) -> None:
        """Ініціалізація CCXT клієнта"""
        if not self.api_key or not self.api_secret or not self.api_passphrase:
            return
        
        try:
            self.ccxt_client = ccxt.kucoin({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_passphrase,
                'sandbox': self.config.testnet,
                'enableRateLimit': self.config.rate_limit,
                'timeout': self.config.timeout * 1000,
                'options': {'defaultType': 'spot'}
            })
            
            if not self.config.testnet:
                self.ccxt_client.load_markets()
                self.logger.info("✅ CCXT Kucoin клієнт ініціалізовано")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Не вдалося ініціалізувати CCXT: {e}")
            self.ccxt_client = None

    def _generate_signature(self, timestamp: str, method: str, endpoint: str, body: str = '') -> str:
        """Генерація підпису для приватних запитів"""
        str_to_sign = timestamp + method + endpoint + body
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                str_to_sign.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()
        
        # KC-API-PASSPHRASE також потребує підпису
        passphrase = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                self.api_passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()
        
        return signature, passphrase

    def _public_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Виконання публічного запиту до Kucoin API"""
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
                
                if data.get('code') != '200000':
                    raise KucoinAPIException(f"API помилка: {data.get('msg')}", data.get('code'))
                
                return data
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise KucoinAPIException(f"Помилка мережі: {e}")
                
                self.logger.warning(f"Повтор запиту через {self.config.retry_delay}с...")
                time.sleep(self.config.retry_delay)
        
        raise KucoinAPIException("Максимальна кількість спроб вичерпана")

    def _private_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Виконання приватного запиту до Kucoin API"""
        if not self.api_key or not self.api_secret or not self.api_passphrase:
            raise KucoinAPIException("API ключі не налаштовані")
        
        if params is None:
            params = {}
        
        timestamp = str(int(time.time() * 1000))
        body = json.dumps(params) if method.upper() != 'GET' else ''
        
        # Генерація підпису
        signature, passphrase = self._generate_signature(timestamp, method.upper(), endpoint, body)
        
        headers = {
            'KC-API-SIGN': signature,
            'KC-API-TIMESTAMP': timestamp,
            'KC-API-KEY': self.api_key,
            'KC-API-PASSPHRASE': passphrase,
            'KC-API-KEY-VERSION': '2',
            'Content-Type': 'application/json',
            'User-Agent': self.config.user_agent
        }
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, params=params, headers=headers, timeout=self.config.timeout)
                elif method.upper() == 'POST':
                    response = requests.post(url, data=body, headers=headers, timeout=self.config.timeout)
                elif method.upper() == 'DELETE':
                    response = requests.delete(url, headers=headers, timeout=self.config.timeout)
                else:
                    raise KucoinAPIException(f"Непідтримуваний HTTP метод: {method}")
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('code') != '200000':
                    raise KucoinAPIException(f"API помилка: {data.get('msg')}", data.get('code'))
                
                return data
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise KucoinAPIException(f"Помилка мережі: {e}")
                
                self.logger.warning(f"Повтор запиту через {self.config.retry_delay}с...")
                time.sleep(self.config.retry_delay)
        
        raise KucoinAPIException("Максимальна кількість спроб вичерпана")

    def _update_cache(self) -> None:
        """Оновлення кешу торгових пар та курсів"""
        current_time = time.time()
        if current_time - self._cache_timestamp < self.cache_ttl:
            return
        
        try:
            # Оновлення символів торгових пар
            symbols_data = self._public_request('GET', '/api/v1/symbols')
            if 'data' in symbols_data:
                self._symbols_cache = {symbol['symbol']: symbol for symbol in symbols_data['data']}
            
            # Оновлення тікерів
            ticker_data = self._public_request('GET', '/api/v1/market/allTickers')
            if 'data' in ticker_data and 'ticker' in ticker_data['data']:
                self._ticker_cache = {ticker['symbol']: ticker for ticker in ticker_data['data']['ticker']}
            
            self._cache_timestamp = current_time
            self.logger.info("✅ Кеш Kucoin даних оновлено")
            
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

    def get_balance(self, currency: str) -> Union[Balance, float]:
        """Отримання балансу валюти"""
        if self.config.testnet:
            # Тестовий баланс для демонстрації
            test_balances = {
                'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 
                'KCS': 100.0, 'ADA': 5000.0, 'DOT': 100.0
            }
            amount = test_balances.get(currency.upper(), 0.0)
            return Balance(currency=currency.upper(), available=amount, hold=0.0, balance=amount)
        
        try:
            accounts_response = self._private_request('GET', '/api/v1/accounts')
            if 'data' in accounts_response:
                for account in accounts_response['data']:
                    if account['currency'] == currency.upper() and account['type'] == 'trade':
                        return Balance(
                            currency=currency.upper(),
                            available=float(account['available']),
                            hold=float(account['holds']),
                            balance=float(account['balance'])
                        )
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання балансу {currency}: {e}")
            
        return Balance(currency=currency.upper(), available=0.0, hold=0.0, balance=0.0)

    def get_all_balances(self) -> List[Balance]:
        """Отримання всіх балансів"""
        balances = []
        
        if self.config.testnet:
            test_balances = {
                'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 
                'KCS': 100.0, 'ADA': 5000.0, 'DOT': 100.0
            }
            for currency, amount in test_balances.items():
                balances.append(Balance(currency=currency, available=amount, hold=0.0, balance=amount))
            return balances
        
        try:
            accounts_response = self._private_request('GET', '/api/v1/accounts')
            if 'data' in accounts_response:
                for account in accounts_response['data']:
                    if account['type'] == 'trade' and float(account['balance']) > 0:
                        balances.append(Balance(
                            currency=account['currency'],
                            available=float(account['available']),
                            hold=float(account['holds']),
                            balance=float(account['balance'])
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
                'clientOid': order_request.client_oid or str(int(time.time() * 1000)),
                'side': order_request.side.value,
                'symbol': order_request.symbol,
                'type': order_request.order_type.value
            }
            
            if order_request.size:
                params['size'] = order_request.size
            if order_request.funds:
                params['funds'] = order_request.funds
            if order_request.price:
                params['price'] = order_request.price
            
            response = self._private_request('POST', '/api/v1/orders', params)
            
            if 'data' in response and 'orderId' in response['data']:
                order_id = response['data']['orderId']
                self.logger.info(f"✅ Ордер створено: {order_id}")
                return order_id
                
        except Exception as e:
            self.logger.error(f"❌ Помилка створення ордера: {e}")
            
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Скасування ордера"""
        if self.config.testnet:
            self.logger.info(f"🧪 Тестове скасування ордера: {order_id}")
            return True
            
        try:
            response = self._private_request('DELETE', f'/api/v1/orders/{order_id}')
            
            if 'data' in response:
                self.logger.info(f"✅ Ордер скасовано: {order_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Помилка скасування ордера: {e}")
            
        return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Отримання статусу ордера"""
        if self.config.testnet:
            return {
                'id': order_id,
                'symbol': 'BTC-USDT',
                'side': 'buy',
                'type': 'market',
                'dealSize': '0.1',
                'isActive': False
            }
            
        try:
            response = self._private_request('GET', f'/api/v1/orders/{order_id}')
            
            if 'data' in response:
                return response['data']
                
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання статусу ордера: {e}")
            
        return None

    def convert(self, from_currency: str, to_currency: str, amount: Union[str, float]) -> bool:
        """Конвертація валют через Kucoin"""
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        if from_currency == to_currency:
            self.logger.warning("❌ Однакові валюти для конвертації")
            return False
        
        # Отримання балансу
        balance = self.get_balance(from_currency)
        if isinstance(balance, Balance):
            available_amount = balance.available
        else:
            available_amount = float(balance)
            
        if available_amount <= 0:
            self.logger.warning(f"❌ Недостатньо {from_currency} для конвертації")
            return False
        
        # Обробка суми
        is_max = str(amount).lower() == 'max'
        convert_amount = available_amount if is_max else float(amount)
        
        if convert_amount > available_amount:
            self.logger.warning(f"❌ Недостатньо коштів: потрібно {convert_amount}, доступно {available_amount}")
            return False
        
        # Знаходження торгової пари
        symbol = f"{from_currency}-{to_currency}"
        reverse_symbol = f"{to_currency}-{from_currency}"
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            symbol_info = self.get_symbol_info(reverse_symbol)
            if symbol_info:
                symbol = reverse_symbol
        
        if not symbol_info:
            self.logger.error(f"❌ Торгова пара {from_currency}/{to_currency} не знайдена")
            return False
        
        # Створення ордера
        order_request = OrderRequest(
            symbol=symbol,
            side=KucoinOrderSide.SELL if symbol.startswith(from_currency) else KucoinOrderSide.BUY,
            order_type=KucoinOrderType.MARKET,
            size=str(convert_amount) if symbol.startswith(from_currency) else None,
            funds=str(convert_amount) if not symbol.startswith(from_currency) else None
        )
        
        order_id = self.place_order(order_request)
        
        if order_id:
            self.logger.info(f"✅ Конвертація успішна: {convert_amount} {from_currency} → {to_currency}")
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
            rates = {'BTC': 45000, 'ETH': 2500, 'USDT': 1, 'KCS': 8, 'ADA': 0.5, 'DOT': 8}
            rate = rates.get(balance.currency, 1.0)
            usd_value = balance.balance * rate
            total_usd += usd_value
            
            print(f"  {balance.currency}: {balance.balance:,.8f} (~${usd_value:,.2f})")
        
        print(f"💵 Загалом: ~${total_usd:,.2f}")

    def get_trading_symbols(self) -> List[str]:
        """Отримання списку доступних торгових пар"""
        self._update_cache()
        return list(self._symbols_cache.keys())

    def get_market_price(self, symbol: str) -> Optional[float]:
        """Отримання ринкової ціни для символу"""
        ticker = self.get_ticker(symbol)
        if ticker and 'last' in ticker:
            return float(ticker['last'])
        return None

# Функція демонстрації
def demo_kucoin_client():
    """Демонстрація роботи з Kucoin клієнтом"""
    print("🌐 === KUCOIN API КЛІЄНТ ДЕМО ===")
    
    # Створення конфігурації
    config = KucoinConfig(testnet=True)
    
    # Ініціалізація клієнта
    client = KucoinClient(config)
    
    # Показ балансів
    print("\n1. Показ балансів:")
    client.show_balances()
    
    # Отримання інформації про торгові пари
    print("\n2. Доступні торгові пари:")
    symbols = client.get_trading_symbols()
    if client.config.testnet:
        print("Пари в тестовому режимі: BTC-USDT, ETH-USDT, KCS-USDT та інші")
    else:
        print(f"Знайдено {len(symbols)} торгових пар")
    
    # Демонстрація отримання ціни
    print("\n3. Ринкова ціна BTC-USDT:")
    price = client.get_market_price('BTC-USDT')
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
    demo_kucoin_client()