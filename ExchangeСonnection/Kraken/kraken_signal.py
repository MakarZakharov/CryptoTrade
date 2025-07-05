import os
import time
import hashlib
import hmac
import base64
import urllib.parse
import requests
from typing import Dict, Optional, List, Tuple, Union
import json
import logging
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
import ccxt

load_dotenv()


class KrakenOrderType(Enum):
    """Типи ордерів Kraken"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    STOP_LOSS_LIMIT = "stop-loss-limit"
    TAKE_PROFIT = "take-profit"
    TAKE_PROFIT_LIMIT = "take-profit-limit"


class KrakenOrderSide(Enum):
    """Сторони ордера"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class KrakenConfig:
    """Конфігурація для Kraken клієнта"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: bool = True
    user_agent: str = "KrakenTrader/2.0"


@dataclass
class OrderRequest:
    """Запит на створення ордера"""
    pair: str
    side: KrakenOrderSide
    order_type: KrakenOrderType
    volume: float
    price: Optional[float] = None
    leverage: Optional[int] = None
    validate: bool = False


@dataclass
class Balance:
    """Інформація про баланс"""
    asset: str
    free: float
    locked: float
    total: float


class KrakenAPIException(Exception):
    """Виняток для помилок Kraken API"""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class KrakenClient:
    """Клієнт для роботи з Kraken API"""

    # Kraken API endpoints
    BASE_URL = "https://api.kraken.com"
    API_VERSION = "0"

    # Маппінг активів до Kraken формату
    ASSET_MAPPING = {
        'BTC': 'XXBT',
        'ETH': 'XETH', 
        'USD': 'ZUSD',
        'EUR': 'ZEUR',
        'USDT': 'USDT',
        'USDC': 'USDC',
        'ADA': 'ADA',
        'DOT': 'DOT',
        'SOL': 'SOL',
        'XRP': 'XXRP',
        'LTC': 'XLTC',
        'BCH': 'BCH',
        'LINK': 'LINK',
        'UNI': 'UNI',
        'ATOM': 'ATOM'
    }

    def __init__(self, config: Optional[KrakenConfig] = None):
        """
        Ініціалізація Kraken клієнта
        
        Args:
            config: Конфігурація клієнта
        """
        self.config = config or KrakenConfig()
        
        # Отримання API ключів з .env файлу або конфігурації
        self.api_key = self.config.api_key or os.getenv('KRAKEN_API_KEY')
        self.api_secret = self.config.api_secret or os.getenv('KRAKEN_API_SECRET')
        
        # Ініціалізація logger
        self.logger = self._setup_logger()
        
        # Кеш для торгових пар та активів
        self._asset_pairs: Dict = {}
        self._tradeable_assets: Dict = {}
        self._ticker_cache: Dict = {}
        self._cache_timestamp = 0
        self.cache_ttl = 60  # Кеш на 1 хвилину
        
        # CCXT клієнт для резервного використання
        self.ccxt_client: Optional[ccxt.kraken] = None
        
        # Перевірка підключення
        self._validate_connection()

    def _setup_logger(self) -> logging.Logger:
        """Налаштування логера"""
        logger = logging.getLogger(f'KrakenClient_{id(self)}')
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
            self.logger.info("🧪 Kraken клієнт в тестовому режимі")
            return
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("⚠️ API ключі не налаштовані - доступні тільки публічні методи")
            return
        
        try:
            # Ініціалізація CCXT клієнта
            self._init_ccxt_client()
            
            # Тест API підключення
            balance_response = self._private_request('Balance')
            if balance_response.get('error'):
                raise KrakenAPIException(f"API помилка: {balance_response['error']}")
            
            self.logger.info("✅ Kraken API успішно підключено")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка підключення до Kraken API: {e}")
            if not self.config.testnet:
                raise KrakenAPIException(f"Неможливо підключитися до Kraken API: {e}")

    def _init_ccxt_client(self) -> None:
        """Ініціалізація CCXT клієнта"""
        if not self.api_key or not self.api_secret:
            return
        
        try:
            self.ccxt_client = ccxt.kraken({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.config.testnet,
                'enableRateLimit': self.config.rate_limit,
                'timeout': self.config.timeout * 1000,
                'options': {'defaultType': 'spot'}
            })
            
            if not self.config.testnet:
                self.ccxt_client.load_markets()
                self.logger.info("✅ CCXT Kraken клієнт ініціалізовано")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Не вдалося ініціалізувати CCXT: {e}")
            self.ccxt_client = None

    def _generate_signature(self, url_path: str, postdata: str, nonce: str) -> str:
        """Генерація підпису для приватних запитів"""
        encoded = (nonce + postdata).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(
            base64.b64decode(self.api_secret), 
            message, 
            hashlib.sha512
        )
        return base64.b64encode(signature.digest()).decode()

    def _public_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Виконання публічного запиту до Kraken API"""
        if params is None:
            params = {}
        
        url = f"{self.BASE_URL}/{self.API_VERSION}/public/{method}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('error'):
                    raise KrakenAPIException(f"API помилка: {data['error']}")
                
                return data
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise KrakenAPIException(f"Помилка мережі: {e}")
                
                self.logger.warning(f"Повтор запиту через {self.config.retry_delay}с...")
                time.sleep(self.config.retry_delay)
        
        raise KrakenAPIException("Максимальна кількість спроб вичерпана")

    def _private_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Виконання приватного запиту до Kraken API"""
        if not self.api_key or not self.api_secret:
            raise KrakenAPIException("API ключі не налаштовані")
        
        if params is None:
            params = {}
        
        # Додавання nonce
        nonce = str(int(time.time() * 1000))
        params['nonce'] = nonce
        
        url_path = f"/{self.API_VERSION}/private/{method}"
        url = self.BASE_URL + url_path
        postdata = urllib.parse.urlencode(params)
        
        # Генерація підпису
        signature = self._generate_signature(url_path, postdata, nonce)
        
        headers = {
            'API-Key': self.api_key,
            'API-Sign': signature,
            'User-Agent': self.config.user_agent
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    data=postdata, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('error'):
                    raise KrakenAPIException(f"API помилка: {data['error']}")
                
                return data
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise KrakenAPIException(f"Помилка мережі: {e}")
                
                self.logger.warning(f"Повтор запиту через {self.config.retry_delay}с...")
                time.sleep(self.config.retry_delay)
        
        raise KrakenAPIException("Максимальна кількість спроб вичерпана")

    def _update_cache(self) -> None:
        """Оновлення кешу торгових пар та курсів"""
        current_time = time.time()
        if current_time - self._cache_timestamp < self.cache_ttl:
            return
        
        try:
            # Оновлення торгових пар
            pairs_data = self._public_request('AssetPairs')
            if 'result' in pairs_data:
                self._asset_pairs = pairs_data['result']
            
            # Оновлення активів
            assets_data = self._public_request('Assets')
            if 'result' in assets_data:
                self._tradeable_assets = assets_data['result']
            
            # Оновлення тікерів
            ticker_data = self._public_request('Ticker')
            if 'result' in ticker_data:
                self._ticker_cache = ticker_data['result']
            
            self._cache_timestamp = current_time
            self.logger.info("✅ Кеш Kraken даних оновлено")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка оновлення кешу: {e}")

    def normalize_asset(self, asset: str) -> str:
        """Нормалізація назви активу до Kraken формату"""
        return self.ASSET_MAPPING.get(asset.upper(), asset.upper())

    def denormalize_asset(self, kraken_asset: str) -> str:
        """Денормалізація назви активу з Kraken формату"""
        reverse_mapping = {v: k for k, v in self.ASSET_MAPPING.items()}
        return reverse_mapping.get(kraken_asset, kraken_asset)

    def get_trading_pair(self, base_asset: str, quote_asset: str) -> Optional[str]:
        """Знаходження торгової пари на Kraken"""
        if self.config.testnet:
            # Мок торгових пар для тестового режиму
            test_pairs = {
                ('BTC', 'USDT'): 'XBTUSDT',
                ('BTC', 'ETH'): 'XBTETH', 
                ('ETH', 'USDT'): 'ETHUSDT',
                ('ADA', 'USDT'): 'ADAUSDT',
                ('DOT', 'USDT'): 'DOTUSDT',
                ('SOL', 'USDT'): 'SOLUSDT',
                ('XRP', 'USDT'): 'XRPUSDT',
                ('ETH', 'BTC'): 'ETHXBT',
                ('USDT', 'BTC'): 'USDTXBT'
            }
            
            # Перевірка прямої та зворотної пари
            pair_key = (base_asset.upper(), quote_asset.upper())
            reverse_pair_key = (quote_asset.upper(), base_asset.upper())
            
            if pair_key in test_pairs:
                return test_pairs[pair_key]
            elif reverse_pair_key in test_pairs:
                return test_pairs[reverse_pair_key]
            
            return None
        
        self._update_cache()
        
        base_norm = self.normalize_asset(base_asset)
        quote_norm = self.normalize_asset(quote_asset)
        
        # Можливі варіанти назв пар
        possible_pairs = [
            f"{base_norm}{quote_norm}",
            f"X{base_norm}Z{quote_norm}",
            f"{base_norm}Z{quote_norm}",
            f"X{base_norm}{quote_norm}",
            f"{base_asset.upper()}{quote_asset.upper()}"
        ]
        
        for pair in possible_pairs:
            if pair in self._asset_pairs:
                return pair
        
        return None

    def get_balance(self, asset: str) -> Union[Balance, float]:
        """Отримання балансу активу"""
        if self.config.testnet:
            # Тестовий баланс для демонстрації
            test_balances = {
                'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 
                'ADA': 5000.0, 'DOT': 100.0, 'SOL': 25.0
            }
            amount = test_balances.get(asset.upper(), 0.0)
            return Balance(asset=asset.upper(), free=amount, locked=0.0, total=amount)
        
        try:
            balance_data = self._private_request('Balance')
            if 'result' in balance_data:
                kraken_asset = self.normalize_asset(asset)
                balance_amount = float(balance_data['result'].get(kraken_asset, 0.0))
                return Balance(
                    asset=asset.upper(), 
                    free=balance_amount, 
                    locked=0.0, 
                    total=balance_amount
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
                'ADA': 5000.0, 'DOT': 100.0, 'SOL': 25.0
            }
            for asset, amount in test_balances.items():
                balances.append(Balance(asset=asset, free=amount, locked=0.0, total=amount))
            return balances
        
        try:
            balance_data = self._private_request('Balance')
            if 'result' in balance_data:
                for kraken_asset, amount_str in balance_data['result'].items():
                    amount = float(amount_str)
                    if amount > 0:
                        normalized_asset = self.denormalize_asset(kraken_asset)
                        balances.append(Balance(
                            asset=normalized_asset,
                            free=amount,
                            locked=0.0,
                            total=amount
                        ))
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання балансів: {e}")
        
        return balances

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Отримання інформації про тікер"""
        self._update_cache()
        
        pair = self.get_trading_pair(*symbol.split('/')) if '/' in symbol else symbol
        if not pair:
            return None
            
        return self._ticker_cache.get(pair)

    def place_order(self, order_request: OrderRequest) -> Optional[str]:
        """Створення ордера"""
        if self.config.testnet:
            self.logger.info(f"🧪 Тестовий ордер: {order_request}")
            return f"test_order_{int(time.time())}"
        
        try:
            params = {
                'pair': order_request.pair,
                'type': order_request.side.value,
                'ordertype': order_request.order_type.value,
                'volume': str(order_request.volume)
            }
            
            if order_request.price:
                params['price'] = str(order_request.price)
                
            if order_request.leverage:
                params['leverage'] = str(order_request.leverage)
                
            if order_request.validate:
                params['validate'] = 'true'
            
            response = self._private_request('AddOrder', params)
            
            if 'result' in response and 'txid' in response['result']:
                order_id = response['result']['txid'][0]
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
            params = {'txid': order_id}
            response = self._private_request('CancelOrder', params)
            
            if 'result' in response:
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
                'status': 'closed',
                'type': 'market',
                'side': 'buy',
                'filled': '1.0'
            }
            
        try:
            params = {'txid': order_id}
            response = self._private_request('QueryOrders', params)
            
            if 'result' in response and order_id in response['result']:
                return response['result'][order_id]
                
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання статусу ордера: {e}")
            
        return None

    def convert(self, from_asset: str, to_asset: str, amount: Union[str, float]) -> bool:
        """Конвертація активів через Kraken"""
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
        trading_pair = self.get_trading_pair(from_asset, to_asset)
        if not trading_pair:
            self.logger.error(f"❌ Торгова пара {from_asset}/{to_asset} не знайдена")
            return False
        
        # Створення ордера
        order_request = OrderRequest(
            pair=trading_pair,
            side=KrakenOrderSide.SELL,  # Продаємо from_asset
            order_type=KrakenOrderType.MARKET,
            volume=convert_amount,
            validate=False
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
            rates = {'BTC': 45000, 'ETH': 2500, 'USDT': 1, 'ADA': 0.5, 'DOT': 8, 'SOL': 100}
            rate = rates.get(balance.asset, 1.0)
            usd_value = balance.total * rate
            total_usd += usd_value
            
            print(f"  {balance.asset}: {balance.total:,.8f} (~${usd_value:,.2f})")
        
        print(f"💵 Загалом: ~${total_usd:,.2f}")


# Функція демонстрації
def demo_kraken_client():
    """Демонстрація роботи з Kraken клієнтом"""
    print("🐙 === KRAKEN API КЛІЄНТ ДЕМО ===")
    
    # Створення конфігурації
    config = KrakenConfig(testnet=True)
    
    # Ініціалізація клієнта
    client = KrakenClient(config)
    
    # Показ балансів
    print("\n1. Показ балансів:")
    client.show_balances()
    
    # Отримання інформації про торгову пару
    print("\n2. Пошук торгової пари:")
    pair = client.get_trading_pair('BTC', 'USDT')
    print(f"BTC/USDT пара: {pair}")
    
    # Демонстрація конвертації
    print("\n3. Тестова конвертація:")
    success = client.convert('BTC', 'ETH', 0.1)
    print(f"Результат конвертації: {'✅ Успішно' if success else '❌ Помилка'}")
    
    # Показ оновлених балансів
    print("\n4. Баланси після конвертації:")
    client.show_balances()


if __name__ == "__main__":
    demo_kraken_client()