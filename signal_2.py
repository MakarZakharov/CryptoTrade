import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import logging
from decimal import Decimal
from typing import Dict, List, Optional
import time

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceTrader:
    """
    Клас для торгівлі на біржі Binance через API
    Підтримує як реальну торгівлю, так і тестовий режим
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        """
        Ініціалізація клієнта Binance
        
        Args:
            api_key: API ключ Binance
            api_secret: Секретний ключ Binance
            testnet: Використовувати тестову мережу (True для тестування)
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.testnet = testnet
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API ключі не знайдені. Встановіть BINANCE_API_KEY та BINANCE_API_SECRET")
        
        try:
            if testnet:
                # Тестова мережа Binance
                self.client = Client(
                    self.api_key, 
                    self.api_secret,
                    testnet=True
                )
                logger.info("Підключено до тестової мережі Binance")
            else:
                # Реальна мережа Binance
                self.client = Client(self.api_key, self.api_secret)
                logger.info("Підключено до реальної мережі Binance")
                
            # Перевірка підключення
            self.client.ping()
            logger.info("Підключення до Binance успішне!")
            
        except Exception as e:
            logger.error(f"Помилка підключення до Binance: {e}")
            raise
    
    def get_account_balance(self) -> Dict:
        """
        Отримати баланс акаунта
        
        Returns:
            Словник з інформацією про баланс
        """
        try:
            account_info = self.client.get_account()
            balances = {}
            
            for balance in account_info['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:  # Показувати тільки токени з балансом
                    balances[balance['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
            
            return balances
            
        except BinanceAPIException as e:
            logger.error(f"Помилка отримання балансу: {e}")
            return {}
    
    def get_symbol_price(self, symbol: str) -> float:
        """
        Отримати поточну ціну символу
        
        Args:
            symbol: Торгова пара (наприклад, 'BTCUSDT')
            
        Returns:
            Поточна ціна
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol.upper())
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Помилка отримання ціни для {symbol}: {e}")
            return 0.0
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """
        Отримати книгу ордерів
        
        Args:
            symbol: Торгова пара
            limit: Кількість рівнів (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Книга ордерів з bid/ask
        """
        try:
            depth = self.client.get_order_book(symbol=symbol.upper(), limit=limit)
            return {
                'bids': [[float(price), float(qty)] for price, qty in depth['bids']],
                'asks': [[float(price), float(qty)] for price, qty in depth['asks']]
            }
        except BinanceAPIException as e:
            logger.error(f"Помилка отримання книги ордерів для {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """
        Розмістити ринковий ордер
        
        Args:
            symbol: Торгова пара (наприклад, 'BTCUSDT')
            side: 'BUY' або 'SELL'
            quantity: Кількість для торгівлі
            
        Returns:
            Інформація про ордер
        """
        try:
            if self.testnet:
                logger.info(f"ТЕСТ: {side} {quantity} {symbol} за ринковою ціною")
                return {
                    'symbol': symbol,
                    'orderId': f'TEST_{int(time.time())}',
                    'side': side,
                    'type': 'MARKET',
                    'quantity': str(quantity),
                    'status': 'FILLED',
                    'test': True
                }
            
            order = self.client.order_market(
                symbol=symbol.upper(),
                side=side.upper(),
                quantity=quantity
            )
            
            logger.info(f"Ордер розміщено: {order['orderId']}")
            return order
            
        except BinanceOrderException as e:
            logger.error(f"Помилка розміщення ордера: {e}")
            return {}
        except BinanceAPIException as e:
            logger.error(f"API помилка: {e}")
            return {}
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """
        Розмістити лімітний ордер
        
        Args:
            symbol: Торгова пара
            side: 'BUY' або 'SELL'
            quantity: Кількість
            price: Ціна
            
        Returns:
            Інформація про ордер
        """
        try:
            if self.testnet:
                logger.info(f"ТЕСТ: {side} {quantity} {symbol} за ціною {price}")
                return {
                    'symbol': symbol,
                    'orderId': f'TEST_{int(time.time())}',
                    'side': side,
                    'type': 'LIMIT',
                    'quantity': str(quantity),
                    'price': str(price),
                    'status': 'NEW',
                    'test': True
                }
            
            order = self.client.order_limit(
                symbol=symbol.upper(),
                side=side.upper(),
                quantity=quantity,
                price=price
            )
            
            logger.info(f"Лімітний ордер розміщено: {order['orderId']}")
            return order
            
        except BinanceOrderException as e:
            logger.error(f"Помилка розміщення лімітного ордера: {e}")
            return {}
        except BinanceAPIException as e:
            logger.error(f"API помилка: {e}")
            return {}
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Скасувати ордер
        
        Args:
            symbol: Торгова пара
            order_id: ID ордера
            
        Returns:
            True якщо успішно скасовано
        """
        try:
            if self.testnet:
                logger.info(f"ТЕСТ: Скасування ордера {order_id}")
                return True
            
            result = self.client.cancel_order(symbol=symbol.upper(), orderId=order_id)
            logger.info(f"Ордер {order_id} скасовано")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"Помилка скасування ордера: {e}")
            return False
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Отримати відкриті ордери
        
        Args:
            symbol: Торгова пара (опціонально)
            
        Returns:
            Список відкритих ордерів
        """
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol.upper())
            else:
                orders = self.client.get_open_orders()
            
            return orders
            
        except BinanceAPIException as e:
            logger.error(f"Помилка отримання відкритих ордерів: {e}")
            return []
    
    def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Отримати історію торгів
        
        Args:
            symbol: Торгова пара
            limit: Кількість записів
            
        Returns:
            Історія торгів
        """
        try:
            trades = self.client.get_my_trades(symbol=symbol.upper(), limit=limit)
            return trades
            
        except BinanceAPIException as e:
            logger.error(f"Помилка отримання історії торгів: {e}")
            return []
    
    def get_24hr_ticker(self, symbol: str) -> Dict:
        """
        Отримати 24-годинну статистику
        
        Args:
            symbol: Торгова пара
            
        Returns:
            24-годинна статистика
        """
        try:
            ticker = self.client.get_24hr_ticker(symbol=symbol.upper())
            return {
                'symbol': ticker['symbol'],
                'price_change': float(ticker['priceChange']),
                'price_change_percent': float(ticker['priceChangePercent']),
                'prev_close_price': float(ticker['prevClosePrice']),
                'last_price': float(ticker['lastPrice']),
                'bid_price': float(ticker['bidPrice']),
                'ask_price': float(ticker['askPrice']),
                'volume': float(ticker['volume']),
                'high_price': float(ticker['highPrice']),
                'low_price': float(ticker['lowPrice'])
            }
        except BinanceAPIException as e:
            logger.error(f"Помилка отримання 24hr статистики: {e}")
            return {}

def demo_trading():
    """
    Демонстрація роботи з Binance API
    """
    print("=== ДЕМО ТОРГІВЛІ НА BINANCE ===")
    print("УВАГА: Це тестовий режим!")
    
    # Ініціалізація (тестовий режим)
    try:
        trader = BinanceTrader(testnet=True)
    except Exception as e:
        print(f"Помилка ініціалізації: {e}")
        print("Переконайтеся, що у вас є API ключі в змінних середовища:")
        print("BINANCE_API_KEY та BINANCE_API_SECRET")
        return
    
    # Отримання балансу
    print("\n--- БАЛАНС АКАУНТА ---")
    balance = trader.get_account_balance()
    if balance:
        for asset, info in balance.items():
            print(f"{asset}: {info['total']:.8f} (вільно: {info['free']:.8f})")
    else:
        print("Баланс порожній або помилка отримання")
    
    # Поточна ціна BTC/USDT
    print("\n--- ПОТОЧНА ЦІНА ---")
    btc_price = trader.get_symbol_price('BTCUSDT')
    print(f"BTC/USDT: ${btc_price:,.2f}")
    
    # 24-годинна статистика
    print("\n--- 24-ГОДИННА СТАТИСТИКА ---")
    ticker = trader.get_24hr_ticker('BTCUSDT')
    if ticker:
        print(f"Зміна ціни: {ticker['price_change_percent']:.2f}%")
        print(f"Максимум: ${ticker['high_price']:,.2f}")
        print(f"Мінімум: ${ticker['low_price']:,.2f}")
        print(f"Обсяг: {ticker['volume']:,.2f} BTC")
    
    # Книга ордерів
    print("\n--- КНИГА ОРДЕРІВ (топ 5) ---")
    order_book = trader.get_order_book('BTCUSDT', 5)
    if order_book['asks']:
        print("Продаж (Ask):")
        for price, qty in order_book['asks'][:3]:
            print(f"  ${price:,.2f} - {qty:.6f} BTC")
    if order_book['bids']:
        print("Купівля (Bid):")
        for price, qty in order_book['bids'][:3]:
            print(f"  ${price:,.2f} - {qty:.6f} BTC")
    
    # Тестовий ордер
    print("\n--- ТЕСТОВИЙ ОРДЕР ---")
    test_order = trader.place_market_order('BTCUSDT', 'BUY', 0.001)
    if test_order:
        print(f"Тестовий ордер створено: {test_order['orderId']}")
    
    print("\n=== ДЕМО ЗАВЕРШЕНО ===")

if __name__ == "__main__":
    demo_trading()