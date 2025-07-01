"""
Binance Data Adapter

Adapter for fetching data from Binance exchange.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import time

from .base_adapter import BaseDataAdapter


class BinanceAdapter(BaseDataAdapter):
    """
    Binance exchange data adapter
    
    Provides access to Binance spot and futures market data
    through the CCXT library.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        rate_limit: int = 1200,  # Binance allows 1200 requests per minute
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Binance adapter
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet
            rate_limit: Rate limit for API calls
            config: Additional configuration
        """
        super().__init__(api_key, api_secret, testnet, rate_limit, config)
        
        # Binance-specific settings
        self.exchange_name = 'binance'
        self.timeframe_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        
    def _initialize_client(self) -> None:
        """Initialize Binance client"""
        try:
            self.client = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.testnet,
                'rateLimit': 60000 / self.rate_limit,  # Convert to milliseconds
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # spot, future, delivery
                }
            })
            
            # Test connection
            if self.api_key and self.api_secret:
                self.client.load_markets()
                self.logger.info("Binance client initialized successfully")
            else:
                self.logger.info("Binance client initialized in public mode")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            self.client = None
    
    def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            markets = self.client.load_markets()
            symbols = [symbol for symbol, market in markets.items() 
                      if market['active'] and market['type'] == 'spot']
            
            self.logger.info(f"Retrieved {len(symbols)} symbols from Binance")
            return sorted(symbols)
            
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {e}")
            return []
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            since: Start datetime
            limit: Number of candles to fetch (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        # Validate and standardize timeframe
        timeframe = self.standardize_timeframe(timeframe)
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        try:
            # Convert datetime to timestamp
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)  # Binance uses milliseconds
            
            # Limit default and maximum
            if limit is None:
                limit = 1000
            limit = min(limit, 1000)  # Binance limit
            
            # Fetch data
            ohlcv = self.client.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe_map[timeframe],
                since=since_ts,
                limit=limit
            )
            
            if not ohlcv:
                self.logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Standardize format
            df = self.standardize_ohlcv(df)
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book data
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to fetch (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Order book data
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            # Binance supported limits
            supported_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
            limit = min(supported_limits, key=lambda x: abs(x - limit))
            
            orderbook = self.client.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now(),
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'bid_price': orderbook['bids'][0][0] if orderbook['bids'] else None,
                'ask_price': orderbook['asks'][0][0] if orderbook['asks'] else None,
                'spread': (orderbook['asks'][0][0] - orderbook['bids'][0][0]) 
                         if orderbook['bids'] and orderbook['asks'] else None
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {}
    
    def get_trades(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get recent trades
        
        Args:
            symbol: Trading pair symbol
            since: Start datetime
            limit: Number of trades to fetch (max 1000)
            
        Returns:
            DataFrame with trade data
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            # Convert datetime to timestamp
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)
            
            if limit is None:
                limit = 1000
            limit = min(limit, 1000)
            
            trades = self.client.fetch_trades(symbol, since_ts, limit)
            
            if not trades:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Standardize columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Select relevant columns
            columns = ['price', 'amount', 'side', 'cost']
            df = df[columns]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching trades for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24h ticker data
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            ticker = self.client.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now(),
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'high': ticker['high'],
                'low': ticker['low'],
                'open': ticker['open'],
                'close': ticker['close'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'vwap': ticker['vwap']
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        """Get 24h statistics for symbol"""
        return self.get_ticker(symbol)
    
    def get_kline_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get kline/candlestick data (alternative to get_ohlcv)
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_time: Start datetime
            end_time: End datetime
            limit: Number of klines
            
        Returns:
            DataFrame with kline data
        """
        # Convert interval to timeframe
        interval_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        
        timeframe = interval_map.get(interval, '1h')
        
        return self.get_ohlcv(symbol, timeframe, start_time, limit)
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed symbol information
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Symbol information
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            markets = self.client.load_markets()
            
            if symbol not in markets:
                return {}
            
            market = markets[symbol]
            
            return {
                'symbol': symbol,
                'base_asset': market['base'],
                'quote_asset': market['quote'],
                'status': 'TRADING' if market['active'] else 'INACTIVE',
                'base_precision': market['precision']['base'],
                'quote_precision': market['precision']['quote'],
                'min_qty': market['limits']['amount']['min'],
                'max_qty': market['limits']['amount']['max'],
                'min_price': market['limits']['price']['min'],
                'max_price': market['limits']['price']['max'],
                'min_notional': market['limits']['cost']['min'],
                'tick_size': market['precision']['price'],
                'step_size': market['precision']['amount']
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return {}
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            # Get exchange status and basic info
            status = self.client.fetch_status()
            
            return {
                'exchange': 'binance',
                'status': status.get('status', 'unknown'),
                'updated': status.get('updated'),
                'rate_limits': {
                    'requests_per_minute': self.rate_limit,
                    'orders_per_second': 10,
                    'orders_per_day': 200000
                },
                'trading_fees': {
                    'maker': 0.001,  # 0.1%
                    'taker': 0.001   # 0.1%
                },
                'testnet': self.testnet
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching exchange info: {e}")
            return {'exchange': 'binance', 'status': 'error'}
    
    def validate_connection(self) -> bool:
        """Validate connection to Binance"""
        try:
            if not self.client:
                return False
            
            # Try to fetch server time
            server_time = self.client.fetch_time()
            return server_time is not None
            
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False