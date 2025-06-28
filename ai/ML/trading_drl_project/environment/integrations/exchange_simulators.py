"""
Exchange Simulators

Realistic exchange simulation for different trading platforms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import time


class ExchangeType(Enum):
    """Supported exchange types"""
    BINANCE = "binance"
    KUCOIN = "kucoin"
    KRAKEN = "kraken"
    MEXC = "mexc"
    GENERIC = "generic"


@dataclass
class OrderResult:
    """Result of order execution"""
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    executed_amount: float
    executed_price: float
    commission: float
    timestamp: pd.Timestamp
    status: str  # 'filled', 'partial', 'rejected'


class BaseExchangeSimulator(ABC):
    """Base class for exchange simulators"""
    
    def __init__(
        self,
        commission_rate: float = 0.001,
        min_order_size: float = 10.0,
        max_order_size: float = 1000000.0,
        latency_ms: int = 50
    ):
        """
        Initialize exchange simulator
        
        Args:
            commission_rate: Trading commission rate
            min_order_size: Minimum order size in USD
            max_order_size: Maximum order size in USD
            latency_ms: Simulated network latency in milliseconds
        """
        self.commission_rate = commission_rate
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        self.latency_ms = latency_ms
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Order tracking
        self.order_counter = 0
        self.order_history = []
        
    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        market_data: Dict[str, Any]
    ) -> OrderResult:
        """Place market order"""
        pass
    
    @abstractmethod
    def get_trading_fees(self, symbol: str, amount: float) -> float:
        """Calculate trading fees"""
        pass
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"ORDER_{self.order_counter:06d}"
    
    def _simulate_latency(self) -> None:
        """Simulate network latency"""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)


class BinanceSimulator(BaseExchangeSimulator):
    """
    Binance exchange simulator
    
    Simulates Binance-specific trading characteristics and fee structure.
    """
    
    def __init__(self, **kwargs):
        # Binance default commission rate
        kwargs.setdefault('commission_rate', 0.001)  # 0.1%
        super().__init__(**kwargs)
        
        # Binance-specific parameters
        self.maker_commission = 0.001
        self.taker_commission = 0.001
        self.bnb_discount = 0.25  # 25% discount with BNB
        
    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        market_data: Dict[str, Any]
    ) -> OrderResult:
        """Place market order on Binance simulator"""
        self._simulate_latency()
        
        order_id = self._generate_order_id()
        
        # Validate order size
        order_value = amount * current_price
        if order_value < self.min_order_size:
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                amount=amount,
                price=current_price,
                executed_amount=0,
                executed_price=0,
                commission=0,
                timestamp=pd.Timestamp.now(),
                status='rejected'
            )
        
        # Simulate slippage based on order size and market conditions
        slippage = self._calculate_binance_slippage(amount, current_price, market_data)
        
        if side.lower() == 'buy':
            executed_price = current_price * (1 + slippage)
        else:
            executed_price = current_price * (1 - slippage)
        
        # Calculate commission
        commission = self._calculate_binance_commission(amount, executed_price)
        
        # Execute order
        executed_amount = amount
        
        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=current_price,
            executed_amount=executed_amount,
            executed_price=executed_price,
            commission=commission,
            timestamp=pd.Timestamp.now(),
            status='filled'
        )
        
        self.order_history.append(result)
        return result
    
    def _calculate_binance_slippage(
        self,
        amount: float,
        price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate Binance-specific slippage"""
        volume = market_data.get('volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        
        # Order size relative to volume
        relative_size = (amount * price) / (volume * price)
        
        # Base slippage
        base_slippage = 0.0005  # 0.05%
        
        # Size impact
        size_impact = relative_size * 0.001
        
        # Volatility impact
        volatility_impact = volatility * 0.5
        
        total_slippage = base_slippage + size_impact + volatility_impact
        
        return min(total_slippage, 0.01)  # Cap at 1%
    
    def _calculate_binance_commission(self, amount: float, price: float) -> float:
        """Calculate Binance commission"""
        trade_value = amount * price
        return trade_value * self.taker_commission
    
    def get_trading_fees(self, symbol: str, amount: float) -> float:
        """Get Binance trading fees"""
        return amount * self.taker_commission


class KucoinSimulator(BaseExchangeSimulator):
    """Kucoin exchange simulator"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('commission_rate', 0.001)
        super().__init__(**kwargs)
        
        # Kucoin fee structure
        self.level1_fee = 0.001  # 0.1%
        self.kcs_discount = 0.2  # 20% discount with KCS
        
    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        market_data: Dict[str, Any]
    ) -> OrderResult:
        """Place market order on Kucoin simulator"""
        self._simulate_latency()
        
        order_id = self._generate_order_id()
        
        # Kucoin-specific slippage calculation
        slippage = self._calculate_kucoin_slippage(amount, current_price, market_data)
        
        if side.lower() == 'buy':
            executed_price = current_price * (1 + slippage)
        else:
            executed_price = current_price * (1 - slippage)
        
        commission = amount * executed_price * self.level1_fee
        
        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=current_price,
            executed_amount=amount,
            executed_price=executed_price,
            commission=commission,
            timestamp=pd.Timestamp.now(),
            status='filled'
        )
        
        self.order_history.append(result)
        return result
    
    def _calculate_kucoin_slippage(
        self,
        amount: float,
        price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate Kucoin-specific slippage"""
        # Kucoin typically has slightly higher slippage
        base_slippage = 0.0008
        
        volume = market_data.get('volume', 500000)
        relative_size = (amount * price) / (volume * price)
        
        size_impact = relative_size * 0.0015
        
        return base_slippage + size_impact
    
    def get_trading_fees(self, symbol: str, amount: float) -> float:
        """Get Kucoin trading fees"""
        return amount * self.level1_fee


class KrakenSimulator(BaseExchangeSimulator):
    """Kraken exchange simulator"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('commission_rate', 0.0026)  # Higher fees
        super().__init__(**kwargs)
        
    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        market_data: Dict[str, Any]
    ) -> OrderResult:
        """Place market order on Kraken simulator"""
        self._simulate_latency()
        
        order_id = self._generate_order_id()
        
        # Kraken has higher fees but better execution
        slippage = self._calculate_kraken_slippage(amount, current_price, market_data)
        
        if side.lower() == 'buy':
            executed_price = current_price * (1 + slippage)
        else:
            executed_price = current_price * (1 - slippage)
        
        commission = amount * executed_price * 0.0026  # Kraken taker fee
        
        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=current_price,
            executed_amount=amount,
            executed_price=executed_price,
            commission=commission,
            timestamp=pd.Timestamp.now(),
            status='filled'
        )
        
        self.order_history.append(result)
        return result
    
    def _calculate_kraken_slippage(
        self,
        amount: float,
        price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate Kraken-specific slippage (generally lower)"""
        base_slippage = 0.0003  # Lower due to better liquidity
        
        volume = market_data.get('volume', 800000)
        relative_size = (amount * price) / (volume * price)
        
        size_impact = relative_size * 0.0008
        
        return base_slippage + size_impact
    
    def get_trading_fees(self, symbol: str, amount: float) -> float:
        """Get Kraken trading fees"""
        return amount * 0.0026


class ExchangeSimulatorFactory:
    """Factory for creating exchange simulators"""
    
    @staticmethod
    def create_simulator(
        exchange_type: ExchangeType,
        **kwargs
    ) -> BaseExchangeSimulator:
        """
        Create exchange simulator
        
        Args:
            exchange_type: Type of exchange to simulate
            **kwargs: Simulator parameters
            
        Returns:
            Exchange simulator instance
        """
        simulators = {
            ExchangeType.BINANCE: BinanceSimulator,
            ExchangeType.KUCOIN: KucoinSimulator,
            ExchangeType.KRAKEN: KrakenSimulator,
            ExchangeType.MEXC: BinanceSimulator,  # Use Binance as template
            ExchangeType.GENERIC: BaseExchangeSimulator
        }
        
        if exchange_type not in simulators:
            raise ValueError(f"Unsupported exchange type: {exchange_type}")
        
        # For abstract base class, create a simple implementation
        if exchange_type == ExchangeType.GENERIC:
            class GenericExchangeSimulator(BaseExchangeSimulator):
                def place_market_order(self, symbol, side, amount, current_price, market_data):
                    order_id = self._generate_order_id()
                    commission = amount * current_price * self.commission_rate
                    
                    return OrderResult(
                        order_id=order_id,
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=current_price,
                        executed_amount=amount,
                        executed_price=current_price,
                        commission=commission,
                        timestamp=pd.Timestamp.now(),
                        status='filled'
                    )
                
                def get_trading_fees(self, symbol, amount):
                    return amount * self.commission_rate
            
            return GenericExchangeSimulator(**kwargs)
        
        return simulators[exchange_type](**kwargs)


class MultiExchangeSimulator:
    """
    Multi-exchange simulator for arbitrage strategies
    
    Simulates trading across multiple exchanges simultaneously.
    """
    
    def __init__(self, exchanges: List[ExchangeType]):
        """
        Initialize multi-exchange simulator
        
        Args:
            exchanges: List of exchanges to simulate
        """
        self.exchanges = {}
        
        for exchange_type in exchanges:
            self.exchanges[exchange_type] = ExchangeSimulatorFactory.create_simulator(
                exchange_type
            )
        
        self.logger = logging.getLogger(__name__)
    
    def get_best_price(self, symbol: str, side: str, market_data: Dict[str, Any]) -> Tuple[ExchangeType, float]:
        """
        Get best price across all exchanges
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            market_data: Current market data
            
        Returns:
            Tuple of (best_exchange, best_price)
        """
        prices = {}
        
        for exchange_type, simulator in self.exchanges.items():
            # Simulate getting current price from each exchange
            base_price = market_data.get('close', 100.0)
            
            # Add exchange-specific spread simulation
            if exchange_type == ExchangeType.BINANCE:
                spread = 0.0005
            elif exchange_type == ExchangeType.KUCOIN:
                spread = 0.0008
            elif exchange_type == ExchangeType.KRAKEN:
                spread = 0.0003
            else:
                spread = 0.0006
            
            if side.lower() == 'buy':
                price = base_price * (1 + spread)
            else:
                price = base_price * (1 - spread)
            
            prices[exchange_type] = price
        
        if side.lower() == 'buy':
            best_exchange = min(prices, key=prices.get)
        else:
            best_exchange = max(prices, key=prices.get)
        
        return best_exchange, prices[best_exchange]
    
    def execute_arbitrage(
        self,
        symbol: str,
        amount: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, OrderResult]:
        """
        Execute arbitrage opportunity across exchanges
        
        Args:
            symbol: Trading symbol
            amount: Amount to trade
            market_data: Current market data
            
        Returns:
            Dictionary of exchange -> order results
        """
        # Find best buy and sell exchanges
        best_buy_exchange, buy_price = self.get_best_price(symbol, 'buy', market_data)
        best_sell_exchange, sell_price = self.get_best_price(symbol, 'sell', market_data)
        
        results = {}
        
        # Only execute if profitable
        if sell_price > buy_price:
            # Buy on cheaper exchange
            buy_result = self.exchanges[best_buy_exchange].place_market_order(
                symbol, 'buy', amount, buy_price, market_data
            )
            results[best_buy_exchange] = buy_result
            
            # Sell on more expensive exchange
            sell_result = self.exchanges[best_sell_exchange].place_market_order(
                symbol, 'sell', amount, sell_price, market_data
            )
            results[best_sell_exchange] = sell_result
            
            self.logger.info(
                f"Arbitrage executed: Buy {amount} at {buy_price} on {best_buy_exchange.value}, "
                f"Sell at {sell_price} on {best_sell_exchange.value}"
            )
        
        return results