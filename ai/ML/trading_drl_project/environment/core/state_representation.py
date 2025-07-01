"""
State Representation

Advanced state representation methods for trading environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import talib


class StateRepresentation:
    """
    Advanced state representation for trading environments
    
    Provides various methods to represent market and portfolio state
    for DRL agents.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        normalization_method: str = "standard",  # "standard", "minmax", "robust"
        include_technical_indicators: bool = True,
        include_portfolio_state: bool = True,
        include_market_microstructure: bool = False
    ):
        """
        Initialize state representation
        
        Args:
            window_size: Number of time steps in observation window
            normalization_method: Method for normalizing features
            include_technical_indicators: Whether to include technical indicators
            include_portfolio_state: Whether to include portfolio state
            include_market_microstructure: Whether to include microstructure features
        """
        self.window_size = window_size
        self.normalization_method = normalization_method
        self.include_technical_indicators = include_technical_indicators
        self.include_portfolio_state = include_portfolio_state
        self.include_market_microstructure = include_market_microstructure
        
        # Initialize scalers
        self.scalers = {}
        self._setup_scalers()
        
    def _setup_scalers(self) -> None:
        """Setup normalization scalers"""
        if self.normalization_method == "standard":
            self.price_scaler = StandardScaler()
            self.volume_scaler = StandardScaler()
            self.indicator_scaler = StandardScaler()
        elif self.normalization_method == "minmax":
            self.price_scaler = MinMaxScaler()
            self.volume_scaler = MinMaxScaler()
            self.indicator_scaler = MinMaxScaler()
        else:
            # Robust scaling for outliers
            from sklearn.preprocessing import RobustScaler
            self.price_scaler = RobustScaler()
            self.volume_scaler = RobustScaler()
            self.indicator_scaler = RobustScaler()
    
    def create_observation(
        self,
        market_data: pd.DataFrame,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_step: int = 0
    ) -> np.ndarray:
        """
        Create observation from market data and portfolio state
        
        Args:
            market_data: OHLCV market data
            portfolio_state: Current portfolio state
            current_step: Current time step
            
        Returns:
            Normalized observation array
        """
        # Extract window of market data
        start_idx = max(0, current_step - self.window_size + 1)
        end_idx = current_step + 1
        window_data = market_data.iloc[start_idx:end_idx].copy()
        
        # Pad if needed
        if len(window_data) < self.window_size:
            padding = self.window_size - len(window_data)
            first_row = market_data.iloc[0:1]
            padding_data = pd.concat([first_row] * padding, ignore_index=True)
            window_data = pd.concat([padding_data, window_data], ignore_index=True)
        
        observations = []
        
        # 1. Basic OHLCV features
        ohlcv_features = self._extract_ohlcv_features(window_data)
        observations.append(ohlcv_features)
        
        # 2. Technical indicators
        if self.include_technical_indicators:
            technical_features = self._extract_technical_indicators(window_data)
            observations.append(technical_features)
        
        # 3. Portfolio state features
        if self.include_portfolio_state and portfolio_state:
            portfolio_features = self._extract_portfolio_features(
                portfolio_state, self.window_size
            )
            observations.append(portfolio_features)
        
        # 4. Market microstructure features
        if self.include_market_microstructure:
            microstructure_features = self._extract_microstructure_features(window_data)
            observations.append(microstructure_features)
        
        # Combine all features
        combined_observation = np.concatenate(observations, axis=1)
        
        return combined_observation.astype(np.float32)
    
    def _extract_ohlcv_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract and normalize OHLCV features"""
        # Basic OHLCV
        ohlcv = data[['open', 'high', 'low', 'close', 'volume']].values
        
        # Price returns
        close_prices = data['close'].values
        returns = np.diff(close_prices, prepend=close_prices[0]) / close_prices
        
        # Log returns
        log_returns = np.diff(np.log(close_prices), prepend=0)
        
        # High-Low spread
        hl_spread = (data['high'] - data['low']) / data['close']
        
        # Open-Close spread  
        oc_spread = (data['close'] - data['open']) / data['open']
        
        # Volume features
        volume_ma = data['volume'].rolling(window=5, min_periods=1).mean()
        volume_ratio = data['volume'] / volume_ma
        
        # Combine features
        features = np.column_stack([
            ohlcv,
            returns.reshape(-1, 1),
            log_returns.reshape(-1, 1),
            hl_spread.values.reshape(-1, 1),
            oc_spread.values.reshape(-1, 1),
            volume_ratio.values.reshape(-1, 1)
        ])
        
        return self._normalize_features(features, 'price')
    
    def _extract_technical_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """Extract technical indicators"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        indicators = []
        
        # Moving averages
        sma_10 = talib.SMA(close, timeperiod=10)
        sma_20 = talib.SMA(close, timeperiod=20)
        ema_12 = talib.EMA(close, timeperiod=12)
        ema_26 = talib.EMA(close, timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        
        # Williams %R
        willr = talib.WILLR(high, low, close)
        
        # Average True Range
        atr = talib.ATR(high, low, close)
        
        # Commodity Channel Index
        cci = talib.CCI(high, low, close)
        
        # Volume indicators
        ad = talib.AD(high, low, close, volume)
        obv = talib.OBV(close, volume)
        
        # Combine all indicators
        all_indicators = [
            sma_10, sma_20, ema_12, ema_26,
            macd, macd_signal, macd_hist,
            rsi, bb_width, bb_position,
            stoch_k, stoch_d, willr,
            atr, cci, ad, obv
        ]
        
        # Stack and fill NaN values
        indicators_array = np.column_stack(all_indicators)
        indicators_array = np.nan_to_num(indicators_array, nan=0.0)
        
        return self._normalize_features(indicators_array, 'indicator')
    
    def _extract_portfolio_features(
        self, 
        portfolio_state: Dict[str, Any], 
        window_size: int
    ) -> np.ndarray:
        """Extract portfolio state features"""
        # Portfolio-level features
        total_value = portfolio_state.get('portfolio_value', 0)
        cash_ratio = portfolio_state.get('cash_balance', 0) / total_value if total_value > 0 else 1
        total_pnl = portfolio_state.get('total_pnl', 0)
        sharpe_ratio = portfolio_state.get('sharpe_ratio', 0)
        max_drawdown = portfolio_state.get('max_drawdown', 0)
        
        # Position information
        positions = portfolio_state.get('asset_positions', {})
        position_count = len([p for p in positions.values() if p > 0.01])
        max_position = max(positions.values()) if positions else 0
        position_diversity = 1 - sum(p**2 for p in positions.values())  # Herfindahl index
        
        portfolio_features = np.array([
            total_value,
            cash_ratio, 
            total_pnl,
            sharpe_ratio,
            max_drawdown,
            position_count,
            max_position,
            position_diversity
        ])
        
        # Repeat for window size
        portfolio_features = np.repeat(
            portfolio_features.reshape(1, -1),
            window_size,
            axis=0
        )
        
        return portfolio_features
    
    def _extract_microstructure_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract market microstructure features"""
        # Price impact proxies
        close = data['close'].values
        volume = data['volume'].values
        
        # VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        vwap_deviation = (close - vwap.values) / vwap.values
        
        # Volatility measures
        returns = np.diff(close, prepend=close[0]) / close
        realized_vol = pd.Series(returns).rolling(window=10, min_periods=1).std()
        
        # Bid-ask spread proxy (high-low as percentage of close)
        spread_proxy = (data['high'] - data['low']) / data['close']
        
        # Volume-weighted price change
        price_change = np.diff(close, prepend=close[0])
        volume_weighted_change = price_change * volume / np.mean(volume)
        
        # Market impact proxy
        volume_normalized = volume / np.mean(volume)
        impact_proxy = np.abs(returns) / (volume_normalized + 1e-8)
        
        microstructure_features = np.column_stack([
            vwap_deviation.reshape(-1, 1),
            realized_vol.values.reshape(-1, 1),
            spread_proxy.values.reshape(-1, 1),
            volume_weighted_change.reshape(-1, 1),
            impact_proxy.reshape(-1, 1)
        ])
        
        return self._normalize_features(microstructure_features, 'microstructure')
    
    def _normalize_features(self, features: np.ndarray, feature_type: str) -> np.ndarray:
        """Normalize features using appropriate scaler"""
        if feature_type not in self.scalers:
            if feature_type == 'price':
                scaler = self.price_scaler
            elif feature_type == 'volume':
                scaler = self.volume_scaler
            else:
                scaler = self.indicator_scaler
            
            # Fit scaler on non-NaN values
            valid_data = features[~np.isnan(features).any(axis=1)]
            if len(valid_data) > 0:
                scaler.fit(valid_data)
            self.scalers[feature_type] = scaler
        
        # Transform features
        scaler = self.scalers[feature_type]
        
        # Handle NaN values
        features_clean = np.nan_to_num(features, nan=0.0)
        
        try:
            normalized = scaler.transform(features_clean)
        except:
            # Fallback to original features if transformation fails
            normalized = features_clean
        
        return normalized
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in observation"""
        feature_names = []
        
        # OHLCV features
        ohlcv_names = ['open', 'high', 'low', 'close', 'volume', 
                       'returns', 'log_returns', 'hl_spread', 'oc_spread', 'volume_ratio']
        feature_names.extend(ohlcv_names)
        
        # Technical indicators
        if self.include_technical_indicators:
            indicator_names = [
                'sma_10', 'sma_20', 'ema_12', 'ema_26',
                'macd', 'macd_signal', 'macd_hist',
                'rsi', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'willr',
                'atr', 'cci', 'ad', 'obv'
            ]
            feature_names.extend(indicator_names)
        
        # Portfolio features
        if self.include_portfolio_state:
            portfolio_names = [
                'total_value', 'cash_ratio', 'total_pnl', 'sharpe_ratio',
                'max_drawdown', 'position_count', 'max_position', 'position_diversity'
            ]
            feature_names.extend(portfolio_names)
        
        # Microstructure features
        if self.include_market_microstructure:
            microstructure_names = [
                'vwap_deviation', 'realized_vol', 'spread_proxy',
                'volume_weighted_change', 'impact_proxy'
            ]
            feature_names.extend(microstructure_names)
        
        return feature_names