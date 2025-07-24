"""
Backtesting engine for STAS_ML v2
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.config import Config
from ..core.base import BaseBacktester, BaseModel, MetricsCalculator, Logger


class BacktestEngine(BaseBacktester):
    """Advanced backtesting engine."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = Logger("BacktestEngine")
    
    def run_backtest(self, model: BaseModel, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive backtest."""
        try:
            # Generate signals
            signals = self._generate_signals(model, data)
            
            # Execute trades
            trades = self._execute_trades(signals, data)
            
            # Calculate returns
            returns = self._calculate_returns(trades, data)
            
            # Calculate metrics
            metrics = self.calculate_metrics(returns)
            
            # Add additional info
            metrics.update({
                'total_signals': len(signals[signals != 0]),
                'total_trades': len(trades),
                'backtest_period': f"{data.index[0]} to {data.index[-1]}",
                'initial_capital': self.config.backtest.initial_capital
            })
            
            self.backtest_results = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}
    
    def _generate_signals(self, model: BaseModel, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        # Create features (this should match the training data preprocessing)
        # For now, simplified approach
        features = self._prepare_features(data)
        
        if len(features) == 0:
            return pd.Series(0, index=data.index)
        
        # Get predictions
        predictions = model.predict(features)
        
        # Convert to signals
        if model.is_classification():
            # Binary classification: 1 for buy, 0 for sell/hold
            signals = predictions
        else:
            # Regression: threshold-based signals
            threshold = self.config.backtest.signal_threshold
            signals = np.where(predictions > threshold, 1, 
                             np.where(predictions < -threshold, -1, 0))
        
        # Apply confidence filter if available
        if hasattr(model, 'predict_proba') and model.is_classification():
            try:
                probabilities = model.predict_proba(features)
                confidence = np.max(probabilities, axis=1)
                signals = np.where(confidence >= self.config.backtest.min_confidence, signals, 0)
            except:
                pass
        
        # Create signals series aligned with data
        signal_index = data.index[-len(signals):]
        return pd.Series(signals, index=signal_index)
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction (simplified)."""
        # This is a simplified version - in practice, should match training preprocessing
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_cols):
            return np.array([])
        
        # Basic features
        df = data[required_cols].copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(14).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Remove NaN and create feature matrix
        df = df.dropna()
        
        if len(df) < self.config.features.lookback_window:
            return np.array([])
        
        # Simple lookback window approach
        features = []
        lookback = min(self.config.features.lookback_window, 10)  # Simplified
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i].values.flatten()
            features.append(window)
        
        return np.array(features) if features else np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Simple RSI calculation."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _execute_trades(self, signals: pd.Series, data: pd.DataFrame) -> list:
        """Execute trades based on signals."""
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for date, signal in signals.items():
            if date not in data.index:
                continue
                
            current_price = data.loc[date, 'close']
            
            # Entry logic
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                entry_date = date
                
            # Exit logic
            elif position != 0:
                should_exit = False
                exit_reason = ""
                
                # Signal-based exit
                if signal == -position:
                    should_exit = True
                    exit_reason = "signal"
                
                # Stop loss/take profit
                elif position > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    if self.config.backtest.stop_loss and pnl_pct <= -self.config.backtest.stop_loss:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif self.config.backtest.take_profit and pnl_pct >= self.config.backtest.take_profit:
                        should_exit = True
                        exit_reason = "take_profit"
                
                # Time-based exit
                if self.config.backtest.max_holding_period:
                    days_held = (date - entry_date).days
                    if days_held >= self.config.backtest.max_holding_period:
                        should_exit = True
                        exit_reason = "time_limit"
                
                if should_exit:
                    pnl = (current_price - entry_price) * position
                    pnl_pct = pnl / entry_price * position
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_date = None
        
        return trades
    
    def _calculate_returns(self, trades: list, data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from trades."""
        if not trades:
            return pd.Series(0, index=data.index)
        
        # Create returns series
        returns = pd.Series(0.0, index=data.index)
        
        for trade in trades:
            # Simple approach: assign return to exit date
            if trade['exit_date'] in returns.index:
                # Account for transaction costs
                gross_return = trade['pnl_pct']
                transaction_cost = self.config.backtest.transaction_cost * 2  # Entry + exit
                net_return = gross_return - transaction_cost
                
                returns.loc[trade['exit_date']] = net_return
        
        return returns
    
    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0 or returns.sum() == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic metrics using MetricsCalculator
        basic_metrics = MetricsCalculator.trading_metrics(returns)
        
        # Additional metrics
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        additional_metrics = {
            'profit_factor': abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else float('inf'),
            'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'win_loss_ratio': abs(winning_trades.mean() / losing_trades.mean()) if len(losing_trades) > 0 else float('inf')
        }
        
        # Combine metrics
        basic_metrics.update(additional_metrics)
        return basic_metrics