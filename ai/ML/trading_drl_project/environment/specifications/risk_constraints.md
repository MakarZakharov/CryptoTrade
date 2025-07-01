# Risk Management Constraints Specification

## Portfolio Risk Limits

### Position Sizing Constraints
- **Maximum Position Size**: 20% of portfolio per asset
- **Minimum Position Size**: 1% to avoid dust trades
- **Leverage Limits**: No leverage > 3x total portfolio value
- **Concentration Limits**: No more than 50% in any single asset class

### Risk Metrics Thresholds
- **Value at Risk (VaR) 95%**: Maximum 5% daily portfolio loss
- **Expected Shortfall**: Maximum 8% daily portfolio loss
- **Maximum Drawdown**: 15% from peak portfolio value
- **Volatility Limit**: Annualized volatility < 40%

### Liquidity Constraints
- **Minimum Cash Reserve**: 5% of portfolio in cash
- **Trading Volume Limits**: Max 10% of daily volume per asset
- **Market Impact Limits**: Estimated slippage < 0.5%

## Dynamic Risk Management

### Circuit Breakers
```python
class CircuitBreaker:
    def __init__(self):
        self.daily_loss_limit = 0.05  # 5%
        self.hourly_loss_limit = 0.02  # 2%
        self.consecutive_loss_limit = 3  # trades
        
    def should_halt_trading(self, portfolio_state):
        # Implementation details
        pass
```

### Correlation Risk
- **Maximum Correlation**: 0.8 between any two positions
- **Diversification Score**: Minimum 0.3 (Herfindahl-Hirschman Index)
- **Sector Exposure**: Max 40% in any crypto sector

### Time-Based Constraints
- **Maximum Holding Period**: 30 days without review
- **Minimum Holding Period**: 1 hour to reduce noise trading
- **Daily Trade Limit**: Maximum 50 trades per day
- **Cool-down Period**: 15 minutes between large trades

## Implementation

### Real-time Monitoring
- Continuous risk metric calculation
- Alert system for constraint violations
- Automatic position adjustment triggers

### Stress Testing
- Monte Carlo scenarios
- Historical stress periods
- Correlation breakdown scenarios
- Liquidity crisis simulations