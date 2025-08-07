#!/usr/bin/env python3
"""
Test script for optimized MomentumStrategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_backtester import UniversalBacktester

def main():
    """Test the optimized MomentumStrategy"""
    print("🚀 Testing Optimized MomentumStrategy")
    print("=" * 60)
    
    # Create backtester
    backtester = UniversalBacktester(
        initial_cash=100000,
        commission=0.001,
        spread=0.0005,
        slippage=0.0002
    )
    
    # Test the optimized MomentumStrategy
    try:
        result = backtester.run_single_backtest(
            strategy_name="MomentumStrategy",
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1d",
            show_plot=False,
            verbose=True
        )
        
        print("\n🎯 TARGET ANALYSIS:")
        print("=" * 40)
        print(f"Target: 100+ trades | Achieved: {result.get('total_trades', 0)} trades")
        print(f"Target: $2000+ profit | Achieved: ${result.get('profit_loss', 0):.2f} profit")
        
        trades_ok = result.get('total_trades', 0) >= 100
        profit_ok = result.get('profit_loss', 0) >= 2000
        
        print(f"\n{'✅' if trades_ok else '❌'} Trades target: {'ACHIEVED' if trades_ok else 'NOT ACHIEVED'}")
        print(f"{'✅' if profit_ok else '❌'} Profit target: {'ACHIEVED' if profit_ok else 'NOT ACHIEVED'}")
        
        if trades_ok and profit_ok:
            print("\n🎉 SUCCESS! Both targets achieved!")
        else:
            print("\n⚠️ Targets not fully achieved. Consider further optimization.")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing strategy: {e}")
        return None

if __name__ == "__main__":
    result = main()