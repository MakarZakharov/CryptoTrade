"""
Basic usage example of CryptoTradingEnv.

This script demonstrates:
1. Creating an environment with default configuration
2. Running a random policy for one episode
3. Accessing trade history and metrics
4. Exporting results

Usage:
    python basic_usage.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from DRL.Environment import CryptoTradingEnv, get_config
import numpy as np


def main():
    """Run basic trading environment example."""

    print("\n" + "="*60)
    print("  CryptoTradingEnv - Basic Usage Example")
    print("="*60 + "\n")

    # Create environment with default configuration
    print("📊 Creating environment...")
    config = get_config(
        "default",
        symbol="BTCUSDT",
        timeframe="1h",
        window_size=50,
        max_episode_steps=500,
        initial_balance=10000.0,
        seed=42,
        verbose=1  # Print episode summary
    )

    env = CryptoTradingEnv(config)
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Symbol: {config.symbol}")
    print(f"  - Timeframe: {config.timeframe}")
    print(f"  - Initial Balance: ${config.initial_balance:,.2f}")

    # Reset environment
    print("\n🔄 Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset")
    print(f"  - Start time: {info['timestamp']}")
    print(f"  - Start price: ${info['price']:,.2f}")

    # Run episode with random policy
    print("\n🎲 Running random policy...")
    print("-" * 60)

    episode_step = 0
    total_reward = 0
    done = False

    while not done:
        # Sample random action
        action = env.action_space.sample()

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        episode_step += 1

        # Print progress every 50 steps
        if episode_step % 50 == 0:
            print(f"Step {episode_step:4d} | "
                  f"Price: ${info['price']:8,.2f} | "
                  f"Position: {info['position']:7.4f} | "
                  f"Portfolio: ${info['portfolio_value']:10,.2f} | "
                  f"Total Reward: {total_reward:8.2f}")

    print("-" * 60)
    print(f"✓ Episode completed in {episode_step} steps\n")

    # Display episode metrics
    if 'episode_metrics' in info:
        metrics = info['episode_metrics']

        print("\n📈 Episode Performance Metrics")
        print("="*60)
        print(f"Total Return:        {metrics['total_return_pct']:>10.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%")
        print(f"Volatility:          {metrics['volatility']:>10.2f}")
        print(f"Total Trades:        {metrics['total_trades']:>10d}")
        print(f"Win Rate:            {metrics['win_rate']*100:>10.2f}%")
        print(f"Total Fees:          ${metrics['total_fees']:>10.2f}")
        print(f"Final Value:         ${metrics['final_portfolio_value']:>10,.2f}")
        print("="*60)

    # Get trade history
    print("\n📋 Trade History")
    print("="*60)
    trade_df = env.get_trade_history()

    if len(trade_df) > 0:
        print(f"Total trades executed: {len(trade_df)}")
        print("\nFirst 5 trades:")
        print(trade_df.head()[['timestamp', 'size', 'price', 'fee', 'position', 'portfolio_value']])

        print("\nLast 5 trades:")
        print(trade_df.tail()[['timestamp', 'size', 'price', 'fee', 'position', 'portfolio_value']])

        # Export trade history
        output_file = "trade_history.csv"
        trade_df.to_csv(output_file, index=False)
        print(f"\n✓ Trade history exported to: {output_file}")
    else:
        print("No trades executed during episode")

    # Close environment
    env.close()

    print("\n" + "="*60)
    print("  ✅ Example completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
