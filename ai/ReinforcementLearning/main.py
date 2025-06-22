"""
Main entry point for RL Trading System
Example usage and training script
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from CryptoTrade.ai.ReinforcementLearning.environment.trading_env import TradingEnvironment
from CryptoTrade.ai.ReinforcementLearning.agents.ppo_agent import PPOAgent
from CryptoTrade.ai.ReinforcementLearning.training.trainer import RLTrainer
from CryptoTrade.ai.ReinforcementLearning.config import *


def load_data(file_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load and prepare trading data
    
    Args:
        file_path: Path to CSV file with OHLCV data
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Prepared DataFrame
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Filter by date if specified
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by index
    df.sort_index(inplace=True)
    
    # Remove any NaN values
    df.dropna(inplace=True)
    
    print(f"Loaded data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    return df


def get_all_data_files(data_dir: str = "../../data", 
                      exchange: str = None, 
                      symbol: str = None, 
                      timeframe: str = None) -> List[str]:
    """
    Get all available data files from the project
    
    Args:
        data_dir: Base data directory
        exchange: Filter by exchange (e.g., 'binance', 'Kraken')
        symbol: Filter by symbol (e.g., 'BTCUSDT')
        timeframe: Filter by timeframe (e.g., '1d', '4h')
        
    Returns:
        List of file paths
    """
    import glob
    
    # Build pattern
    pattern_parts = [data_dir]
    
    if exchange:
        pattern_parts.append(exchange)
    else:
        pattern_parts.append("*")
    
    if symbol:
        pattern_parts.append(symbol)
    else:
        pattern_parts.append("*")
    
    if timeframe:
        pattern_parts.append(timeframe)
    else:
        pattern_parts.append("*")
    
    pattern_parts.append("*.csv")
    
    pattern = os.path.join(*pattern_parts)
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} data files matching criteria")
    for file in files[:5]:  # Show first 5 files
        print(f"  - {file}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")
    
    return files


def load_multiple_datasets(file_paths: List[str], 
                         start_date: str = None, 
                         end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load multiple datasets
    
    Args:
        file_paths: List of file paths to load
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Dictionary mapping file path to DataFrame
    """
    datasets = {}
    
    for file_path in file_paths:
        try:
            df = load_data(file_path, start_date, end_date)
            if len(df) > 0:
                datasets[file_path] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"\nSuccessfully loaded {len(datasets)} datasets")
    return datasets


def select_data_interactive(data_dir: str = "../../data") -> str:
    """
    Interactively select data file for training
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Selected file path
    """
    # Get all available data files
    all_files = get_all_data_files(data_dir)
    
    if not all_files:
        raise ValueError("No data files found")
    
    print("\n" + "=" * 60)
    print("AVAILABLE DATA FILES")
    print("=" * 60)
    
    # Group files by exchange and symbol
    file_groups = {}
    for i, file_path in enumerate(all_files):
        parts = file_path.split(os.sep)
        if len(parts) >= 4:
            exchange = parts[-4]
            symbol = parts[-3]
            timeframe = parts[-2]
            key = f"{exchange}/{symbol}/{timeframe}"
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append((i, file_path))
    
    # Display options
    idx = 0
    file_map = {}
    for group, files in sorted(file_groups.items()):
        print(f"\n{group}:")
        for i, file_path in files:
            idx += 1
            file_map[idx] = file_path
            filename = os.path.basename(file_path)
            print(f"  [{idx}] {filename}")
    
    # Get user selection
    print("\n" + "-" * 60)
    while True:
        try:
            selection = input("Select data file number (or 'q' to quit): ")
            if selection.lower() == 'q':
                sys.exit(0)
            
            selection = int(selection)
            if selection in file_map:
                selected_file = file_map[selection]
                print(f"\nSelected: {selected_file}")
                return selected_file
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def train_rl_agent(
    data_path: str = None,
    timeframe: str = '1h',
    agent_type: str = 'ppo',
    episodes: int = 1000,
    validation_split: float = 0.2,
    interactive: bool = True,
    **kwargs
):
    """
    Train RL trading agent
    
    Args:
        data_path: Path to trading data (if None and interactive=True, will ask user)
        timeframe: Trading timeframe
        agent_type: Type of agent ('ppo', 'dqn', 'a2c')
        episodes: Number of training episodes
        validation_split: Validation data split
        interactive: Whether to use interactive data selection
        **kwargs: Additional arguments
    """
    # Select data file if not provided
    if data_path is None and interactive:
        data_path = select_data_interactive()
    elif data_path is None:
        raise ValueError("data_path must be provided when interactive=False")
    
    # Extract timeframe from file path if not specified
    if timeframe == 'auto':
        path_parts = data_path.split(os.sep)
        if len(path_parts) >= 2:
            timeframe = path_parts[-2]
            print(f"Auto-detected timeframe: {timeframe}")
    
    print("\n" + "=" * 60)
    print("RL TRADING SYSTEM")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Timeframe: {timeframe}")
    print(f"Agent: {agent_type.upper()}")
    print(f"Episodes: {episodes}")
    print(f"Device: {DEVICE}")
    print("-" * 60)
    
    # Load data
    data = load_data(data_path, kwargs.get('start_date'), kwargs.get('end_date'))
    
    # Split data
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    print("-" * 60)
    
    # Create environments
    train_env = TradingEnvironment(
        data=train_data,
        timeframe=timeframe,
        initial_balance=INITIAL_BALANCE,
        trading_fees=TRADING_FEES,
        max_position_size=MAX_POSITION_SIZE
    )
    
    val_env = TradingEnvironment(
        data=val_data,
        timeframe=timeframe,
        initial_balance=INITIAL_BALANCE,
        trading_fees=TRADING_FEES,
        max_position_size=MAX_POSITION_SIZE
    )
    
    # Create agent
    if agent_type.lower() == 'ppo':
        agent = PPOAgent(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported yet")
    
    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        train_env=train_env,
        val_env=val_env,
        config={
            'episodes': episodes,
            'steps_per_episode': STEPS_PER_EPISODE,
            'checkpoint_frequency': CHECKPOINT_FREQUENCY,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE
        }
    )
    
    # Train agent
    trainer.train()
    
    return trainer


def backtest_agent(
    model_path: str,
    data_path: str,
    timeframe: str = '1h',
    **kwargs
):
    """
    Backtest trained agent
    
    Args:
        model_path: Path to saved model
        data_path: Path to test data
        timeframe: Trading timeframe
        **kwargs: Additional arguments
    """
    print("=" * 60)
    print("BACKTESTING RL AGENT")
    print("=" * 60)
    
    # Load data
    data = load_data(data_path, kwargs.get('start_date'), kwargs.get('end_date'))
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        timeframe=timeframe,
        initial_balance=INITIAL_BALANCE,
        trading_fees=TRADING_FEES,
        max_position_size=MAX_POSITION_SIZE
    )
    
    # Create and load agent
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    agent.load(model_path)
    
    print(f"Loaded model from: {model_path}")
    print("-" * 60)
    
    # Run backtest
    obs = env.reset()
    done = False
    step = 0
    
    while not done:
        # Get action (deterministic for testing)
        action = agent.act(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Portfolio Value = ${info['portfolio_value']:.2f}")
        
        step += 1
    
    # Get final metrics
    metrics = env.get_metrics()
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Final Portfolio Value: ${metrics['portfolio_value']:.2f}")
    print(f"Total Profit: ${metrics['total_profit']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    
    return metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RL Trading System')
    parser.add_argument('mode', choices=['train', 'backtest'], help='Mode to run')
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    parser.add_argument('--agent', default='ppo', choices=['ppo'], help='Agent type')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model', help='Path to model file (for backtesting)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_rl_agent(
            data_path=args.data,
            timeframe=args.timeframe,
            agent_type=args.agent,
            episodes=args.episodes,
            start_date=args.start_date,
            end_date=args.end_date
        )
    elif args.mode == 'backtest':
        if not args.model:
            raise ValueError("Model path required for backtesting")
        
        backtest_agent(
            model_path=args.model,
            data_path=args.data,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )


if __name__ == "__main__":
    # Example usage without command line arguments
    # Uncomment and modify as needed
    
    # Train example - interactive selection enabled by default
    train_rl_agent(
        data_path=None,  # Will prompt for selection
        timeframe='auto',  # Will auto-detect from selected file
        agent_type='ppo',
        episodes=100,  # Start with fewer episodes for testing
        validation_split=0.2,
        interactive=True  # Enable interactive selection
    )
    
    # Backtest example
    # backtest_agent(
    #     model_path="models/20240101_120000/best_model.pth",
    #     data_path="../../data/BTCUSDT_1h_test.csv",
    #     timeframe='1h'
    # )