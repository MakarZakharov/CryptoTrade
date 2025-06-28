"""
Environment Performance Benchmarks

Benchmarks for measuring environment performance and efficiency.
"""

import time
import numpy as np
import pandas as pd
import psutil
import gc
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from environment.core.trading_env import TradingEnv
from environment.core.multi_asset_env import MultiAssetTradingEnv
from environment.core.market_simulator import MarketSimulator
from environment.core.portfolio_manager import PortfolioManager


@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    name: str
    duration: float
    memory_usage: float
    steps_per_second: float
    peak_memory: float
    avg_reward: float
    additional_metrics: Dict[str, Any]


class EnvironmentBenchmark:
    """Environment performance benchmark suite"""
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.results = []
        
    def create_sample_data(self, size: int = 10000, num_assets: int = 1) -> Dict[str, pd.DataFrame]:
        """Create sample data for benchmarking"""
        dates = pd.date_range('2020-01-01', periods=size, freq='H')
        
        data_dict = {}
        
        for i in range(num_assets):
            if num_assets == 1:
                symbol = 'BTC/USDT'
            else:
                symbol = f'ASSET_{i}/USDT'
            
            # Generate realistic price data with some trend and volatility
            np.random.seed(42 + i)  # For reproducible results
            
            returns = np.random.normal(0.0001, 0.02, size)  # Small positive drift with volatility
            prices = 50000 * np.cumprod(1 + returns)  # Start at $50,000
            
            # Create OHLCV data
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, size)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, size))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, size))),
                'close': prices,
                'volume': np.random.uniform(100, 1000, size)
            }, index=dates)
            
            # Ensure high >= low and other price relationships
            data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
            data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
            
            data_dict[symbol] = data
        
        return data_dict
    
    def benchmark_single_asset_env(self, data_size: int = 10000, num_episodes: int = 5) -> BenchmarkResult:
        """Benchmark single asset environment"""
        print(f"Benchmarking Single Asset Environment (size={data_size}, episodes={num_episodes})")
        
        # Create test data
        data_dict = self.create_sample_data(data_size, 1)
        data = data_dict['BTC/USDT']
        
        # Initialize environment
        env = TradingEnv(
            data=data,
            initial_balance=10000.0,
            commission=0.001,
            window_size=50
        )
        
        return self._run_benchmark(env, num_episodes, "Single Asset Environment")
    
    def benchmark_multi_asset_env(self, data_size: int = 5000, num_assets: int = 5, num_episodes: int = 3) -> BenchmarkResult:
        """Benchmark multi-asset environment"""
        print(f"Benchmarking Multi Asset Environment (size={data_size}, assets={num_assets}, episodes={num_episodes})")
        
        # Create test data
        data_dict = self.create_sample_data(data_size, num_assets)
        
        # Initialize environment
        env = MultiAssetTradingEnv(
            data_dict=data_dict,
            initial_balance=10000.0,
            commission=0.001,
            window_size=50,
            action_type="continuous"
        )
        
        return self._run_benchmark(env, num_episodes, f"Multi Asset Environment ({num_assets} assets)")
    
    def benchmark_market_simulator(self, data_size: int = 50000) -> BenchmarkResult:
        """Benchmark market simulator performance"""
        print(f"Benchmarking Market Simulator (size={data_size})")
        
        # Create test data
        data_dict = self.create_sample_data(data_size, 1)
        data = data_dict['BTC/USDT']
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Initialize simulator
        simulator = MarketSimulator(data)
        
        # Benchmark operations
        num_operations = 10000
        for i in range(num_operations):
            step = i % (len(data) - 100)
            simulator.current_idx = step
            
            # Test various operations
            price = simulator.get_current_price()
            ohlcv = simulator.get_current_ohlcv()
            window = simulator.get_observation_window(step, 50)
            is_finished = simulator.is_finished()
            progress = simulator.get_progress()
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        ops_per_second = num_operations / duration
        
        return BenchmarkResult(
            name="Market Simulator",
            duration=duration,
            memory_usage=memory_usage,
            steps_per_second=ops_per_second,
            peak_memory=end_memory,
            avg_reward=0.0,  # N/A for simulator
            additional_metrics={
                'operations': num_operations,
                'data_size': data_size,
                'avg_operation_time_ms': (duration / num_operations) * 1000
            }
        )
    
    def benchmark_portfolio_manager(self, num_trades: int = 10000) -> BenchmarkResult:
        """Benchmark portfolio manager performance"""
        print(f"Benchmarking Portfolio Manager ({num_trades} trades)")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Initialize portfolio manager
        portfolio = PortfolioManager(initial_balance=100000.0, commission_rate=0.001)
        
        # Benchmark trading operations
        np.random.seed(42)
        prices = np.random.uniform(45000, 55000, num_trades)
        actions = np.random.choice(['buy', 'sell', 'hold'], num_trades, p=[0.4, 0.4, 0.2])
        
        successful_trades = 0
        
        for i, (price, action) in enumerate(zip(prices, actions)):
            if action == 'buy':
                result = portfolio.buy(price, quantity=None)  # Buy max possible
                if result.get('success', False):
                    successful_trades += 1
            elif action == 'sell' and portfolio.assets > 0:
                result = portfolio.sell(price, quantity=None)  # Sell all
                if result.get('success', False):
                    successful_trades += 1
            else:
                portfolio.hold(price)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        trades_per_second = num_trades / duration
        
        final_value = portfolio.get_total_value(prices[-1])
        total_return = (final_value / portfolio.initial_balance - 1) * 100
        
        return BenchmarkResult(
            name="Portfolio Manager",
            duration=duration,
            memory_usage=memory_usage,
            steps_per_second=trades_per_second,
            peak_memory=end_memory,
            avg_reward=total_return,
            additional_metrics={
                'total_trades': num_trades,
                'successful_trades': successful_trades,
                'success_rate': successful_trades / num_trades * 100,
                'final_portfolio_value': final_value,
                'total_return_pct': total_return
            }
        )
    
    def _run_benchmark(self, env, num_episodes: int, name: str) -> BenchmarkResult:
        """Run benchmark on environment"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = start_memory
        
        total_steps = 0
        total_reward = 0.0
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                # Random action for benchmarking
                if hasattr(env.action_space, 'sample'):
                    action = env.action_space.sample()
                else:
                    # Fallback for custom action spaces
                    if hasattr(env, 'num_assets'):
                        action = np.random.uniform(-1, 1, env.num_assets)
                    else:
                        action = np.random.randint(0, 3)
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                total_steps += 1
                
                # Track peak memory
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                # Safety limit to prevent infinite episodes
                if total_steps > 50000:
                    break
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
            
            if episode % max(1, num_episodes // 5) == 0:
                print(f"  Episode {episode + 1}/{num_episodes} completed")
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        steps_per_second = total_steps / duration if duration > 0 else 0
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0
        
        # Additional metrics
        additional_metrics = {
            'total_episodes': num_episodes,
            'total_steps': total_steps,
            'avg_steps_per_episode': total_steps / num_episodes,
            'reward_std': np.std(episode_rewards) if episode_rewards else 0,
            'min_episode_reward': min(episode_rewards) if episode_rewards else 0,
            'max_episode_reward': max(episode_rewards) if episode_rewards else 0
        }
        
        return BenchmarkResult(
            name=name,
            duration=duration,
            memory_usage=memory_usage,
            steps_per_second=steps_per_second,
            peak_memory=peak_memory,
            avg_reward=avg_reward,
            additional_metrics=additional_metrics
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark tests"""
        print("Starting Environment Benchmark Suite")
        print("=" * 50)
        
        benchmarks = [
            lambda: self.benchmark_single_asset_env(data_size=5000, num_episodes=3),
            lambda: self.benchmark_multi_asset_env(data_size=2000, num_assets=3, num_episodes=2),
            lambda: self.benchmark_market_simulator(data_size=10000),
            lambda: self.benchmark_portfolio_manager(num_trades=5000),
        ]
        
        results = []
        
        for i, benchmark in enumerate(benchmarks):
            print(f"\nRunning benchmark {i + 1}/{len(benchmarks)}")
            
            # Force garbage collection before each benchmark
            gc.collect()
            
            try:
                result = benchmark()
                results.append(result)
                self.results.append(result)
                
                print(f"✓ {result.name} completed")
                print(f"  Duration: {result.duration:.2f}s")
                print(f"  Steps/sec: {result.steps_per_second:.1f}")
                print(f"  Memory: {result.memory_usage:.1f} MB")
                
            except Exception as e:
                print(f"✗ Benchmark failed: {e}")
                continue
        
        return results
    
    def generate_report(self, results: List[BenchmarkResult] = None) -> str:
        """Generate benchmark report"""
        if results is None:
            results = self.results
        
        if not results:
            return "No benchmark results available"
        
        report = ["Environment Benchmark Report"]
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("Performance Summary:")
        report.append("-" * 30)
        
        for result in results:
            report.append(f"{result.name}:")
            report.append(f"  Duration: {result.duration:.2f}s")
            report.append(f"  Steps/sec: {result.steps_per_second:.1f}")
            report.append(f"  Memory usage: {result.memory_usage:.1f} MB")
            report.append(f"  Peak memory: {result.peak_memory:.1f} MB")
            if result.avg_reward != 0:
                report.append(f"  Avg reward: {result.avg_reward:.4f}")
            report.append("")
        
        # Detailed metrics
        report.append("Detailed Metrics:")
        report.append("-" * 30)
        
        for result in results:
            report.append(f"\n{result.name}:")
            for key, value in result.additional_metrics.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
        
        return "\n".join(report)


def main():
    """Run benchmark suite"""
    benchmark = EnvironmentBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)
    
    report = benchmark.generate_report(results)
    print(report)
    
    # Save report to file
    with open("environment_benchmark_report.txt", "w") as f:
        f.write(report)
    
    print("\nReport saved to environment_benchmark_report.txt")


if __name__ == "__main__":
    main()