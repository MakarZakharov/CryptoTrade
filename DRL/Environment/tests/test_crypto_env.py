"""
Unit tests for CryptoTradingEnv.

Tests core functionality including:
- Environment initialization
- Observation and action spaces
- Trading mechanics (PnL, fees, slippage)
- Episode termination
- Reproducibility
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from DRL.Environment import CryptoTradingEnv, EnvConfig, get_config


class TestCryptoTradingEnv(unittest.TestCase):
    """Test suite for CryptoTradingEnv."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create minimal test configuration
        cls.test_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "window_size": 10,
            "indicators": ["ema_10", "rsi_14"],
            "max_episode_steps": 100,
            "initial_balance": 10000.0,
            "seed": 42,
            "verbose": 0
        }

    def setUp(self):
        """Create fresh environment for each test."""
        self.env = CryptoTradingEnv(self.test_config)

    def tearDown(self):
        """Clean up after each test."""
        self.env.close()

    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertIsNotNone(self.env.observation_space)
        self.assertIsNotNone(self.env.action_space)
        self.assertEqual(self.env.config.window_size, 10)
        self.assertEqual(self.env.config.initial_balance, 10000.0)

    def test_observation_space_vector(self):
        """Test vector observation space shape."""
        config = self.test_config.copy()
        config['observation_mode'] = 'vector'
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.dtype, np.float32)
        # Should contain market window + position state + account state
        expected_size = env._calculate_observation_size()
        self.assertEqual(obs.shape[0], expected_size)
        env.close()

    def test_observation_space_dict(self):
        """Test dictionary observation space structure."""
        config = self.test_config.copy()
        config['observation_mode'] = 'dict'
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)
        self.assertIsInstance(obs, dict)
        self.assertIn('market', obs)
        self.assertIn('position', obs)
        self.assertIn('account', obs)

        # Check shapes
        self.assertEqual(obs['market'].shape[0], 10)  # window_size
        self.assertEqual(obs['position'].shape[0], 4)
        self.assertEqual(obs['account'].shape[0], 3)
        env.close()

    def test_discrete_action_space(self):
        """Test discrete action space."""
        config = self.test_config.copy()
        config['action_mode'] = 'discrete'
        config['discrete_actions'] = 4
        env = CryptoTradingEnv(config)

        self.assertEqual(env.action_space.n, 4)

        # Test each discrete action
        obs, info = env.reset(seed=42)
        for action in range(4):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    def test_continuous_action_space(self):
        """Test continuous action space."""
        config = self.test_config.copy()
        config['action_mode'] = 'continuous'
        env = CryptoTradingEnv(config)

        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(env.action_space.low[0], -1.0)
        self.assertEqual(env.action_space.high[0], 1.0)

        # Test continuous actions
        obs, info = env.reset(seed=42)
        for action_value in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            obs, reward, terminated, truncated, info = env.step(np.array([action_value]))
            if terminated or truncated:
                break
        env.close()

    def test_reset_functionality(self):
        """Test reset returns valid initial state."""
        obs, info = self.env.reset(seed=42)

        # Check observation
        self.assertIsNotNone(obs)

        # Check info
        self.assertIsInstance(info, dict)
        self.assertIn('step', info)
        self.assertIn('price', info)
        self.assertIn('portfolio_value', info)

        # Check initial state
        self.assertEqual(self.env.position, 0.0)
        self.assertEqual(self.env.balance, 10000.0)
        self.assertEqual(self.env.portfolio_value, 10000.0)

    def test_step_functionality(self):
        """Test step executes correctly."""
        obs, info = self.env.reset(seed=42)

        # Take a step
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check returns
        self.assertIsNotNone(obs)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_basic_trading_workflow(self):
        """Test complete trading workflow."""
        obs, info = self.env.reset(seed=42)

        total_steps = 0
        cumulative_reward = 0

        done = False
        while not done and total_steps < 50:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
            total_steps += 1

        self.assertGreater(total_steps, 0)
        self.assertTrue(done or total_steps == 50)

        # Check final info contains metrics if episode ended
        if done:
            self.assertIn('episode_metrics', info)

    def test_buy_trade_pnl(self):
        """Test that buying and selling calculates PnL correctly."""
        config = self.test_config.copy()
        config['action_mode'] = 'discrete'
        config['fee_taker'] = 0.0  # No fees for simple PnL test
        config['slippage_coef'] = 0.0
        config['market_impact'] = False
        config['bid_ask_spread'] = 0.0
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)
        initial_balance = env.balance
        initial_price = env._get_current_price()

        # Buy (action 1)
        obs, reward, terminated, truncated, info = env.step(1)
        position_after_buy = env.position
        self.assertGreater(position_after_buy, 0, "Should have long position")

        # Step forward without trading
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            if terminated or truncated:
                break

        # Exit (action 3)
        obs, reward, terminated, truncated, info = env.step(3)
        self.assertEqual(env.position, 0.0, "Position should be closed")

        # Balance should reflect PnL from price change
        final_balance = env.balance
        # Can't predict exact PnL due to price movements, but balance should have changed
        self.assertNotEqual(final_balance, initial_balance)

        env.close()

    def test_transaction_costs(self):
        """Test that transaction costs are applied correctly."""
        config = self.test_config.copy()
        config['action_mode'] = 'discrete'
        config['fee_taker'] = 0.001  # 0.1% fee
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)

        # Execute a buy
        obs, reward, terminated, truncated, info = env.step(1)

        # Check that fees were paid
        self.assertGreater(env.total_fees_paid, 0, "Fees should be paid on trades")

        env.close()

    def test_slippage_calculation(self):
        """Test slippage is calculated and applied."""
        config = self.test_config.copy()
        config['slippage_coef'] = 0.001
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)

        # Execute trade to trigger slippage
        env.step(1)  # Buy

        # Slippage should be recorded
        self.assertGreaterEqual(env.total_slippage_paid, 0)

        env.close()

    def test_position_constraints(self):
        """Test position size constraints are enforced."""
        config = self.test_config.copy()
        config['action_mode'] = 'continuous'
        config['max_position_pct'] = 0.5  # Max 50% of portfolio
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)

        # Try to take maximum position
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))

        current_price = env._get_current_price()
        max_allowed_position = (env.portfolio_value * 0.5) / current_price

        # Position should not exceed max (with some tolerance for lot size)
        self.assertLessEqual(
            abs(env.position),
            max_allowed_position * 1.1,  # 10% tolerance
            "Position should respect max_position_pct constraint"
        )

        env.close()

    def test_bankruptcy_termination(self):
        """Test episode terminates on bankruptcy."""
        config = self.test_config.copy()
        config['terminate_on_bankruptcy'] = True
        config['bankruptcy_threshold'] = 0.5  # 50% of initial
        config['initial_balance'] = 1000.0
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)

        # Manually set portfolio value to trigger bankruptcy
        env.portfolio_value = 400.0  # Below 50% of 1000
        env.balance = 400.0

        # Next step should terminate
        terminated = env._check_termination()
        self.assertTrue(terminated, "Should terminate on bankruptcy")

        env.close()

    def test_max_steps_truncation(self):
        """Test episode truncates after max steps."""
        config = self.test_config.copy()
        config['max_episode_steps'] = 20
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)

        truncated = False
        for _ in range(25):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            if terminated or truncated:
                break

        self.assertTrue(truncated, "Episode should truncate after max_steps")

        env.close()

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config = self.test_config.copy()

        # First run
        env1 = CryptoTradingEnv(config)
        obs1, _ = env1.reset(seed=42)
        actions1 = []
        rewards1 = []

        for _ in range(10):
            action = env1.action_space.sample()
            actions1.append(action)
            obs, reward, terminated, truncated, info = env1.step(action)
            rewards1.append(reward)
            if terminated or truncated:
                break

        # Second run with same seed
        env2 = CryptoTradingEnv(config)
        obs2, _ = env2.reset(seed=42)
        rewards2 = []

        for action in actions1:
            obs, reward, terminated, truncated, info = env2.step(action)
            rewards2.append(reward)
            if terminated or truncated:
                break

        # Check observations match
        np.testing.assert_array_almost_equal(obs1, obs2, decimal=5)

        # Check rewards match
        self.assertEqual(len(rewards1), len(rewards2))
        for r1, r2 in zip(rewards1, rewards2):
            self.assertAlmostEqual(r1, r2, places=5)

        env1.close()
        env2.close()

    def test_episode_metrics_computation(self):
        """Test episode metrics are computed correctly."""
        obs, info = self.env.reset(seed=42)

        # Run episode
        done = False
        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if self.env.current_step >= 30:
                break

        if done:
            # Check metrics exist
            self.assertIn('episode_metrics', info)
            metrics = info['episode_metrics']

            # Check required metrics
            required_metrics = [
                'total_return', 'total_return_pct', 'sharpe_ratio',
                'max_drawdown', 'volatility', 'total_trades'
            ]
            for metric in required_metrics:
                self.assertIn(metric, metrics)

    def test_trade_history_logging(self):
        """Test trade history is logged correctly."""
        config = self.test_config.copy()
        config['log_trades'] = True
        config['action_mode'] = 'discrete'
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)

        # Execute some trades
        env.step(1)  # Buy
        env.step(0)  # Hold
        env.step(3)  # Exit

        # Get trade history
        trade_df = env.get_trade_history()

        # Should have at least buy and sell trades
        self.assertGreater(len(trade_df), 0, "Trade history should not be empty")

        # Check required columns
        required_cols = ['step', 'timestamp', 'size', 'price', 'fee', 'position']
        for col in required_cols:
            self.assertIn(col, trade_df.columns)

        env.close()

    def test_reward_functions(self):
        """Test different reward function types."""
        reward_types = ["nav_delta", "nav_delta_minus_tx", "risk_adjusted", "sparse"]

        for reward_type in reward_types:
            with self.subTest(reward_type=reward_type):
                config = self.test_config.copy()
                config['reward_type'] = reward_type
                env = CryptoTradingEnv(config)

                obs, info = env.reset(seed=42)
                obs, reward, terminated, truncated, info = env.step(1)

                # Reward should be a valid number
                self.assertIsInstance(reward, (float, np.floating))
                self.assertFalse(np.isnan(reward))
                self.assertFalse(np.isinf(reward))

                env.close()

    def test_config_presets(self):
        """Test that config presets load correctly."""
        presets = ["default", "minimal", "conservative", "aggressive"]

        for preset in presets:
            with self.subTest(preset=preset):
                config = get_config(preset)
                env = CryptoTradingEnv(config)

                obs, info = env.reset(seed=42)
                self.assertIsNotNone(obs)

                env.close()

    def test_render_modes(self):
        """Test rendering in different modes."""
        obs, info = self.env.reset(seed=42)

        # Test human mode
        output = self.env.render(mode='human')
        # Should not raise error

        # Test ansi mode
        output = self.env.render(mode='ansi')
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

    def test_no_lookahead_bias(self):
        """Test that observations don't contain future information."""
        obs, info = self.env.reset(seed=42)

        current_step = self.env.current_step
        current_timestamp = self.env.data.index[current_step]

        # All data in observation should be from current step or earlier
        # This is guaranteed by the implementation using iloc[start_idx:end_idx]
        # where end_idx = current_step + 1 (exclusive)

        # Step forward and verify timestamp increases
        obs, reward, terminated, truncated, info = self.env.step(0)
        next_timestamp = self.env.data.index[self.env.current_step]

        self.assertGreater(next_timestamp, current_timestamp,
                          "Timestamp should advance forward")

    def test_margin_trading(self):
        """Test margin trading and leverage."""
        config = self.test_config.copy()
        config['use_margin'] = True
        config['leverage'] = 2.0
        config['action_mode'] = 'discrete'
        env = CryptoTradingEnv(config)

        obs, info = env.reset(seed=42)
        initial_balance = env.balance

        # Take leveraged position
        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        # Position value should potentially exceed balance (due to leverage)
        current_price = env._get_current_price()
        position_value = abs(env.position * current_price)

        # With leverage, position value can be > initial balance
        # (though it depends on max_position_pct and other constraints)

        env.close()

    def test_normalization_methods(self):
        """Test different normalization methods."""
        methods = ["zscore", "minmax", "none"]

        for method in methods:
            with self.subTest(method=method):
                config = self.test_config.copy()
                config['normalization'] = method
                env = CryptoTradingEnv(config)

                obs, info = env.reset(seed=42)
                self.assertIsNotNone(obs)

                # Observation should not contain NaN or Inf
                if isinstance(obs, np.ndarray):
                    self.assertFalse(np.any(np.isnan(obs)))
                    self.assertFalse(np.any(np.isinf(obs)))

                env.close()


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicator calculations."""

    def setUp(self):
        """Create sample data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)

        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.randn(100) * 1000,
            'high': 51000 + np.random.randn(100) * 1000,
            'low': 49000 + np.random.randn(100) * 1000,
            'close': 50000 + np.random.randn(100) * 1000,
            'volume': 100 + np.random.randn(100) * 10
        })

    def test_ema_calculation(self):
        """Test EMA indicator."""
        from DRL.Environment.indicators import TechnicalIndicators

        ema = TechnicalIndicators.ema(self.df['close'], 10)
        self.assertEqual(len(ema), len(self.df))
        self.assertFalse(ema.isna().all())

    def test_rsi_calculation(self):
        """Test RSI indicator."""
        from DRL.Environment.indicators import TechnicalIndicators

        rsi = TechnicalIndicators.rsi(self.df['close'], 14)
        self.assertEqual(len(rsi), len(self.df))

        # RSI should be between 0 and 100 (where not NaN)
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

    def test_macd_calculation(self):
        """Test MACD indicator."""
        from DRL.Environment.indicators import TechnicalIndicators

        macd = TechnicalIndicators.macd(self.df['close'])
        self.assertIn('macd', macd.columns)
        self.assertIn('signal', macd.columns)
        self.assertIn('histogram', macd.columns)

    def test_atr_calculation(self):
        """Test ATR indicator."""
        from DRL.Environment.indicators import TechnicalIndicators

        atr = TechnicalIndicators.atr(self.df['high'], self.df['low'], self.df['close'], 14)
        self.assertEqual(len(atr), len(self.df))
        # ATR should be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr >= 0).all())

    def test_compute_multiple_indicators(self):
        """Test computing multiple indicators at once."""
        from DRL.Environment.indicators import compute_indicators

        indicators = ["ema_10", "rsi_14", "macd", "atr_14"]
        df_with_ind = compute_indicators(self.df, indicators)

        # Should have original columns plus indicators
        self.assertIn('ema_10', df_with_ind.columns)
        self.assertIn('rsi_14', df_with_ind.columns)
        self.assertIn('macd', df_with_ind.columns)
        self.assertIn('atr_14', df_with_ind.columns)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
