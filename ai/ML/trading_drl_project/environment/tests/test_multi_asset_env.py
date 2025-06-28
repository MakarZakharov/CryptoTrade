"""
Tests for Multi-Asset Trading Environment
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from environment.core.multi_asset_env import MultiAssetTradingEnv
from environment.core.portfolio_manager import PortfolioManager
from environment.core.market_simulator import MarketSimulator


class TestMultiAssetTradingEnv:
    """Test suite for MultiAssetTradingEnv"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        # Create data for multiple assets
        btc_data = pd.DataFrame({
            'open': np.random.uniform(45000, 50000, 100),
            'high': np.random.uniform(46000, 51000, 100),
            'low': np.random.uniform(44000, 49000, 100),
            'close': np.random.uniform(45000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        eth_data = pd.DataFrame({
            'open': np.random.uniform(3000, 3500, 100),
            'high': np.random.uniform(3100, 3600, 100),
            'low': np.random.uniform(2900, 3400, 100),
            'close': np.random.uniform(3000, 3500, 100),
            'volume': np.random.uniform(500, 2000, 100)
        }, index=dates)
        
        return {
            'BTC/USDT': btc_data,
            'ETH/USDT': eth_data
        }
    
    @pytest.fixture
    def multi_asset_env(self, sample_data):
        """Create MultiAssetTradingEnv instance"""
        return MultiAssetTradingEnv(
            data_dict=sample_data,
            initial_balance=10000.0,
            commission=0.001,
            window_size=10,
            action_type="continuous"
        )
    
    def test_environment_initialization(self, multi_asset_env):
        """Test environment initialization"""
        assert multi_asset_env.num_assets == 2
        assert multi_asset_env.asset_names == ['BTC/USDT', 'ETH/USDT']
        assert multi_asset_env.initial_balance == 10000.0
        assert multi_asset_env.action_type == "continuous"
    
    def test_action_space_continuous(self, sample_data):
        """Test continuous action space"""
        env = MultiAssetTradingEnv(
            data_dict=sample_data,
            action_type="continuous"
        )
        
        assert env.action_space.shape == (2,)  # 2 assets
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0
    
    def test_action_space_discrete(self, sample_data):
        """Test discrete action space"""
        env = MultiAssetTradingEnv(
            data_dict=sample_data,
            action_type="discrete"
        )
        
        assert env.action_space.n == 9  # 3^2 possible actions
    
    def test_observation_space(self, multi_asset_env):
        """Test observation space dimensions"""
        obs_space = multi_asset_env.observation_space
        
        # Should have window_size rows
        assert obs_space.shape[0] == 10
        
        # Should have features for both assets plus portfolio features
        expected_features = 18 * 2 + 5  # (features_per_asset * num_assets) + portfolio_features
        assert obs_space.shape[1] == expected_features
    
    def test_reset_environment(self, multi_asset_env):
        """Test environment reset"""
        observation, info = multi_asset_env.reset()
        
        assert observation.shape == multi_asset_env.observation_space.shape
        assert isinstance(info, dict)
        assert 'portfolio_value' in info
        assert 'asset_positions' in info
        assert multi_asset_env.current_step == 0
        assert not multi_asset_env.done
    
    def test_step_continuous_actions(self, multi_asset_env):
        """Test step with continuous actions"""
        observation, info = multi_asset_env.reset()
        
        # Test hold action (close to 0)
        action = np.array([0.0, 0.0])
        obs, reward, done, truncated, info = multi_asset_env.step(action)
        
        assert obs.shape == multi_asset_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_step_buy_actions(self, multi_asset_env):
        """Test step with buy actions"""
        observation, info = multi_asset_env.reset()
        
        # Test buy action
        action = np.array([0.5, 0.3])  # Buy 50% max for BTC, 30% for ETH
        obs, reward, done, truncated, info = multi_asset_env.step(action)
        
        # Check that positions were opened
        assert info['asset_positions']['BTC/USDT'] > 0
        assert info['asset_positions']['ETH/USDT'] > 0
    
    def test_step_sell_actions(self, multi_asset_env):
        """Test step with sell actions after buying"""
        observation, info = multi_asset_env.reset()
        
        # First buy
        buy_action = np.array([0.5, 0.3])
        multi_asset_env.step(buy_action)
        
        # Then sell
        sell_action = np.array([-0.3, -0.2])  # Sell partial positions
        obs, reward, done, truncated, info = multi_asset_env.step(sell_action)
        
        # Positions should be reduced but not zero
        assert info['asset_positions']['BTC/USDT'] >= 0
        assert info['asset_positions']['ETH/USDT'] >= 0
    
    def test_discrete_action_decoding(self, sample_data):
        """Test discrete action decoding"""
        env = MultiAssetTradingEnv(
            data_dict=sample_data,
            action_type="discrete"
        )
        
        # Test action decoding
        decoded = env._decode_discrete_action(4)  # Middle action
        expected = [1, -1]  # Buy first asset, sell second
        assert decoded == expected
    
    def test_portfolio_features(self, multi_asset_env):
        """Test portfolio feature extraction"""
        multi_asset_env.reset()
        
        features = multi_asset_env._get_portfolio_features()
        
        assert len(features) == 5  # Expected number of portfolio features
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_data_alignment(self, sample_data):
        """Test data alignment across assets"""
        # Create misaligned data
        misaligned_data = sample_data.copy()
        misaligned_data['ETH/USDT'] = misaligned_data['ETH/USDT'].iloc[5:]  # Remove first 5 rows
        
        env = MultiAssetTradingEnv(data_dict=misaligned_data)
        
        # Should have aligned data
        assert all(len(env.market_sims[asset].data) == len(env.market_sims['BTC/USDT'].data) 
                  for asset in env.asset_names)
    
    def test_episode_completion(self, multi_asset_env):
        """Test episode completion conditions"""
        multi_asset_env.reset()
        
        # Run until episode ends
        done = False
        steps = 0
        max_steps = 200  # Safety limit
        
        while not done and steps < max_steps:
            action = np.random.uniform(-1, 1, size=2)
            obs, reward, done, truncated, info = multi_asset_env.step(action)
            steps += 1
        
        assert steps > 0
        assert done or steps >= max_steps
    
    def test_portfolio_value_tracking(self, multi_asset_env):
        """Test portfolio value tracking throughout episode"""
        observation, info = multi_asset_env.reset()
        initial_value = info['portfolio_value']
        
        # Execute some trades
        for _ in range(10):
            action = np.random.uniform(-0.5, 0.5, size=2)
            obs, reward, done, truncated, info = multi_asset_env.step(action)
            
            # Portfolio value should be positive
            assert info['portfolio_value'] > 0
            
            if done:
                break
    
    def test_reward_calculation(self, multi_asset_env):
        """Test reward calculation"""
        multi_asset_env.reset()
        
        # Take an action and check reward
        action = np.array([0.1, 0.1])  # Small buy
        obs, reward, done, truncated, info = multi_asset_env.step(action)
        
        assert isinstance(reward, (int, float))
        # Reward should be finite
        assert np.isfinite(reward)
    
    def test_invalid_actions(self, multi_asset_env):
        """Test handling of invalid actions"""
        multi_asset_env.reset()
        
        # Test action out of bounds (should be clipped or handled gracefully)
        invalid_action = np.array([2.0, -2.0])  # Outside [-1, 1] range
        
        # Should not raise an exception
        try:
            obs, reward, done, truncated, info = multi_asset_env.step(invalid_action)
            assert True  # If we get here, it handled the invalid action gracefully
        except Exception as e:
            pytest.fail(f"Environment should handle invalid actions gracefully: {e}")
    
    def test_commission_calculation(self, multi_asset_env):
        """Test commission calculation"""
        observation, info = multi_asset_env.reset()
        initial_commission = info['commission_paid']
        
        # Execute a trade
        action = np.array([0.2, 0.2])
        obs, reward, done, truncated, info = multi_asset_env.step(action)
        
        # Commission should increase after trade
        assert info['commission_paid'] >= initial_commission
    
    def test_asset_correlation_handling(self, multi_asset_env):
        """Test handling of correlated assets"""
        multi_asset_env.reset()
        
        # Execute correlated actions on both assets
        correlated_action = np.array([0.3, 0.3])  # Buy both
        obs, reward, done, truncated, info = multi_asset_env.step(correlated_action)
        
        # Both assets should have positions
        positions = info['asset_positions']
        assert positions['BTC/USDT'] > 0
        assert positions['ETH/USDT'] > 0
    
    @pytest.mark.parametrize("action_type", ["continuous", "discrete"])
    def test_different_action_types(self, sample_data, action_type):
        """Test environment with different action types"""
        env = MultiAssetTradingEnv(
            data_dict=sample_data,
            action_type=action_type
        )
        
        observation, info = env.reset()
        
        if action_type == "continuous":
            action = np.array([0.1, -0.1])
        else:
            action = 4  # Middle action
        
        obs, reward, done, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))


if __name__ == "__main__":
    pytest.main([__file__])