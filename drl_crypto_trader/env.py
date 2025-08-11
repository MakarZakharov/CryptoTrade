import gymnasium as gym
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    """
    A custom environment for reinforcement learning-based cryptocurrency trading.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=24, initial_balance=100000, fee_rate=0.001):
        super(CryptoTradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space
        # Observation consists of the market data for the past `window_size` steps,
        # plus the current balance and crypto holdings.
        # Shape: (window_size, num_features) + 2 (balance, crypto_held)
        # We flatten this to a 1D array.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(window_size * self.df.shape[1] + 2,),
            dtype=np.float64
        )

        # Initialize state
        self.balance = 0
        self.crypto_held = 0
        self.current_step = 0
        self.total_steps = len(self.df) - self.window_size
        self.net_worth = 0

    def _get_obs(self):
        """
        Get the observation for the current step.
        """
        # Get the market data for the window
        window = self.df.iloc[self.current_step:self.current_step + self.window_size].values

        # Flatten the window and append balance and crypto holdings
        obs = np.concatenate([window.flatten(), [self.balance, self.crypto_held]])
        return obs

    def _get_info(self):
        """
        Get auxiliary information for the current step.
        """
        # The price used for info should be the closing price of the current window
        end_of_window_step = self.current_step + self.window_size - 1
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'current_price': self.df.iloc[end_of_window_step]['Close']
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        # Start at a random point in the dataset to improve generalization
        self.current_step = np.random.randint(0, self.total_steps)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        current_price = self.df.iloc[self.current_step + self.window_size]['Close']

        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                # Buy with all available balance
                amount_to_buy = self.balance / current_price
                fee = amount_to_buy * self.fee_rate
                self.crypto_held += amount_to_buy - fee
                self.balance = 0
        elif action == 2:  # Sell
            if self.crypto_held > 0:
                # Sell all held crypto
                amount_to_sell = self.crypto_held * current_price
                fee = amount_to_sell * self.fee_rate
                self.balance += amount_to_sell - fee
                self.crypto_held = 0

        # Update net worth
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        # Calculate reward
        reward = self.net_worth - prev_net_worth

        # Move to the next step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= self.total_steps

        observation = self._get_obs()
        info = self._get_info()

        # Gymnasium API expects a 'truncated' flag as well
        truncated = False

        return observation, reward, done, truncated, info

    def render(self, mode='human'):
        info = self._get_info()
        print(f"Step: {self.current_step} | Net Worth: {info['net_worth']:.2f} | "
              f"Balance: {info['balance']:.2f} | Crypto Held: {info['crypto_held']:.6f} | "
              f"Price: {info['current_price']:.2f}")
