import pandas as pd
from env import CryptoTradingEnv
from stable_baselines3 import PPO
import os

# Create directory to save models
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# --- 1. Load and preprocess data ---
try:
    df = pd.read_csv('data/btc_usd_1h.csv', index_col='Datetime', parse_dates=True)
except Exception as e:
    # Handle the multi-level column issue from yfinance if it exists
    df = pd.read_csv('data/btc_usd_1h.csv', header=[0, 1], index_col=0, parse_dates=True)
    # Flatten the multi-level columns
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    # Keep only the relevant columns
    df = df[['Open_BTC-USD', 'High_BTC-USD', 'Low_BTC-USD', 'Close_BTC-USD', 'Volume_BTC-USD']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']


# Data cleaning
df.ffill(inplace=True) # Forward fill missing values
df.dropna(inplace=True) # Drop any remaining NaNs

# For faster training, let's use a smaller portion of the data
# df = df.tail(5000)

print("Data loaded and preprocessed successfully.")
print(df.head())
print(f"\nData shape: {df.shape}")

# --- 2. Create the trading environment ---
env = CryptoTradingEnv(df)
# It's a good practice to wrap the environment for monitoring
# from stable_baselines3.common.monitor import Monitor
# env = Monitor(env, filename=f"{logdir}/monitor.csv")

# --- 3. Instantiate the PPO agent ---
# MlpPolicy is used for environments with flat feature vectors
model = PPO('MlpPolicy', env, verbose=1)

# --- 4. Train the agent ---
# Using a smaller number of timesteps for a quick demonstration
TIMESTEPS = 20000
print(f"\nStarting training for {TIMESTEPS} timesteps...")

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

print("\nTraining finished.")

# --- 5. Save the trained model ---
model_path = os.path.join(models_dir, f"ppo_crypto_trader_{TIMESTEPS}.zip")
model.save(model_path)

print(f"\nModel saved to {model_path}")


# --- 6. Evaluate the agent ---
import matplotlib.pyplot as plt

print("\n--- Starting Evaluation ---")

# Split data into training and testing sets
split_index = int(len(df) * 0.8)
train_df = df[:split_index]
test_df = df[split_index:]

# Re-initialize the environment with the full training data for a fair final training run
train_env = CryptoTradingEnv(train_df)
model = PPO('MlpPolicy', train_env, verbose=0) # verbose=0 for cleaner output
model.learn(total_timesteps=TIMESTEPS)
print("Model re-trained on the training split.")

# Create a fresh environment with the test data
eval_env = CryptoTradingEnv(test_df)
obs, info = eval_env.reset()

# deterministic=True ensures the agent always chooses the best action
# instead of a stochastic one, which is better for evaluation.
agent_net_worths = [eval_env.initial_balance]
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    agent_net_worths.append(info['net_worth'])

print("Evaluation finished.")

# --- 7. "Buy and Hold" Benchmark ---
buy_hold_net_worths = []
initial_price = test_df.iloc[0]['Close']
crypto_bought = eval_env.initial_balance / initial_price
for i in range(len(test_df)):
    current_price = test_df.iloc[i]['Close']
    buy_hold_net_worths.append(crypto_bought * current_price)

# --- 8. Visualize the results ---
plt.figure(figsize=(15, 8))
plt.plot(agent_net_worths, label='DRL Agent', color='blue')
plt.plot(buy_hold_net_worths, label='Buy and Hold', color='orange')
plt.title('DRL Agent Performance vs. Buy and Hold')
plt.xlabel('Timesteps in Test Period')
plt.ylabel('Net Worth (USD)')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = "evaluation_results.png"
plt.savefig(plot_path)
print(f"\nEvaluation plot saved to {plot_path}")
