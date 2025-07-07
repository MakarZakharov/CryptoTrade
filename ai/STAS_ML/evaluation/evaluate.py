"""
Модуль для оцінки навчених STAS_ML агентів.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Додаємо шлях до модулів проекту
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.STAS_ML.config.trading_config import TradingConfig
from CryptoTrade.ai.STAS_ML.environment.trading_env import TradingEnv
from CryptoTrade.ai.STAS_ML.agents.dqn_agent import DQNAgent
from CryptoTrade.ai.STAS_ML.agents.ppo_agent import PPOAgent


def quick_evaluate(model_path: str, symbol: str = "BTCUSDT", timeframe: str = "1d",
                  agent_type: str = "PPO", episodes: int = 10):
    """Швидка оцінка моделі."""
    config = TradingConfig(
        symbol=symbol,
        timeframe=timeframe,
        reward_scheme='optimized',
        initial_balance=10000.0
    )
    
    print(f"🔍 Оцінка моделі: {model_path}")
    print(f"📊 {episodes} епізодів для {symbol} {timeframe}")
    
    try:
        # Створюємо середовище
        env = TradingEnv(config)
        
        # Створюємо агента
        if agent_type.upper() == "DQN":
            agent = DQNAgent(config)
        elif agent_type.upper() == "PPO":
            agent = PPOAgent(config)
        else:
            raise ValueError(f"Непідтримуваний тип агента: {agent_type}")
        
        # Завантажуємо модель
        agent.load(model_path, env)
        
        results = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            results.append({
                'episode': episode,
                'reward': episode_reward,
                'return': info.get('total_return', 0),
                'final_balance': info.get('portfolio_value', 0),
                'steps': steps
            })
            
            print(f"  Епізод {episode+1}: Доходність={info.get('total_return', 0):.2%}, "
                  f"Баланс=${info.get('portfolio_value', 0):,.0f}")
        
        # Агреговані результати
        mean_return = np.mean([r['return'] for r in results])
        mean_balance = np.mean([r['final_balance'] for r in results])
        
        print(f"\n✅ Результати оцінки:")
        print(f"   Середня доходність: {mean_return:.2%}")
        print(f"   Середній баланс: ${mean_balance:,.0f}")
        
        return None, results, {'mean_return': mean_return, 'mean_balance': mean_balance}
        
    except Exception as e:
        print(f"❌ Помилка при оцінці: {e}")
        return None, [], {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Оцінка STAS_ML агента')
    parser.add_argument('model_path', help='Шлях до моделі')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торгова пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--agent', default='PPO', choices=['PPO', 'DQN'], help='Тип агента')
    parser.add_argument('--episodes', type=int, default=10, help='Кількість епізодів для оцінки')
    
    args = parser.parse_args()
    
    quick_evaluate(
        model_path=args.model_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        agent_type=args.agent,
        episodes=args.episodes
    )