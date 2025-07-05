"""
Скрипт для оценки STAS_ML-агента.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.STAS_ML.config.trading_config import TradingConfig
from CryptoTrade.ai.STAS_ML.environment.trading_env import TradingEnv
from CryptoTrade.ai.STAS_ML.agents.dqn_agent import DQNAgent
from CryptoTrade.ai.STAS_ML.agents.ppo_agent import PPOAgent
from CryptoTrade.ai.STAS_ML.utils.metrics import MetricsCalculator, TradingMetrics, calculate_metrics


class DRLEvaluator:
    """Класс для оценки обученных STAS_ML агентов."""
    
    def __init__(self, model_path: str, config: TradingConfig, agent_type: str = "PPO"):
        self.model_path = model_path
        self.config = config
        self.agent_type = agent_type
        self.agent = None
        self.results = {}
        
    def load_agent(self):
        """Загрузить обученного агента."""
        env = TradingEnv(self.config)
        
        if self.agent_type.upper() == "DQN":
            self.agent = DQNAgent(self.config)
        elif self.agent_type.upper() == "PPO":
            self.agent = PPOAgent(self.config)
        else:
            raise ValueError(f"Неподдерживаемый тип агента: {self.agent_type}")
        
        # Загружаем модель
        self.agent.load(self.model_path, env)
        print(f"✅ Агент {self.agent_type} загружен из {self.model_path}")
        return self.agent
    
    def evaluate_episodes(self, env: TradingEnv, num_episodes: int = 10, 
                         deterministic: bool = True) -> Dict:
        """Оценить агента на нескольких эпизодах."""
        if not self.agent:
            self.load_agent()
        
        episode_results = []
        all_actions = []
        
        print(f"🔄 Запуск оценки на {num_episodes} эпизодах...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()  # Gymnasium API returns (obs, info)
            episode_reward = 0
            episode_actions = []
            episode_steps = 0
            
            while True:
                action = self.agent.act(obs)
                all_actions.append(action[0])
                episode_actions.append(action[0])
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # Сохраняем результаты эпизода
            episode_result = {
                'episode': episode,
                'total_reward': episode_reward,
                'total_return': info.get('total_return', 0),
                'total_return_pct': info.get('total_return_pct', 0),
                'total_return_usd': info.get('total_return_usd', 0),
                'max_drawdown': info.get('max_drawdown', 0),
                'max_drawdown_pct': info.get('max_drawdown_pct', 0),
                'max_drawdown_usd': info.get('max_drawdown_usd', 0),
                'win_rate': info.get('win_rate', 0),
                'win_rate_pct': info.get('win_rate_pct', 0),
                'total_trades': info.get('total_trades', 0),
                'final_portfolio': info.get('portfolio_value', 0),
                'initial_balance': info.get('initial_balance', 10000),
                'steps': episode_steps,
                'sharpe_ratio': info.get('sharpe_ratio', 0),
                'profit_factor': info.get('profit_factor', 0)
            }
            episode_results.append(episode_result)
            
            print(f"  Эпизод {episode+1}: "
                  f"${episode_result['initial_balance']:,.0f} → ${episode_result['final_portfolio']:,.0f} "
                  f"({episode_result['total_return_pct']:.1f}%), "
                  f"просадка={episode_result['max_drawdown_pct']:.1f}%, "
                  f"винрейт={episode_result['win_rate_pct']:.1f}%, "
                  f"сделок={episode_result['total_trades']}")
        
        # Агрегированные результаты
        results = {
            'episodes': episode_results,
            'mean_reward': np.mean([ep['total_reward'] for ep in episode_results]),
            'mean_return': np.mean([ep['total_return'] for ep in episode_results]),
            'mean_return_pct': np.mean([ep['total_return_pct'] for ep in episode_results]),
            'mean_return_usd': np.mean([ep['total_return_usd'] for ep in episode_results]),
            'mean_drawdown': np.mean([ep['max_drawdown'] for ep in episode_results]),
            'mean_drawdown_pct': np.mean([ep['max_drawdown_pct'] for ep in episode_results]),
            'mean_drawdown_usd': np.mean([ep['max_drawdown_usd'] for ep in episode_results]),
            'mean_win_rate': np.mean([ep['win_rate'] for ep in episode_results]),
            'mean_win_rate_pct': np.mean([ep['win_rate_pct'] for ep in episode_results]),
            'mean_trades': np.mean([ep['total_trades'] for ep in episode_results]),
            'mean_final_portfolio': np.mean([ep['final_portfolio'] for ep in episode_results]),
            'std_return': np.std([ep['total_return'] for ep in episode_results]),
            'std_return_pct': np.std([ep['total_return_pct'] for ep in episode_results]),
            'sharpe_ratio': self._calculate_sharpe_ratio(episode_results),
            'mean_sharpe_ratio': np.mean([ep.get('sharpe_ratio', 0) for ep in episode_results]),
            'mean_profit_factor': np.mean([ep.get('profit_factor', 0) for ep in episode_results]),
            'win_rate_episodes': sum(1 for ep in episode_results if ep['total_return'] > 0) / num_episodes,
            'profitable_episodes_pct': sum(1 for ep in episode_results if ep['total_return_pct'] > 0) / num_episodes * 100,
            'all_actions': all_actions,
            'best_episode': max(episode_results, key=lambda x: x['total_return_pct']),
            'worst_episode': min(episode_results, key=lambda x: x['total_return_pct'])
        }
        
        self.results = results
        
        # Печатаем итоговую сводку
        self._print_evaluation_summary()
        
        return results
    
    def _print_evaluation_summary(self):
        """Печать итоговой сводки результатов оценки."""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
        print("="*80)
        
        # Основные метрики
        print(f"💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   Средний итоговый баланс:     ${self.results['mean_final_portfolio']:,.0f}")
        print(f"   Средняя доходность:          {self.results['mean_return_pct']:+.1f}% (${self.results['mean_return_usd']:+,.0f})")
        print(f"   Стандартное отклонение:      ±{self.results['std_return_pct']:.1f}%")
        print(f"   Максимальная просадка:       {self.results['mean_drawdown_pct']:.1f}% (${self.results['mean_drawdown_usd']:,.0f})")
        
        print(f"\n📊 ТОРГОВЫЕ МЕТРИКИ:")
        print(f"   Средний винрейт:             {self.results['mean_win_rate_pct']:.1f}%")
        print(f"   Среднее кол-во сделок:       {self.results['mean_trades']:.0f}")
        print(f"   Прибыльных эпизодов:         {self.results['profitable_episodes_pct']:.1f}%")
        print(f"   Средний коэф. Шарпа:         {self.results['mean_sharpe_ratio']:.3f}")
        print(f"   Средний фактор прибыли:      {self.results['mean_profit_factor']:.2f}")
        
        print(f"\n🏆 ЛУЧШИЙ/ХУДШИЙ ЭПИЗОД:")
        best = self.results['best_episode']
        worst = self.results['worst_episode']
        print(f"   Лучший:  {best['total_return_pct']:+.1f}% (эпизод {best['episode'] + 1})")
        print(f"   Худший:  {worst['total_return_pct']:+.1f}% (эпизод {worst['episode'] + 1})")
        
        print(f"\n🎯 АНАЛИЗ ДЕЙСТВИЙ:")
        actions = self.results['all_actions']
        buy_pct = sum(1 for a in actions if a > 0.1) / len(actions) * 100
        sell_pct = sum(1 for a in actions if a < -0.1) / len(actions) * 100
        hold_pct = 100 - buy_pct - sell_pct
        print(f"   Покупка:    {buy_pct:.1f}%")
        print(f"   Продажа:    {sell_pct:.1f}%") 
        print(f"   Ожидание:   {hold_pct:.1f}%")
        
        print("="*80)
    
    def _calculate_sharpe_ratio(self, episode_results: List[Dict]) -> float:
        """Рассчитать коэффициент Шарпа по эпизодам."""
        returns = [ep['total_return'] for ep in episode_results]
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def create_detailed_report(self, save_path: Optional[str] = None) -> Dict:
        """Создать детальный отчет об оценке."""
        if not self.results:
            raise ValueError("Сначала запустите evaluate_episodes()")
        
        report = {
            'model_info': {
                'model_path': self.model_path,
                'agent_type': self.agent_type,
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'reward_scheme': self.config.reward_scheme,
                'evaluation_date': datetime.now().isoformat()
            },
            'performance_metrics': {
                'mean_return': self.results['mean_return'],
                'mean_return_pct': self.results['mean_return_pct'],
                'mean_return_usd': self.results['mean_return_usd'],
                'std_return': self.results['std_return'],
                'std_return_pct': self.results['std_return_pct'],
                'sharpe_ratio': self.results['sharpe_ratio'],
                'mean_sharpe_ratio': self.results['mean_sharpe_ratio'],
                'mean_drawdown': self.results['mean_drawdown'],
                'mean_drawdown_pct': self.results['mean_drawdown_pct'],
                'mean_drawdown_usd': self.results['mean_drawdown_usd'],
                'mean_win_rate': self.results['mean_win_rate'],
                'mean_win_rate_pct': self.results['mean_win_rate_pct'],
                'mean_profit_factor': self.results['mean_profit_factor'],
                'win_rate_episodes': self.results['win_rate_episodes'],
                'profitable_episodes_pct': self.results['profitable_episodes_pct'],
                'mean_trades_per_episode': self.results['mean_trades'],
                'mean_final_portfolio': self.results['mean_final_portfolio']
            },
            'action_analysis': {
                'mean_action': np.mean(self.results['all_actions']),
                'std_action': np.std(self.results['all_actions']),
                'action_range': [np.min(self.results['all_actions']), np.max(self.results['all_actions'])],
                'buy_actions_pct': sum(1 for a in self.results['all_actions'] if a > 0.1) / len(self.results['all_actions']),
                'sell_actions_pct': sum(1 for a in self.results['all_actions'] if a < -0.1) / len(self.results['all_actions']),
                'hold_actions_pct': sum(1 for a in self.results['all_actions'] if abs(a) <= 0.1) / len(self.results['all_actions'])
            }
        }
        
        if save_path:
            # Сохраняем отчет
            import json
            with open(f"{save_path}/evaluation_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            # Сохраняем результаты эпизодов
            episodes_df = pd.DataFrame(self.results['episodes'])
            episodes_df.to_csv(f"{save_path}/episode_results.csv", index=False)
            
            print(f"📊 Отчет сохранен в {save_path}")
        
        return report
    
    def plot_results(self, save_path: Optional[str] = None):
        """Создать графики результатов."""
        if not self.results:
            raise ValueError("Сначала запустите evaluate_episodes()")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График доходности по эпизодам
        episodes = [ep['episode'] for ep in self.results['episodes']]
        returns = [ep['total_return'] * 100 for ep in self.results['episodes']]
        
        axes[0, 0].plot(episodes, returns, 'b-o')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Доходность по эпизодам (%)')
        axes[0, 0].set_xlabel('Эпизод')
        axes[0, 0].set_ylabel('Доходность (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # График просадок
        drawdowns = [ep['max_drawdown'] * 100 for ep in self.results['episodes']]
        axes[0, 1].plot(episodes, drawdowns, 'r-o')
        axes[0, 1].set_title('Максимальная просадка по эпизодам (%)')
        axes[0, 1].set_xlabel('Эпизод')
        axes[0, 1].set_ylabel('Просадка (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Гистограмма действий
        actions = self.results['all_actions']
        axes[1, 0].hist(actions, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Распределение действий агента')
        axes[1, 0].set_xlabel('Действие (от -1 до 1)')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].grid(True, alpha=0.3)
        
        # График количества сделок
        trades = [ep['total_trades'] for ep in self.results['episodes']]
        axes[1, 1].plot(episodes, trades, 'g-o')
        axes[1, 1].set_title('Количество сделок по эпизодам')
        axes[1, 1].set_xlabel('Эпизод')
        axes[1, 1].set_ylabel('Количество сделок')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/evaluation_plots.png", dpi=300, bbox_inches='tight')
            print(f"📈 Графики сохранены в {save_path}/evaluation_plots.png")
        
        plt.show()
    
    def compare_with_baseline(self, baseline_strategy: str = "buy_hold") -> Dict:
        """Сравнить с базовой стратегией."""
        if not self.results:
            raise ValueError("Сначала запустите evaluate_episodes()")
        
        # Создаем среду для базовой стратегии
        env = TradingEnv(self.config)
        obs = env.reset()
        
        if baseline_strategy == "buy_hold":
            # Стратегия buy and hold
            action = np.array([1.0])  # Покупаем на весь капитал в начале
            obs, reward, done, info = env.step(action)
            
            while not done:
                action = np.array([0.0])  # Держим
                obs, reward, done, info = env.step(action)
            
            baseline_return = info.get('total_return', 0)
            baseline_drawdown = info.get('max_drawdown', 0)
            
        else:
            baseline_return = 0
            baseline_drawdown = 0
        
        comparison = {
            'agent_return': self.results['mean_return'],
            'baseline_return': baseline_return,
            'outperformance': self.results['mean_return'] - baseline_return,
            'agent_drawdown': self.results['mean_drawdown'],
            'baseline_drawdown': baseline_drawdown,
            'risk_adjusted_performance': (self.results['mean_return'] - baseline_return) / max(self.results['mean_drawdown'], 0.01)
        }
        
        print(f"📊 Сравнение с {baseline_strategy}:")
        print(f"  Агент: {comparison['agent_return']:.2%}")
        print(f"  Базовая стратегия: {comparison['baseline_return']:.2%}")
        print(f"  Превышение: {comparison['outperformance']:.2%}")
        
        return comparison


def quick_evaluate(model_path: str, symbol: str = "BTCUSDT", timeframe: str = "1d",
                  agent_type: str = "PPO", episodes: int = 10):
    """Быстрая оценка модели."""
    config = TradingConfig(
        symbol=symbol,
        timeframe=timeframe,
        reward_scheme='optimized',
        initial_balance=10000.0
    )
    
    evaluator = DRLEvaluator(model_path, config, agent_type)
    
    # Создаем тестовую среду
    env = TradingEnv(config)
    
    # Оценка
    results = evaluator.evaluate_episodes(env, episodes)
    
    # Создаем отчет
    report = evaluator.create_detailed_report()
    
    # Показываем графики
    evaluator.plot_results()
    
    return evaluator, results, report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка STAS_ML агента')
    parser.add_argument('model_path', help='Путь к модели')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--agent', default='PPO', choices=['PPO', 'DQN'], help='Тип агента')
    parser.add_argument('--episodes', type=int, default=10, help='Количество эпизодов для оценки')
    
    args = parser.parse_args()
    
    quick_evaluate(
        model_path=args.model_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        agent_type=args.agent,
        episodes=args.episodes
    ) 