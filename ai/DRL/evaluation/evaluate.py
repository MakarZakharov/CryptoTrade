"""
Скрипт для оценки DRL-агента.
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

from CryptoTrade.ai.DRL.config.trading_config import TradingConfig
from CryptoTrade.ai.DRL.environment.trading_env import TradingEnv
from CryptoTrade.ai.DRL.agents.dqn_agent import DQNAgent
from CryptoTrade.ai.DRL.agents.ppo_agent import PPOAgent
from CryptoTrade.ai.DRL.environment.reward_schemes import TradingMetrics


class DRLEvaluator:
    """Класс для оценки обученных DRL агентов."""
    
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
        
        # Проверяем существование модели и автоматически находим правильный путь
        found_model_path = self._find_model_path()
        if not found_model_path:
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        # Загружаем модель
        self.agent.load(found_model_path, env)
        model_path = found_model_path
        print(f"✅ Агент {self.agent_type} загружен из {model_path}")
        return self.agent
    
    def get_available_models(self):
        """Получить список всех доступных моделей."""
        import glob
        
        models_dir = os.path.join("CryptoTrade", "ai", "DRL", "models")
        available_models = []
        
        if not os.path.exists(models_dir):
            print(f"❌ Директория моделей не найдена: {models_dir}")
            return available_models
        
        print(f"🔍 Поиск моделей в: {models_dir}")
        
        # Ищем все директории с моделями
        for item in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, item)
            if os.path.isdir(model_dir):
                model_info = {
                    'name': item,
                    'path': model_dir,
                    'models': [],
                    'checkpoints': []
                }
                
                # Ищем основные модели
                for model_name in ['best_model.zip', 'final_model.zip']:
                    model_path = os.path.join(model_dir, model_name)
                    if os.path.exists(model_path):
                        model_info['models'].append({
                            'type': model_name.replace('.zip', ''),
                            'path': model_path,
                            'size': os.path.getsize(model_path),
                            'modified': os.path.getmtime(model_path)
                        })
                
                # Ищем checkpoints
                checkpoint_dir = os.path.join(model_dir, 'checkpoints')
                if os.path.exists(checkpoint_dir):
                    checkpoint_files = glob.glob(f"{checkpoint_dir}/*.zip")
                    for checkpoint_file in sorted(checkpoint_files):
                        model_info['checkpoints'].append({
                            'type': 'checkpoint',
                            'name': os.path.basename(checkpoint_file),
                            'path': checkpoint_file,
                            'size': os.path.getsize(checkpoint_file),
                            'modified': os.path.getmtime(checkpoint_file)
                        })
                
                if model_info['models'] or model_info['checkpoints']:
                    available_models.append(model_info)
        
        return available_models
    
    def validate_model_compatibility(self, model_path: str) -> bool:
        """Проверить совместимость модели с текущей конфигурацией."""
        try:
            # Создаем временную среду для проверки
            temp_env = TradingEnv(self.config)
            
            # Пытаемся загрузить модель
            if self.agent_type.upper() == "DQN":
                temp_agent = DQNAgent(self.config)
            elif self.agent_type.upper() == "PPO":
                temp_agent = PPOAgent(self.config)
            else:
                return False
            
            # Проверяем загрузку без создания модели
            from stable_baselines3 import PPO, DQN
            
            if self.agent_type.upper() == "PPO":
                model = PPO.load(model_path, env=None)
                # Проверяем совместимость observation space
                expected_shape = temp_env.observation_space.shape
                if hasattr(model, 'observation_space'):
                    actual_shape = model.observation_space.shape
                    return expected_shape == actual_shape
            elif self.agent_type.upper() == "DQN":
                model = DQN.load(model_path, env=None)
                expected_shape = temp_env.observation_space.shape
                if hasattr(model, 'observation_space'):
                    actual_shape = model.observation_space.shape
                    return expected_shape == actual_shape
            
            return True
            
        except Exception as e:
            print(f"⚠️ Модель несовместима: {e}")
            return False
    
    def interactive_model_selection(self):
        """Интерактивный выбор модели."""
        available_models = self.get_available_models()
        
        if not available_models:
            print("❌ Модели не найдены!")
            print("💡 Сначала обучите модель с помощью mvp_train.py")
            return None
        
        print(f"\n📋 Найдено {len(available_models)} директорий с моделями:")
        print("=" * 80)
        
        # Показываем все доступные модели
        model_options = []
        option_counter = 1
        
        for model_info in available_models:
            print(f"\n📁 {model_info['name']}:")
            
            # Основные модели (приоритет: best_model > final_model)
            sorted_models = sorted(model_info['models'], 
                                 key=lambda x: 0 if x['type'] == 'best_model' else 1)
            
            for model in sorted_models:
                size_mb = model['size'] / (1024 * 1024)
                modified_time = pd.to_datetime(model['modified'], unit='s').strftime('%Y-%m-%d %H:%M')
                
                print(f"  {option_counter}. {model['type']}.zip ({size_mb:.1f} MB, {modified_time})")
                model_options.append({
                    'index': option_counter,
                    'path': model['path'],
                    'type': model['type'],
                    'dir_name': model_info['name']
                })
                option_counter += 1
            
            # Показываем последние 3 checkpoint'а
            if model_info['checkpoints']:
                recent_checkpoints = sorted(model_info['checkpoints'], 
                                          key=lambda x: x['modified'], reverse=True)[:3]
                
                print(f"  📊 Последние checkpoints:")
                for checkpoint in recent_checkpoints:
                    size_mb = checkpoint['size'] / (1024 * 1024)
                    modified_time = pd.to_datetime(checkpoint['modified'], unit='s').strftime('%Y-%m-%d %H:%M')
                    
                    print(f"    {option_counter}. {checkpoint['name']} ({size_mb:.1f} MB, {modified_time})")
                    model_options.append({
                        'index': option_counter,
                        'path': checkpoint['path'],
                        'type': 'checkpoint',
                        'dir_name': model_info['name']
                    })
                    option_counter += 1
        
        print("\n" + "=" * 80)
        
        # Интерактивный выбор
        while True:
            try:
                choice = input(f"\nВыберите модель для оценки (1-{len(model_options)}) или 'q' для выхода: ").strip()
                
                if choice.lower() == 'q':
                    print("❌ Оценка отменена пользователем")
                    return None
                
                choice_idx = int(choice)
                if 1 <= choice_idx <= len(model_options):
                    selected_model = model_options[choice_idx - 1]
                    model_path = selected_model['path']
                    
                    print(f"\n✅ Выбрана модель: {selected_model['type']} из {selected_model['dir_name']}")
                    print(f"📁 Путь: {model_path}")
                    
                    # Проверяем совместимость
                    print("🔍 Проверка совместимости модели...")
                    if self.validate_model_compatibility(model_path):
                        print("✅ Модель совместима с текущей конфигурацией")
                        return model_path
                    else:
                        print("❌ Модель несовместима с текущей конфигурацией среды!")
                        print("💡 Возможные причины:")
                        print("  - Изменилась размерность данных")
                        print("  - Другая конфигурация технических индикаторов")
                        print("  - Несовместимые параметры среды")
                        
                        retry = input("Попробовать другую модель? (y/n): ").strip().lower()
                        if retry not in ['y', 'yes', 'да']:
                            print("🔴 Оценка остановлена - нет совместимых моделей")
                            return None
                        continue
                    
                else:
                    print(f"❌ Неверный выбор! Введите число от 1 до {len(model_options)}")
                    
            except ValueError:
                print("❌ Введите корректное число!")
            except KeyboardInterrupt:
                print("\n❌ Оценка отменена пользователем")
                return None
    
    def _find_model_path(self):
        """Найти правильный путь к модели."""
        # Если путь абсолютный и существует, используем его
        if os.path.isabs(self.model_path) and os.path.exists(self.model_path):
            print(f"🔍 Использую абсолютный путь: {self.model_path}")
            return self.model_path
        
        # Если путь относительный, ищем в CryptoTrade/ai/DRL/models/
        models_dir = os.path.join("CryptoTrade", "ai", "DRL", "models")
        
        # Проверяем прямые пути
        possible_paths = [
            self.model_path,
            f"{self.model_path}.zip",
            os.path.join(models_dir, self.model_path),
            os.path.join(models_dir, f"{self.model_path}.zip"),
            os.path.join(models_dir, self.model_path, "best_model.zip"),
            os.path.join(models_dir, self.model_path, "final_model.zip"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"🔍 Найдена модель: {path}")
                return path
        
        # Если ничего не найдено, запускаем интерактивный выбор
        print(f"🤖 Модель '{self.model_path}' не найдена автоматически")
        print("🎯 Запуск интерактивного выбора модели...")
        return self.interactive_model_selection()
    
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
                'max_drawdown': info.get('max_drawdown', 0),
                'win_rate': info.get('win_rate', 0),
                'total_trades': info.get('total_trades', 0),
                'final_portfolio': info.get('portfolio_value', 0),
                'steps': episode_steps
            }
            episode_results.append(episode_result)
            
            print(f"  Эпизод {episode+1}: доходность={episode_result['total_return']:.2%}, "
                  f"просадка={episode_result['max_drawdown']:.2%}, "
                  f"сделок={episode_result['total_trades']}")
        
        # Агрегированные результаты
        results = {
            'episodes': episode_results,
            'mean_reward': np.mean([ep['total_reward'] for ep in episode_results]),
            'mean_return': np.mean([ep['total_return'] for ep in episode_results]),
            'mean_drawdown': np.mean([ep['max_drawdown'] for ep in episode_results]),
            'mean_win_rate': np.mean([ep['win_rate'] for ep in episode_results]),
            'mean_trades': np.mean([ep['total_trades'] for ep in episode_results]),
            'std_return': np.std([ep['total_return'] for ep in episode_results]),
            'sharpe_ratio': self._calculate_sharpe_ratio(episode_results),
            'win_rate_episodes': sum(1 for ep in episode_results if ep['total_return'] > 0) / num_episodes,
            'all_actions': all_actions
        }
        
        self.results = results
        return results
    
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
                'std_return': self.results['std_return'],
                'sharpe_ratio': self.results['sharpe_ratio'],
                'mean_drawdown': self.results['mean_drawdown'],
                'mean_win_rate': self.results['mean_win_rate'],
                'win_rate_episodes': self.results['win_rate_episodes'],
                'mean_trades_per_episode': self.results['mean_trades']
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
        reward_scheme='optimized'
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
    
    parser = argparse.ArgumentParser(description='Оценка DRL агента')
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