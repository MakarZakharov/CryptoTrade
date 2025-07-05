"""
Callbacks для процесса обучения DRL агентов.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


class TradingCallback(BaseCallback):
    """Callback для мониторинга торговых метрик."""
    
    def __init__(self, log_dir: str, experiment_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_drawdowns = []
        self.episode_win_rates = []
        self.episode_trades = []
        
        # Создаем директорию для логов
        os.makedirs(log_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        """Вызывается на каждом шаге."""
        # Получаем информацию из среды
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Записываем метрики в tensorboard
            if 'portfolio_value' in info:
                self.logger.record('trading/portfolio_value', info['portfolio_value'])
            if 'total_return' in info:
                self.logger.record('trading/total_return', info['total_return'])
            if 'max_drawdown' in info:
                self.logger.record('trading/max_drawdown', info['max_drawdown'])
            if 'win_rate' in info:
                self.logger.record('trading/win_rate', info['win_rate'])
            if 'total_trades' in info:
                self.logger.record('trading/total_trades', info['total_trades'])
        
        return True
    
    def _on_episode_end(self) -> None:
        """Вызывается в конце эпизода."""
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Сохраняем метрики эпизода
            self.episode_returns.append(info.get('total_return', 0))
            self.episode_drawdowns.append(info.get('max_drawdown', 0))
            self.episode_win_rates.append(info.get('win_rate', 0))
            self.episode_trades.append(info.get('total_trades', 0))
            
            # Записываем агрегированные метрики
            if len(self.episode_returns) > 0:
                self.logger.record('episode/mean_return', np.mean(self.episode_returns[-100:]))
                self.logger.record('episode/mean_drawdown', np.mean(self.episode_drawdowns[-100:]))
                self.logger.record('episode/mean_win_rate', np.mean(self.episode_win_rates[-100:]))
                self.logger.record('episode/mean_trades', np.mean(self.episode_trades[-100:]))
    
    def _on_training_end(self) -> None:
        """Вызывается в конце обучения."""
        # Сохраняем итоговые метрики
        metrics = {
            'final_mean_return': np.mean(self.episode_returns[-50:]) if self.episode_returns else 0,
            'final_mean_drawdown': np.mean(self.episode_drawdowns[-50:]) if self.episode_drawdowns else 0,
            'final_mean_win_rate': np.mean(self.episode_win_rates[-50:]) if self.episode_win_rates else 0,
            'final_mean_trades': np.mean(self.episode_trades[-50:]) if self.episode_trades else 0,
            'total_episodes': len(self.episode_returns)
        }
        
        # Сохраняем в CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{self.log_dir}/final_metrics.csv", index=False)
        
        print(f"📊 Итоговые метрики:")
        print(f"  Средняя доходность: {metrics['final_mean_return']:.2%}")
        print(f"  Средняя просадка: {metrics['final_mean_drawdown']:.2%}")
        print(f"  Средний win rate: {metrics['final_mean_win_rate']:.2%}")
        print(f"  Среднее количество сделок: {metrics['final_mean_trades']:.1f}")


class TensorboardCallback(BaseCallback):
    """Callback для расширенного логирования в Tensorboard."""
    
    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.step_count = 0
        
        # Создаем директорию
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Логирование на каждом шаге."""
        self.step_count += 1
        
        # Логируем каждые 1000 шагов
        if self.step_count % 1000 == 0:
            # Получаем награды
            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                if len(rewards) > 0:
                    self.logger.record('reward/mean_reward', np.mean(rewards))
                    self.logger.record('reward/max_reward', np.max(rewards))
                    self.logger.record('reward/min_reward', np.min(rewards))
            
            # Логируем действия агента
            if 'actions' in self.locals:
                actions = self.locals['actions']
                if len(actions) > 0:
                    self.logger.record('action/mean_action', np.mean(actions))
                    self.logger.record('action/std_action', np.std(actions))
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """Callback для раннего остановки обучения."""
    
    def __init__(self, patience: int = 50, min_improvement: float = 0.01, verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.patience_counter = 0
    
    def _on_step(self) -> bool:
        """Проверка условий остановки."""
        # Получаем текущую среднюю награду
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            # Проверяем улучшение
            if mean_reward > self.best_mean_reward + self.min_improvement:
                self.best_mean_reward = mean_reward
                self.patience_counter = 0
                if self.verbose > 0:
                    print(f"📈 Новый лучший результат: {mean_reward:.4f}")
            else:
                self.patience_counter += 1
            
            # Проверяем условие остановки
            if self.patience_counter >= self.patience:
                if self.verbose > 0:
                    print(f"🛑 Ранняя остановка: нет улучшений {self.patience} шагов")
                return False
        
        return True


class PerformanceMonitorCallback(BaseCallback):
    """Callback для мониторинга производительности системы."""
    
    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.step_count = 0
        
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            if verbose > 0:
                print("⚠️ psutil не доступен, мониторинг производительности отключен")
    
    def _on_step(self) -> bool:
        """Мониторинг производительности."""
        self.step_count += 1
        
        if self.step_count % self.log_freq == 0 and self.psutil_available:
            import psutil
            
            # Мониторинг памяти
            memory_info = psutil.virtual_memory()
            self.logger.record('system/memory_usage_percent', memory_info.percent)
            self.logger.record('system/memory_available_gb', memory_info.available / 1024**3)
            
            # Мониторинг CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.logger.record('system/cpu_usage_percent', cpu_percent)
            
            # Мониторинг GPU (если доступен)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    self.logger.record('system/gpu_memory_gb', gpu_memory)
            except ImportError:
                pass
        
        return True 