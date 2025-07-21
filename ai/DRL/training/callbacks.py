"""Callback система для мониторинга обучения DRL агентов."""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback as SB3CheckpointCallback
from stable_baselines3.common.logger import Figure

from ..utils import DRLLogger


class TradingCallback(BaseCallback):
    """
    Callback для мониторинга торговых метрик во время обучения.
    
    Отслеживает:
    - Производительность портфеля
    - Торговую статистику
    - Метрики risk/reward
    - Прогресс обучения
    """
    
    def __init__(
        self,
        log_dir: str,
        log_freq: int = 1000,
        save_freq: int = 10000,
        verbose: int = 0
    ):
        """
        Инициализация торгового callback.
        
        Args:
            log_dir: директория для логов
            log_freq: частота логирования (в шагах)
            save_freq: частота сохранения метрик
            verbose: уровень детализации
        """
        super().__init__(verbose)
        
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.save_freq = save_freq
        
        # Создаем директории
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Логгер (используем другое имя, чтобы не конфликтовать с BaseCallback.logger)
        self.drl_logger = DRLLogger("trading_callback")
        
        # Метрики для отслеживания
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.portfolio_values: List[float] = []
        self.trade_counts: List[int] = []
        self.win_rates: List[float] = []
        self.drawdowns: List[float] = []
        
        # Состояние текущего эпизода
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_trades = 0
        
        # Счетчики
        self.step_count = 0
        self.episode_count = 0
        
        if self.verbose >= 1:
            self.drl_logger.info(f"TradingCallback инициализирован: {log_dir}")
    
    def _on_training_start(self) -> None:
        """Вызывается в начале обучения."""
        if self.verbose >= 1:
            self.drl_logger.info("Начинаем мониторинг торговых метрик")
    
    def _on_step(self) -> bool:
        """
        Вызывается на каждом шаге обучения.
        
        Returns:
            True для продолжения обучения
        """
        self.step_count += 1
        
        # Получаем информацию из среды
        infos = self.locals.get('infos', [])
        
        if infos:
            for info in infos:
                self._process_step_info(info)
        
        # Логирование через определенные интервалы
        if self.step_count % self.log_freq == 0:
            self._log_progress()
        
        # Сохранение метрик
        if self.step_count % self.save_freq == 0:
            self._save_metrics()
        
        return True
    
    def _process_step_info(self, info: Dict[str, Any]):
        """Обработка информации о шаге."""
        # Проверяем, что info является словарем
        if not isinstance(info, dict):
            return
        
        # Обновляем текущий эпизод
        reward = info.get('reward', 0)
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Торговая информация
        if 'trade' in info and isinstance(info['trade'], dict) and info['trade'].get('executed', False):
            self.current_episode_trades += 1
        
        # Проверяем завершение эпизода
        episode_info = info.get('episode', {})
        if info.get('done', False) or (isinstance(episode_info, dict) and episode_info.get('done', False)):
            self._process_episode_end(info)
    
    def _process_episode_end(self, info: Dict[str, Any]):
        """Обработка завершения эпизода."""
        # Сохраняем метрики эпизода
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        self.trade_counts.append(self.current_episode_trades)
        
        # Портфельные метрики
        portfolio_info = info.get('portfolio', {})
        if portfolio_info:
            total_value = portfolio_info.get('total_value', 0)
            self.portfolio_values.append(total_value)
            
            drawdown = portfolio_info.get('drawdown', 0)
            self.drawdowns.append(drawdown)
        
        # Метрики торговли
        if 'metrics' in info:
            metrics = info['metrics']
            win_rate = metrics.get('win_rate', 0)
            self.win_rates.append(win_rate)
        
        # Увеличиваем счетчик эпизодов
        self.episode_count += 1
        
        # Сброс состояния эпизода
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_trades = 0
        
        if self.verbose >= 2:
            self.drl_logger.debug(f"Эпизод {self.episode_count} завершен: "
                            f"награда={self.episode_rewards[-1]:.4f}, "
                            f"длина={self.episode_lengths[-1]}")
    
    def _log_progress(self):
        """Логирование прогресса обучения."""
        if not self.episode_rewards:
            return
        
        # Вычисляем статистики
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        recent_lengths = self.episode_lengths[-10:] if len(self.episode_lengths) >= 10 else self.episode_lengths
        mean_length = np.mean(recent_lengths)
        
        # Логируем в tensorboard если доступен
        if hasattr(self.logger, 'record'):
            self.logger.record('trading/mean_episode_reward', mean_reward)
            self.logger.record('trading/std_episode_reward', std_reward)
            self.logger.record('trading/mean_episode_length', mean_length)
            self.logger.record('trading/total_episodes', self.episode_count)
            
            if self.portfolio_values:
                recent_values = self.portfolio_values[-10:]
                mean_portfolio_value = np.mean(recent_values)
                self.logger.record('trading/mean_portfolio_value', mean_portfolio_value)
            
            if self.trade_counts:
                recent_trades = self.trade_counts[-10:]
                mean_trades = np.mean(recent_trades)
                self.logger.record('trading/mean_trades_per_episode', mean_trades)
            
            if self.win_rates:
                recent_win_rates = self.win_rates[-10:]
                mean_win_rate = np.mean(recent_win_rates)
                self.logger.record('trading/mean_win_rate', mean_win_rate)
            
            if self.drawdowns:
                recent_drawdowns = self.drawdowns[-10:]
                mean_drawdown = np.mean(recent_drawdowns)
                max_drawdown = np.max(recent_drawdowns) if recent_drawdowns else 0
                self.logger.record('trading/mean_drawdown', mean_drawdown)
                self.logger.record('trading/max_drawdown', max_drawdown)
        
        # Консольный вывод
        if self.verbose >= 1:
            self.drl_logger.info(f"Шаг {self.step_count}: средняя награда={mean_reward:.4f}±{std_reward:.4f}, "
                           f"эпизодов={self.episode_count}")
    
    def _save_metrics(self):
        """Сохранение метрик в файлы."""
        try:
            metrics_data = {
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'portfolio_values': self.portfolio_values,
                'trade_counts': self.trade_counts,
                'win_rates': self.win_rates,
                'drawdowns': self.drawdowns
            }
            
            # Сохраняем как JSON
            metrics_path = self.log_dir / "trading_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=float)
            
            # Также сохраняем статистики
            if self.episode_rewards:
                stats = {
                    'mean_reward': float(np.mean(self.episode_rewards)),
                    'std_reward': float(np.std(self.episode_rewards)),
                    'min_reward': float(np.min(self.episode_rewards)),
                    'max_reward': float(np.max(self.episode_rewards)),
                    'total_episodes': len(self.episode_rewards)
                }
                
                if self.portfolio_values:
                    stats.update({
                        'mean_portfolio_value': float(np.mean(self.portfolio_values)),
                        'final_portfolio_value': float(self.portfolio_values[-1])
                    })
                
                if self.win_rates:
                    stats['mean_win_rate'] = float(np.mean(self.win_rates))
                
                if self.drawdowns:
                    stats['max_drawdown'] = float(np.max(self.drawdowns))
                
                stats_path = self.log_dir / "trading_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
            
            if self.verbose >= 2:
                self.drl_logger.debug(f"Метрики сохранены: {metrics_path}")
                
        except Exception as e:
            self.drl_logger.error(f"Ошибка сохранения метрик: {e}")
    
    def _on_training_end(self) -> None:
        """Вызывается в конце обучения."""
        self._save_metrics()
        
        if self.verbose >= 1:
            self.drl_logger.info(f"Мониторинг завершен. Всего эпизодов: {self.episode_count}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Получение текущих статистик."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'mean_episode_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
            'mean_portfolio_value': float(np.mean(self.portfolio_values)) if self.portfolio_values else 0,
            'mean_win_rate': float(np.mean(self.win_rates)) if self.win_rates else 0,
            'max_drawdown': float(np.max(self.drawdowns)) if self.drawdowns else 0
        }


class EvaluationCallback(EvalCallback):
    """
    Расширенный callback для оценки с торговыми метриками.
    
    Наследует от stable-baselines3 EvalCallback и добавляет
    специфичные для торговли метрики и логирование.
    """
    
    def __init__(
        self,
        eval_env,
        callback_on_new_best=None,
        callback_after_eval=None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        """
        Инициализация расширенного evaluation callback.
        
        Args:
            eval_env: среда для оценки
            callback_on_new_best: callback при новой лучшей модели
            callback_after_eval: callback после оценки
            n_eval_episodes: количество эпизодов для оценки
            eval_freq: частота оценки
            log_path: путь для логов
            best_model_save_path: путь для сохранения лучшей модели
            deterministic: детерминистичная политика
            render: рендеринг среды
            verbose: уровень детализации
            warn: показывать предупреждения
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn
        )
        
        self.trading_logger = DRLLogger("evaluation_callback")
        
        # Дополнительные торговые метрики
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def _on_step(self) -> bool:
        """Переопределение метода оценки."""
        result = super()._on_step()
        
        # Добавляем дополнительную обработку после оценки
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._process_trading_evaluation()
        
        return result
    
    def _process_trading_evaluation(self):
        """Обработка торговых метрик после оценки."""
        if hasattr(self, 'last_mean_reward'):
            # Собираем дополнительную информацию о последней оценке
            evaluation_info = {
                'step': self.n_calls,
                'mean_reward': self.last_mean_reward,
                'std_reward': getattr(self, 'last_std_reward', 0),
                'timestamp': self.num_timesteps
            }
            
            # Если есть доступ к результатам эпизодов, добавляем торговые метрики
            if hasattr(self, '_last_episode_infos'):
                trading_metrics = self._extract_trading_metrics(self._last_episode_infos)
                evaluation_info.update(trading_metrics)
            
            self.evaluation_history.append(evaluation_info)
            
            # Логируем торговые метрики
            if self.verbose >= 1:
                self.trading_logger.info(f"Оценка на шаге {self.n_calls}: награда={self.last_mean_reward:.4f}")
    
    def _extract_trading_metrics(self, episode_infos: List[Dict]) -> Dict[str, Any]:
        """Извлечение торговых метрик из информации об эпизодах."""
        if not episode_infos:
            return {}
        
        portfolio_values = []
        total_returns = []
        win_rates = []
        drawdowns = []
        trade_counts = []
        
        for info in episode_infos:
            portfolio_info = info.get('portfolio', {})
            if portfolio_info:
                portfolio_values.append(portfolio_info.get('total_value', 0))
                total_returns.append(portfolio_info.get('total_return', 0))
                drawdowns.append(portfolio_info.get('drawdown', 0))
                trade_counts.append(portfolio_info.get('trades_count', 0))
            
            metrics = info.get('metrics', {})
            if metrics:
                win_rates.append(metrics.get('win_rate', 0))
        
        trading_metrics = {}
        
        if portfolio_values:
            trading_metrics.update({
                'mean_portfolio_value': float(np.mean(portfolio_values)),
                'std_portfolio_value': float(np.std(portfolio_values))
            })
        
        if total_returns:
            trading_metrics.update({
                'mean_total_return': float(np.mean(total_returns)),
                'std_total_return': float(np.std(total_returns))
            })
        
        if win_rates:
            trading_metrics['mean_win_rate'] = float(np.mean(win_rates))
        
        if drawdowns:
            trading_metrics['max_drawdown'] = float(np.max(drawdowns))
        
        if trade_counts:
            trading_metrics['mean_trades_per_episode'] = float(np.mean(trade_counts))
        
        return trading_metrics
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Получение истории оценок."""
        return self.evaluation_history.copy()


class CheckpointCallback(SB3CheckpointCallback):
    """
    Расширенный callback для сохранения checkpoint с дополнительными данными.
    
    Сохраняет не только модель, но и дополнительную информацию
    о состоянии обучения и торговых метриках.
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        """
        Инициализация checkpoint callback.
        
        Args:
            save_freq: частота сохранения
            save_path: путь для сохранения
            name_prefix: префикс имени файла
            save_replay_buffer: сохранять replay buffer
            save_vecnormalize: сохранять векторную нормализацию
            verbose: уровень детализации
        """
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose
        )
        
        self.checkpoint_logger = DRLLogger("checkpoint_callback")
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Переопределение сохранения checkpoint."""  
        result = super()._on_step()
        
        # Сохраняем дополнительную информацию при checkpoint
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self._save_additional_info()
        
        return result
    
    def _save_additional_info(self):
        """Сохранение дополнительной информации о состоянии обучения."""
        try:
            # Информация о состоянии обучения
            training_info = {
                'step': self.n_calls,
                'num_timesteps': self.num_timesteps,
                'model_class': self.model.__class__.__name__,
                'learning_rate': getattr(self.model, 'learning_rate', None),
                'timestamp': self.n_calls
            }
            
            # Сохраняем в файл рядом с checkpoint
            checkpoint_name = f"{self.name_prefix}_{self.n_calls}_steps"
            info_path = self.save_path / f"{checkpoint_name}_info.json"
            
            with open(info_path, 'w') as f:
                json.dump(training_info, f, indent=2, default=str)
            
            if self.verbose >= 2:
                self.checkpoint_logger.debug(f"Дополнительная информация сохранена: {info_path}")
                
        except Exception as e:
            self.checkpoint_logger.error(f"Ошибка сохранения дополнительной информации: {e}")


class TensorboardCallback(BaseCallback):
    """
    Callback для расширенного логирования в TensorBoard.
    
    Добавляет специфичные для торговли метрики и графики.
    """
    
    def __init__(
        self,
        log_dir: str,
        log_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Инициализация TensorBoard callback.
        
        Args:
            log_dir: директория для TensorBoard логов
            log_freq: частота логирования
            verbose: уровень детализации
        """
        super().__init__(verbose)
        
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        
        # Создаем директорию
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.tb_logger = DRLLogger("tensorboard_callback")
        
        # Счетчики
        self.step_count = 0
        
        # Накопители для метрик
        self.rewards_buffer: List[float] = []
        self.portfolio_values_buffer: List[float] = []
        self.actions_buffer: List[float] = []
    
    def _on_step(self) -> bool:
        """Логирование метрик в TensorBoard."""
        self.step_count += 1
        
        # Собираем данные из среды
        infos = self.locals.get('infos', [])
        
        for info in infos:
            # Награды
            if 'reward' in info:
                self.rewards_buffer.append(info['reward'])
            
            # Информация о портфеле
            portfolio_info = info.get('portfolio', {})
            if portfolio_info and 'total_value' in portfolio_info:
                self.portfolio_values_buffer.append(portfolio_info['total_value'])
        
        # Действия агента
        actions = self.locals.get('actions')
        if actions is not None:
            if hasattr(actions, '__iter__'):
                self.actions_buffer.extend(actions.flatten())
            else:
                self.actions_buffer.append(float(actions))
        
        # Логирование через определенные интервалы
        if self.step_count % self.log_freq == 0:
            self._log_to_tensorboard()
            self._clear_buffers()
        
        return True
    
    def _log_to_tensorboard(self):
        """Логирование накопленных метрик в TensorBoard."""
        # Используем встроенный логгер stable-baselines3
        if hasattr(self.logger, 'record'):
            # Награды
            if self.rewards_buffer:
                self.logger.record('custom/mean_reward', np.mean(self.rewards_buffer))
                self.logger.record('custom/std_reward', np.std(self.rewards_buffer))
                self.logger.record('custom/min_reward', np.min(self.rewards_buffer))
                self.logger.record('custom/max_reward', np.max(self.rewards_buffer))
            
            # Стоимость портфеля
            if self.portfolio_values_buffer:
                self.logger.record('custom/mean_portfolio_value', np.mean(self.portfolio_values_buffer))
                self.logger.record('custom/std_portfolio_value', np.std(self.portfolio_values_buffer))
            
            # Действия агента
            if self.actions_buffer:
                self.logger.record('custom/mean_action', np.mean(self.actions_buffer))
                self.logger.record('custom/std_action', np.std(self.actions_buffer))
                
                # Гистограмма действий (если дискретные)
                unique_actions, counts = np.unique(self.actions_buffer, return_counts=True)
                if len(unique_actions) <= 10:  # Только для дискретных действий
                    for action, count in zip(unique_actions, counts):
                        self.logger.record(f'custom/action_{int(action)}_count', count)
    
    def _clear_buffers(self):
        """Очистка буферов метрик."""
        self.rewards_buffer.clear()
        self.portfolio_values_buffer.clear()
        self.actions_buffer.clear()
    
    def _on_training_end(self) -> None:
        """Финальное логирование в конце обучения."""
        if self.rewards_buffer or self.portfolio_values_buffer or self.actions_buffer:
            self._log_to_tensorboard()
        
        if self.verbose >= 1:
            self.tb_logger.info("TensorBoard логирование завершено")