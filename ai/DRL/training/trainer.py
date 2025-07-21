"""Система обучения DRL агентов."""

import time
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from ..agents.base_agent import BaseAgent
from ..environments import TradingEnv
from ..config import DRLConfig, TradingConfig
from ..utils import DRLLogger
from .callbacks import TradingCallback, EvaluationCallback, CheckpointCallback
from .experiment_manager import ExperimentManager


class Trainer:
    """
    Система обучения DRL агентов.
    
    Управляет процессом обучения, включая:
    - Создание и управление средами
    - Настройка callbacks
    - Мониторинг прогресса
    - Сохранение моделей и результатов
    - Оценка производительности
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        env: TradingEnv,
        drl_config: DRLConfig,
        trading_config: TradingConfig,
        experiment_manager: Optional[ExperimentManager] = None,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация тренера.
        
        Args:
            agent: DRL агент для обучения
            env: торговая среда
            drl_config: конфигурация DRL
            trading_config: торговая конфигурация
            experiment_manager: менеджер экспериментов
            logger: логгер
        """
        self.agent = agent
        self.env = env
        self.drl_config = drl_config
        self.trading_config = trading_config
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.logger = logger or DRLLogger("trainer")
        
        # Среды для обучения и оценки
        self.train_env: Optional[Union[TradingEnv, VecEnv]] = None
        self.eval_env: Optional[TradingEnv] = None
        
        # Callbacks
        self.callbacks: List[BaseCallback] = []
        
        # Статистика обучения
        self.training_history: List[Dict[str, Any]] = []
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Состояние обучения
        self.is_training = False
        self.training_start_time: Optional[float] = None
        self.best_eval_reward = float('-inf')
        self.best_model_path: Optional[str] = None
        
        self.logger.info("Тренер инициализирован")
    
    def setup_environments(self, train_split: float = 0.7, val_split: float = 0.2):
        """
        Настройка сред для обучения и оценки.
        
        Args:
            train_split: доля данных для обучения
            val_split: доля данных для валидации
        """
        self.logger.info("Настройка сред обучения и оценки...")
        
        # Создание обучающей среды
        train_env = TradingEnv(self.trading_config, logger=self.logger)
        train_env.set_data_split("train")
        
        # Мониторинг среды
        monitor_path = self.experiment_manager.experiment_dir / "monitor" / "train"
        monitor_path.mkdir(parents=True, exist_ok=True)
        
        self.train_env = Monitor(train_env, str(monitor_path))
        
        # Создание среды для оценки
        eval_env = TradingEnv(self.trading_config, logger=self.logger)
        eval_env.set_data_split("val")
        
        eval_monitor_path = self.experiment_manager.experiment_dir / "monitor" / "eval"
        eval_monitor_path.mkdir(parents=True, exist_ok=True)
        
        self.eval_env = Monitor(eval_env, str(eval_monitor_path))
        
        # Векторизация обучающей среды если нужно
        if not isinstance(self.train_env, VecEnv):
            self.train_env = DummyVecEnv([lambda: self.train_env])
        
        # Установка сред в агенте
        self.agent.set_environments(self.train_env, self.eval_env)
        
        self.logger.info("Среды настроены успешно")
    
    def setup_callbacks(
        self,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        n_eval_episodes: int = 5
    ):
        """
        Настройка callbacks для мониторинга обучения.
        
        Args:
            eval_freq: частота оценки
            save_freq: частота сохранения
            n_eval_episodes: количество эпизодов для оценки
        """
        self.callbacks.clear()
        
        # Callback для оценки
        if self.eval_env is not None:
            eval_callback = EvaluationCallback(
                eval_env=self.eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                best_model_save_path=str(self.experiment_manager.models_dir),
                log_path=str(self.experiment_manager.logs_dir),
                deterministic=True,
                render=False,
                verbose=self.drl_config.verbose
            )
            self.callbacks.append(eval_callback)
        
        # Callback для сохранения checkpoint
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.experiment_manager.checkpoints_dir),
            name_prefix="checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=self.drl_config.verbose
        )
        self.callbacks.append(checkpoint_callback)
        
        # Callback для торговой статистики
        trading_callback = TradingCallback(
            log_dir=str(self.experiment_manager.logs_dir),
            log_freq=1000,
            save_freq=save_freq
        )
        self.callbacks.append(trading_callback)
        
        self.logger.info(f"Настроено {len(self.callbacks)} callbacks")
    
    def create_model(self, **model_kwargs):
        """
        Создание модели агента.
        
        Args:
            **model_kwargs: дополнительные параметры для модели
        """
        if self.train_env is None:
            self.setup_environments()
        
        self.logger.info("Создание модели агента...")
        
        # Создаем модель
        model = self.agent.create_model(self.train_env, **model_kwargs)
        
        self.logger.info(f"Модель {self.agent.__class__.__name__} создана")
        self.logger.debug(f"Гиперпараметры: {self.agent.get_hyperparameters()}")
        
        return model
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        n_eval_episodes: int = 5,
        **train_kwargs
    ) -> BaseAgent:
        """
        Основной метод обучения.
        
        Args:
            total_timesteps: общее количество шагов обучения
            eval_freq: частота оценки
            save_freq: частота сохранения
            n_eval_episodes: количество эпизодов для оценки
            **train_kwargs: дополнительные параметры обучения
            
        Returns:
            Обученный агент
        """
        self.logger.info("Начинаем процесс обучения...")
        
        try:
            # Подготовка к обучению
            self._prepare_training(eval_freq, save_freq, n_eval_episodes)
            
            # Сохранение конфигурации
            self._save_training_config(total_timesteps, eval_freq, save_freq, n_eval_episodes)
            
            # Создание модели если не создана
            if self.agent.model is None:
                self.create_model()
            
            # Запуск обучения
            self._run_training(total_timesteps, **train_kwargs)
            
            # Пост-обработка
            self._post_training_analysis()
            
            self.logger.info("Обучение завершено успешно!")
            
        except Exception as e:
            self.logger.error(f"Ошибка во время обучения: {e}")
            raise
        
        finally:
            self.is_training = False
        
        return self.agent
    
    def _prepare_training(self, eval_freq: int, save_freq: int, n_eval_episodes: int):
        """Подготовка к обучению."""
        # Настройка сред
        if self.train_env is None or self.eval_env is None:
            self.setup_environments()
        
        # Настройка callbacks
        self.setup_callbacks(eval_freq, save_freq, n_eval_episodes)
        
        # Сброс статистики
        self.training_history.clear()
        self.evaluation_history.clear()
        
        # Установка флагов
        self.is_training = True
        self.training_start_time = time.time()
    
    def _run_training(self, total_timesteps: int, **train_kwargs):
        """Запуск основного цикла обучения."""
        self.logger.info(f"Запуск обучения на {total_timesteps:,} шагов")
        
        # Объединение callbacks
        callback_list = CallbackList(self.callbacks) if len(self.callbacks) > 1 else (
            self.callbacks[0] if self.callbacks else None
        )
        
        # Основное обучение
        self.agent.train(
            total_timesteps=total_timesteps,
            callback=callback_list,
            **train_kwargs
        )
        
        # Обновление статистики
        training_time = time.time() - self.training_start_time
        self.agent.update_training_stats(
            timesteps=total_timesteps,
            training_time=training_time
        )
    
    def _post_training_analysis(self):
        """Анализ после завершения обучения."""
        self.logger.info("Проведение пост-анализа обучения...")
        
        # Финальная оценка
        final_eval = self.evaluate(n_episodes=10, deterministic=True)
        self.evaluation_history.append({
            'type': 'final',
            'timestamp': datetime.now().isoformat(),
            'metrics': final_eval
        })
        
        # Сохранение истории обучения
        self._save_training_history()
        
        # Генерация отчета
        self._generate_training_report()
    
    def _save_training_config(self, total_timesteps: int, eval_freq: int, save_freq: int, n_eval_episodes: int):
        """Сохранение конфигурации обучения."""
        config = {
            'experiment_info': {
                'experiment_name': self.experiment_manager.experiment_name,
                'start_time': datetime.now().isoformat(),
                'agent_type': self.drl_config.agent_type,
                'total_timesteps': total_timesteps,
                'eval_freq': eval_freq,
                'save_freq': save_freq,
                'n_eval_episodes': n_eval_episodes
            },
            'drl_config': self.drl_config.__dict__,
            'trading_config': self.trading_config.__dict__,
            'agent_hyperparameters': self.agent.get_hyperparameters(),
            'environment_info': {
                'symbol': self.trading_config.symbol,
                'timeframe': self.trading_config.timeframe,
                'lookback_window': self.trading_config.lookback_window,
                'action_type': self.trading_config.action_type,
                'reward_scheme': self.trading_config.reward_scheme
            }
        }
        
        config_path = self.experiment_manager.experiment_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Конфигурация сохранена: {config_path}")
    
    def _save_training_history(self):
        """Сохранение истории обучения."""
        history = {
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'agent_stats': self.agent.get_training_stats(),
            'best_eval_reward': self.best_eval_reward,
            'best_model_path': self.best_model_path
        }
        
        history_path = self.experiment_manager.experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Также сохраняем в pickle для Python объектов
        pickle_path = self.experiment_manager.experiment_dir / "training_history.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(history, f)
        
        self.logger.info(f"История обучения сохранена: {history_path}")
    
    def _generate_training_report(self):
        """Генерация отчета об обучении."""
        if not self.evaluation_history:
            return
        
        report = {
            'summary': {
                'experiment_name': self.experiment_manager.experiment_name,
                'agent_type': self.drl_config.agent_type,
                'symbol': self.trading_config.symbol,
                'total_training_time': time.time() - self.training_start_time if self.training_start_time else 0,
                'total_timesteps': self.agent.get_training_stats().get('total_timesteps', 0),
                'best_eval_reward': self.best_eval_reward
            }
        }
        
        # Добавляем последние метрики оценки
        if self.evaluation_history:
            latest_eval = self.evaluation_history[-1].get('metrics', {})
            report['final_performance'] = latest_eval
        
        # Добавляем конфигурации
        report['configurations'] = {
            'drl_config': {k: v for k, v in vars(self.drl_config).items() if not k.startswith('_')},
            'trading_config': {k: v for k, v in vars(self.trading_config).items() if not k.startswith('_')}
        }
        
        report_path = self.experiment_manager.experiment_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Отчет об обучении сохранен: {report_path}")
        
        # Вывод краткой сводки
        self._print_training_summary(report)
    
    def _print_training_summary(self, report: Dict[str, Any]):
        """Вывод краткой сводки обучения."""
        summary = report['summary']
        
        self.logger.info("=" * 60)
        self.logger.info("СВОДКА ОБУЧЕНИЯ")
        self.logger.info("=" * 60)
        self.logger.info(f"Эксперимент: {summary['experiment_name']}")
        self.logger.info(f"Агент: {summary['agent_type']}")
        self.logger.info(f"Символ: {summary['symbol']}")
        self.logger.info(f"Время обучения: {summary['total_training_time']:.2f} сек")
        self.logger.info(f"Общие шаги: {summary['total_timesteps']:,}")
        self.logger.info(f"Лучшая награда: {summary['best_eval_reward']:.4f}")
        
        if 'final_performance' in report:
            perf = report['final_performance']
            self.logger.info("\nФинальная производительность:")
            for key, value in perf.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 60)
    
    def evaluate(
        self, 
        n_episodes: int = 5,
        deterministic: bool = True,
        env: Optional[TradingEnv] = None
    ) -> Dict[str, Any]:
        """
        Оценка производительности агента.
        
        Args:
            n_episodes: количество эпизодов для оценки
            deterministic: использовать детерминистичную политику
            env: среда для оценки
            
        Returns:
            Метрики производительности
        """
        eval_env = env or self.eval_env
        if eval_env is None:
            raise ValueError("Среда для оценки не настроена")
        
        self.logger.info(f"Оценка агента на {n_episodes} эпизодах...")
        
        metrics = self.agent.evaluate(
            env=eval_env,
            n_episodes=n_episodes,
            deterministic=deterministic
        )
        
        # Обновляем лучший результат
        mean_reward = metrics.get('mean_reward', float('-inf'))
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            # Сохраняем лучшую модель
            best_path = self.experiment_manager.models_dir / "best_model"
            self.agent.save(str(best_path))
            self.best_model_path = str(best_path)
        
        return metrics
    
    def save_checkpoint(self, name: str = None) -> str:
        """
        Сохранение checkpoint.
        
        Args:
            name: имя checkpoint
            
        Returns:
            Путь к сохраненному checkpoint
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"checkpoint_{timestamp}"
        
        checkpoint_path = self.experiment_manager.checkpoints_dir / name
        
        # Сохраняем модель
        model_path = str(checkpoint_path) + "_model"
        self.agent.save(model_path)
        
        # Сохраняем состояние тренера
        trainer_state = {
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'best_eval_reward': self.best_eval_reward,
            'best_model_path': self.best_model_path,
            'agent_stats': self.agent.get_training_stats()
        }
        
        state_path = str(checkpoint_path) + "_trainer_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(trainer_state, f)
        
        self.logger.info(f"Checkpoint сохранен: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Загрузка checkpoint.
        
        Args:
            checkpoint_path: путь к checkpoint
            
        Returns:
            True если загрузка успешна
        """
        try:
            # Загружаем модель
            model_path = checkpoint_path + "_model"
            if Path(model_path + ".zip").exists():
                self.agent.load(model_path, self.train_env)
            else:
                self.logger.warning(f"Модель не найдена: {model_path}")
                return False
            
            # Загружаем состояние тренера
            state_path = checkpoint_path + "_trainer_state.pkl"
            if Path(state_path).exists():
                with open(state_path, 'rb') as f:
                    trainer_state = pickle.load(f)
                
                self.training_history = trainer_state.get('training_history', [])
                self.evaluation_history = trainer_state.get('evaluation_history', [])
                self.best_eval_reward = trainer_state.get('best_eval_reward', float('-inf'))
                self.best_model_path = trainer_state.get('best_model_path')
                
                # Восстанавливаем статистику агента
                agent_stats = trainer_state.get('agent_stats', {})
                for key, value in agent_stats.items():
                    self.agent.training_stats[key] = value
            
            self.logger.info(f"Checkpoint загружен: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки checkpoint: {e}")
            return False
    
    def load_latest_checkpoint(self) -> bool:
        """
        Загрузка последнего checkpoint.
        
        Returns:
            True если checkpoint найден и загружен
        """
        checkpoints_dir = self.experiment_manager.checkpoints_dir
        if not checkpoints_dir.exists():
            return False
        
        # Ищем все checkpoint файлы
        checkpoint_files = list(checkpoints_dir.glob("checkpoint_*_model.zip"))
        if not checkpoint_files:
            return False
        
        # Сортируем по времени модификации
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        
        # Получаем базовое имя без суффикса
        checkpoint_base = str(latest_checkpoint).replace("_model.zip", "")
        
        return self.load_checkpoint(checkpoint_base)
    
    def save_final_model(self, name: str = "final_model") -> str:
        """
        Сохранение финальной модели.
        
        Args:
            name: имя модели
            
        Returns:
            Путь к сохраненной модели
        """
        model_path = self.experiment_manager.models_dir / name
        final_path = self.agent.save(str(model_path))
        
        self.logger.info(f"Финальная модель сохранена: {final_path}")
        return final_path
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Получение информации о прогрессе обучения."""
        return {
            'is_training': self.is_training,
            'training_start_time': self.training_start_time,
            'elapsed_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'agent_stats': self.agent.get_training_stats(),
            'best_eval_reward': self.best_eval_reward,
            'evaluations_count': len(self.evaluation_history),
            'latest_evaluation': self.evaluation_history[-1] if self.evaluation_history else None
        }