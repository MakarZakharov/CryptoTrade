"""Система автоматической настройки гиперпараметров для DRL агентов."""

import optuna
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
import pickle

from ..config import DRLConfig, TradingConfig
from .logger import DRLLogger


class HyperparameterTuner:
    """
    Система автоматической настройки гиперпараметров для DRL агентов.
    
    Использует библиотеку Optuna для оптимизации гиперпараметров
    с учетом специфики торговых задач и различных метрик производительности.
    """
    
    def __init__(
        self,
        base_drl_config: DRLConfig,
        base_trading_config: TradingConfig,
        study_name: str = None,
        storage: Optional[str] = None,
        direction: str = "maximize",
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация тюнера гиперпараметров.
        
        Args:
            base_drl_config: базовая DRL конфигурация
            base_trading_config: базовая торговая конфигурация
            study_name: имя исследования Optuna
            storage: URL базы данных для хранения (опционально)
            direction: направление оптимизации ("maximize" или "minimize")
            logger: логгер
        """
        self.base_drl_config = base_drl_config
        self.base_trading_config = base_trading_config
        self.logger = logger or DRLLogger("hyperparameter_tuner")
        
        # Настройка Optuna
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"drl_tuning_{timestamp}"
        
        self.study_name = study_name
        self.direction = direction
        
        # Создание или загрузка study
        if storage:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=direction,
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction
            )
        
        # Менеджер экспериментов для организации результатов (lazy import)
        from ..training import ExperimentManager
        self.experiment_manager = ExperimentManager(
            base_dir="CryptoTrade/ai/DRL/hyperparameter_tuning",
            experiment_name=study_name
        )
        
        # История оптимизации
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Лучшие параметры
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        
        self.logger.info(f"HyperparameterTuner инициализирован: {study_name}")
    
    def suggest_hyperparameters(
        self, 
        trial: optuna.Trial, 
        agent_type: str,
        parameter_ranges: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Предложение гиперпараметров для trial.
        
        Args:
            trial: Optuna trial объект
            agent_type: тип агента ("PPO", "SAC", "DQN", "A2C")
            parameter_ranges: пользовательские диапазоны параметров
            
        Returns:
            Словарь предложенных гиперпараметров
        """
        # Базовые параметры для всех агентов
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
            "net_arch_size": trial.suggest_categorical("net_arch_size", [32, 64, 128, 256]),
            "net_arch_layers": trial.suggest_int("net_arch_layers", 1, 3),
            "activation_fn": trial.suggest_categorical("activation_fn", ["relu", "tanh", "elu"])
        }
        
        # Создание архитектуры сети
        net_arch = [params["net_arch_size"]] * params["net_arch_layers"]
        params["net_arch"] = net_arch
        
        # Специфичные параметры для разных агентов
        if agent_type.upper() == "PPO":
            params.update({
                "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
                "n_epochs": trial.suggest_int("n_epochs", 3, 20),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
                "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99)
            })
        
        elif agent_type.upper() == "SAC":
            params.update({
                "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000, 500000]),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True),
                "learning_starts": trial.suggest_int("learning_starts", 1000, 10000)
            })
        
        elif agent_type.upper() == "DQN":
            params.update({
                "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000, 500000]),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
                "exploration_initial_eps": trial.suggest_float("exploration_initial_eps", 0.8, 1.0),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
                "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1000, 2000, 5000])
            })
        
        elif agent_type.upper() == "A2C":
            params.update({
                "n_steps": trial.suggest_categorical("n_steps", [5, 10, 20, 50]),
                "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0)
            })
        
        # Применение пользовательских диапазонов
        if parameter_ranges:
            for param_name, param_config in parameter_ranges.items():
                if param_name in params:
                    continue  # Пропускаем уже установленные параметры
                
                param_type = param_config.get("type", "float")
                
                if param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
        
        return params
    
    def create_objective_function(
        self,
        agent_type: str,
        training_timesteps: int = 50000,
        evaluation_episodes: int = 5,
        optimization_metric: str = "mean_reward",
        parameter_ranges: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Создание целевой функции для оптимизации.
        
        Args:
            agent_type: тип агента
            training_timesteps: количество шагов обучения
            evaluation_episodes: количество эпизодов для оценки
            optimization_metric: метрика для оптимизации
            parameter_ranges: пользовательские диапазоны параметров
            
        Returns:
            Целевая функция для Optuna
        """
        def objective(trial: optuna.Trial) -> float:
            try:
                # Получаем предложенные гиперпараметры
                suggested_params = self.suggest_hyperparameters(
                    trial, agent_type, parameter_ranges
                )
                
                # Создаем конфигурации с предложенными параметрами
                drl_config = DRLConfig(**{
                    **self.base_drl_config.__dict__,
                    **suggested_params,
                    "agent_type": agent_type.upper(),
                    "total_timesteps": training_timesteps,
                    "verbose": 0  # Отключаем verbose для ускорения
                })
                
                trading_config = TradingConfig(**self.base_trading_config.__dict__)
                
                # Lazy imports to avoid circular dependencies
                from ..environments import TradingEnv
                from ..agents import PPOAgent, SACAgent, DQNAgent, A2CAgent
                from ..training import Trainer, ExperimentManager
                
                # Создание среды и агента
                env = TradingEnv(trading_config, logger=self.logger)
                
                if agent_type.upper() == "PPO":
                    agent = PPOAgent(drl_config, trading_config, self.logger)
                elif agent_type.upper() == "SAC":
                    agent = SACAgent(drl_config, trading_config, self.logger)
                elif agent_type.upper() == "DQN":
                    agent = DQNAgent(drl_config, trading_config, self.logger)
                elif agent_type.upper() == "A2C":
                    agent = A2CAgent(drl_config, trading_config, self.logger)
                else:
                    raise ValueError(f"Неподдерживаемый тип агента: {agent_type}")
                
                # Создание тренера
                experiment_name = f"trial_{trial.number}_{agent_type.lower()}"
                trial_experiment_manager = ExperimentManager(
                    base_dir=self.experiment_manager.experiment_dir / "trials",
                    experiment_name=experiment_name
                )
                
                trainer = Trainer(
                    agent=agent,
                    env=env,
                    drl_config=drl_config,
                    trading_config=trading_config,
                    experiment_manager=trial_experiment_manager,
                    logger=self.logger
                )
                
                # Обучение с минимальным логированием
                trainer.train(
                    total_timesteps=training_timesteps,
                    eval_freq=training_timesteps // 2,  # Одна оценка в середине
                    save_freq=training_timesteps * 2,  # Не сохраняем промежуточные модели
                    n_eval_episodes=evaluation_episodes
                )
                
                # Финальная оценка
                final_metrics = trainer.evaluate(n_episodes=evaluation_episodes)
                
                # Получаем значение целевой метрики
                objective_value = final_metrics.get(optimization_metric, 0.0)
                
                # Дополнительные метрики для анализа
                trial.set_user_attr("final_metrics", final_metrics)
                trial.set_user_attr("suggested_params", suggested_params)
                
                # Сохранение информации о trial
                trial_info = {
                    "trial_number": trial.number,
                    "objective_value": objective_value,
                    "suggested_params": suggested_params,
                    "final_metrics": final_metrics,
                    "agent_type": agent_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.optimization_history.append(trial_info)
                
                self.logger.info(f"Trial {trial.number} завершен: {optimization_metric}={objective_value:.4f}")
                
                return objective_value
                
            except Exception as e:
                self.logger.error(f"Ошибка в trial {trial.number}: {e}")
                # Возвращаем плохое значение для минимизации прерывания оптимизации
                return float('-inf') if self.direction == "maximize" else float('inf')
        
        return objective
    
    def optimize(
        self,
        agent_type: str,
        n_trials: int = 100,
        training_timesteps: int = 50000,
        evaluation_episodes: int = 5,
        optimization_metric: str = "mean_reward",
        parameter_ranges: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Запуск оптимизации гиперпараметров.
        
        Args:
            agent_type: тип агента для оптимизации
            n_trials: количество попыток оптимизации
            training_timesteps: количество шагов обучения на попытку
            evaluation_episodes: количество эпизодов для оценки
            optimization_metric: метрика для оптимизации
            parameter_ranges: пользовательские диапазоны параметров
            timeout: таймаут в секундах (опционально)
            n_jobs: количество параллельных процессов
            
        Returns:
            Результаты оптимизации
        """
        self.logger.info(f"Начинаем оптимизацию {agent_type} агента")
        self.logger.info(f"Количество trials: {n_trials}")
        self.logger.info(f"Метрика оптимизации: {optimization_metric}")
        
        # Создание целевой функции
        objective = self.create_objective_function(
            agent_type=agent_type,
            training_timesteps=training_timesteps,
            evaluation_episodes=evaluation_episodes,
            optimization_metric=optimization_metric,
            parameter_ranges=parameter_ranges
        )
        
        # Запуск оптимизации
        start_time = datetime.now()
        
        try:
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
            
        except KeyboardInterrupt:
            self.logger.info("Оптимизация прервана пользователем")
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации: {e}")
        
        end_time = datetime.now()
        optimization_duration = (end_time - start_time).total_seconds()
        
        # Анализ результатов
        results = self._analyze_optimization_results(
            agent_type, optimization_duration, optimization_metric
        )
        
        # Сохранение результатов
        self._save_optimization_results(results)
        
        self.logger.info(f"Оптимизация завершена за {optimization_duration:.2f} секунд")
        
        return results
    
    def _analyze_optimization_results(
        self, 
        agent_type: str, 
        duration: float,
        optimization_metric: str
    ) -> Dict[str, Any]:
        """Анализ результатов оптимизации."""
        
        # Лучший trial
        best_trial = self.study.best_trial
        
        # Статистика по всем trials
        all_values = [trial.value for trial in self.study.trials if trial.value is not None]
        
        results = {
            "study_name": self.study_name,
            "agent_type": agent_type,
            "optimization_metric": optimization_metric,
            "direction": self.direction,
            "duration_seconds": duration,
            "n_trials": len(self.study.trials),
            "n_complete_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_failed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            
            # Лучший результат
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs
            },
            
            # Статистика по всем trials
            "all_trials_stats": {
                "mean_value": float(np.mean(all_values)) if all_values else 0,
                "std_value": float(np.std(all_values)) if all_values else 0,
                "min_value": float(np.min(all_values)) if all_values else 0,
                "max_value": float(np.max(all_values)) if all_values else 0,
                "median_value": float(np.median(all_values)) if all_values else 0
            },
            
            # Важность параметров
            "parameter_importance": {}
        }
        
        # Анализ важности параметров
        try:
            if len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 1:
                importance = optuna.importance.get_param_importances(self.study)
                results["parameter_importance"] = {k: float(v) for k, v in importance.items()}
        except Exception as e:
            self.logger.warning(f"Не удалось вычислить важность параметров: {e}")
        
        # Обновляем лучшие параметры
        self.best_params = best_trial.params
        self.best_value = best_trial.value
        
        return results
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """Сохранение результатов оптимизации."""
        
        # Сохраняем основные результаты
        results_path = self.experiment_manager.results_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Сохраняем историю всех trials
        history_path = self.experiment_manager.results_dir / "optimization_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.optimization_history, f, indent=2, default=str)
        
        # Сохраняем study объект Optuna
        study_path = self.experiment_manager.results_dir / "optuna_study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        self.logger.info(f"Результаты оптимизации сохранены в {self.experiment_manager.results_dir}")
    
    def get_best_config(self) -> Optional[DRLConfig]:
        """
        Получение лучшей конфигурации DRL.
        
        Returns:
            Лучшая найденная конфигурация или None
        """
        if self.best_params is None:
            return None
        
        # Создаем конфигурацию с лучшими параметрами
        best_config_dict = {
            **self.base_drl_config.__dict__,
            **self.best_params
        }
        
        return DRLConfig(**best_config_dict)
    
    def train_best_model(
        self,
        agent_type: str,
        training_timesteps: int = 200000,
        experiment_name: str = "best_tuned_model"
    ) -> Any:
        """
        Обучение модели с лучшими найденными гиперпараметрами.
        
        Args:
            agent_type: тип агента
            training_timesteps: количество шагов для финального обучения
            experiment_name: имя эксперимента
            
        Returns:
            Обученный агент
        """
        if self.best_params is None:
            raise ValueError("Оптимизация не была проведена или не найдены лучшие параметры")
        
        self.logger.info(f"Обучение лучшей модели {agent_type} с найденными параметрами")
        
        # Создание конфигураций
        best_drl_config = self.get_best_config()
        best_drl_config.total_timesteps = training_timesteps
        best_drl_config.verbose = 1  # Включаем verbose для финального обучения
        
        trading_config = TradingConfig(**self.base_trading_config.__dict__)
        
        # Lazy imports to avoid circular dependencies
        from ..environments import TradingEnv
        from ..agents import PPOAgent, SACAgent, DQNAgent, A2CAgent
        from ..training import Trainer, ExperimentManager
        
        # Создание компонентов
        env = TradingEnv(trading_config, logger=self.logger)
        
        if agent_type.upper() == "PPO":
            agent = PPOAgent(best_drl_config, trading_config, self.logger)
        elif agent_type.upper() == "SAC":
            agent = SACAgent(best_drl_config, trading_config, self.logger)
        elif agent_type.upper() == "DQN":
            agent = DQNAgent(best_drl_config, trading_config, self.logger)
        elif agent_type.upper() == "A2C":
            agent = A2CAgent(best_drl_config, trading_config, self.logger)
        else:
            raise ValueError(f"Неподдерживаемый тип агента: {agent_type}")
        
        # Создание тренера
        best_experiment_manager = ExperimentManager(
            base_dir=self.experiment_manager.experiment_dir / "best_model",
            experiment_name=experiment_name
        )
        
        trainer = Trainer(
            agent=agent,
            env=env,
            drl_config=best_drl_config,
            trading_config=trading_config,
            experiment_manager=best_experiment_manager,
            logger=self.logger
        )
        
        # Обучение
        trained_agent = trainer.train(
            total_timesteps=training_timesteps,
            eval_freq=training_timesteps // 10,
            save_freq=training_timesteps // 5,
            n_eval_episodes=10
        )
        
        # Сохранение информации о лучшей модели
        best_model_info = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "final_training_timesteps": training_timesteps,
            "optimization_study": self.study_name
        }
        
        best_experiment_manager.save_results("best_model_info", best_model_info)
        
        self.logger.info(f"Лучшая модель обучена и сохранена")
        
        return trained_agent
    
    def generate_optimization_report(self) -> str:
        """
        Генерация отчета об оптимизации.
        
        Returns:
            Текстовый отчет
        """
        if not self.optimization_history:
            return "Оптимизация не была проведена"
        
        report_lines = [
            "=" * 60,
            "ОТЧЕТ ОБ ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ",
            "=" * 60,
            f"Исследование: {self.study_name}",
            f"Всего trials: {len(self.optimization_history)}",
            ""
        ]
        
        if self.best_params and self.best_value:
            report_lines.extend([
                "ЛУЧШИЕ ПАРАМЕТРЫ:",
                "-" * 20
            ])
            
            for param, value in self.best_params.items():
                report_lines.append(f"{param}: {value}")
            
            report_lines.extend([
                "",
                f"Лучшее значение: {self.best_value:.6f}",
                ""
            ])
        
        # Статистика по trials
        all_values = [trial["objective_value"] for trial in self.optimization_history 
                     if trial["objective_value"] is not None]
        
        if all_values:
            report_lines.extend([
                "СТАТИСТИКА TRIALS:",
                "-" * 20,
                f"Среднее значение: {np.mean(all_values):.6f}",
                f"Стандартное отклонение: {np.std(all_values):.6f}",
                f"Минимум: {np.min(all_values):.6f}",
                f"Максимум: {np.max(all_values):.6f}",
                f"Медиана: {np.median(all_values):.6f}",
                ""
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)