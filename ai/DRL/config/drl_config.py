"""Основные настройки DRL системы."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DRLConfig:
    """Конфигурация для DRL системы."""
    
    # Основные параметры
    agent_type: str = "PPO"  # PPO, DQN, SAC, A2C
    total_timesteps: int = 500000  # Общее количество шагов обучения
    learning_rate: float = 3e-4  # Скорость обучения
    
    # Архитектура сети
    net_arch: List[int] = field(default_factory=lambda: [64, 64])  # Архитектура нейронной сети
    activation_fn: str = "relu"  # relu, tanh, elu
    use_lstm: bool = False  # Использовать LSTM слои
    lstm_hidden_size: int = 128  # Размер скрытого слоя LSTM
    
    # Параметры обучения
    batch_size: int = 256  # Размер батча
    buffer_size: int = 100000  # Размер буфера опыта (для off-policy алгоритмов)
    gamma: float = 0.99  # Дисконт фактор
    tau: float = 0.005  # Коэффициент мягкого обновления target сети
    
    # PPO специфичные параметры
    n_steps: int = 2048  # Количество шагов на обновление
    n_epochs: int = 10  # Количество эпох на обновление
    clip_range: float = 0.2  # Диапазон обрезки PPO
    ent_coef: float = 0.01  # Коэффициент энтропии
    vf_coef: float = 0.5  # Коэффициент value function
    max_grad_norm: float = 0.5  # Максимальная норма градиента
    
    # DQN специфичные параметры
    exploration_fraction: float = 0.1  # Доля времени на исследование
    exploration_final_eps: float = 0.05  # Финальный epsilon для ε-greedy
    exploration_initial_eps: float = 1.0  # Начальный epsilon для ε-greedy
    target_update_interval: int = 1000  # Интервал обновления target сети
    
    # SAC специфичные параметры
    alpha: float = 0.2  # Коэффициент энтропии для SAC
    target_entropy: str = "auto"  # Целевая энтропия
    use_sde: bool = False  # Использовать State Dependent Exploration
    
    # Оценка и сохранение
    eval_freq: int = 10000  # Частота оценки модели
    save_freq: int = 50000  # Частота сохранения модели
    n_eval_episodes: int = 5  # Количество эпизодов для оценки
    eval_env_seed: int = 42  # Сид для среды оценки
    
    # Логирование
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    tensorboard_log: bool = True  # Использовать TensorBoard
    verbose: int = 1  # Уровень детализации вывода
    
    # Пути к файлам
    models_dir: str = "CryptoTrade/ai/DRL/models"  # Директория для сохранения моделей
    logs_dir: str = "CryptoTrade/ai/DRL/logs"  # Директория для логов
    data_dir: str = "CryptoTrade/data"  # Директория с данными
    
    # Воспроизводимость
    seed: int = 42  # Сид для воспроизводимости
    deterministic: bool = True  # Детерминированное выполнение
    
    # Оптимизация
    device: str = "auto"  # cuda, cpu, auto
    normalize_advantage: bool = True  # Нормализация преимущества
    use_gae: bool = True  # Использовать Generalized Advantage Estimation
    gae_lambda: float = 0.95  # Лямбда для GAE
    
    def __post_init__(self):
        """Валидация конфигурации после инициализации."""
        if self.agent_type not in ["PPO", "DQN", "SAC", "A2C"]:
            raise ValueError(f"Неподдерживаемый тип агента: {self.agent_type}")
        
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps должен быть положительным")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным")
        
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma должен быть в диапазоне (0, 1]")
    
    def get_agent_params(self) -> Dict[str, Union[int, float, str, bool]]:
        """Получить параметры для создания агента."""
        base_params = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'verbose': self.verbose,
            'seed': self.seed,
            'device': self.device,
        }
        
        if self.agent_type == "PPO":
            base_params.update({
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'clip_range': self.clip_range,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
                'use_sde': self.use_sde,
                'gae_lambda': self.gae_lambda,
            })
        elif self.agent_type == "DQN":
            base_params.update({
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'exploration_fraction': self.exploration_fraction,
                'exploration_final_eps': self.exploration_final_eps,
                'exploration_initial_eps': self.exploration_initial_eps,
                'target_update_interval': self.target_update_interval,
                'tau': self.tau,
            })
        elif self.agent_type == "SAC":
            base_params.update({
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'tau': self.tau,
                'ent_coef': self.alpha,
                'target_entropy': self.target_entropy,
                'use_sde': self.use_sde,
            })
        
        return base_params