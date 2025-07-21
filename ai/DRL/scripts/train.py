"""Главный скрипт для обучения DRL агентов."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from CryptoTrade.ai.DRL.config import DRLConfig, TradingConfig
from CryptoTrade.ai.DRL.environments import TradingEnv
from CryptoTrade.ai.DRL.agents import PPOAgent, SACAgent, DQNAgent, A2CAgent
from CryptoTrade.ai.DRL.training import Trainer, ExperimentManager
from CryptoTrade.ai.DRL.utils import DRLLogger


def create_configs_from_args(args) -> tuple[DRLConfig, TradingConfig]:
    """Создание конфигураций на основе аргументов командной строки."""
    
    # Торговая конфигурация
    trading_config = TradingConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        exchange=args.exchange,
        initial_balance=args.initial_balance,
        commission_rate=args.commission_rate,
        max_risk_per_trade=args.max_risk,
        reward_scheme=args.reward_scheme,
        action_type=args.action_type,
        lookback_window=args.lookback_window,
        max_episode_steps=args.max_episode_steps
    )
    
    # DRL конфигурация
    drl_config = DRLConfig(
        agent_type=args.agent_type,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        net_arch=[int(x) for x in args.net_arch.split(',') if x.strip()],
        activation_fn=args.activation_fn,
        use_lstm=args.use_lstm,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.tensorboard,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir
    )
    
    # Специфичные параметры для разных агентов
    if args.agent_type in ["PPO", "A2C"]:
        drl_config.n_steps = args.n_steps
        drl_config.n_epochs = getattr(args, 'n_epochs', 10)
        drl_config.clip_range = getattr(args, 'clip_range', 0.2)
        drl_config.ent_coef = args.ent_coef
        drl_config.vf_coef = args.vf_coef
        drl_config.max_grad_norm = args.max_grad_norm
        drl_config.gae_lambda = args.gae_lambda
    
    elif args.agent_type == "SAC":
        drl_config.buffer_size = args.buffer_size
        drl_config.tau = args.tau
        drl_config.alpha = args.alpha
        drl_config.target_entropy = args.target_entropy
    
    elif args.agent_type == "DQN":
        drl_config.buffer_size = args.buffer_size
        drl_config.tau = args.tau
        drl_config.exploration_fraction = args.exploration_fraction
        drl_config.exploration_initial_eps = args.exploration_initial_eps
        drl_config.exploration_final_eps = args.exploration_final_eps
        drl_config.target_update_interval = args.target_update_interval
    
    return drl_config, trading_config


def create_agent(agent_type: str, drl_config: DRLConfig, trading_config: TradingConfig, logger: DRLLogger):
    """Создание агента на основе типа."""
    
    if agent_type.upper() == "PPO":
        return PPOAgent(drl_config, trading_config, logger)
    elif agent_type.upper() == "SAC":
        return SACAgent(drl_config, trading_config, logger)
    elif agent_type.upper() == "DQN":
        return DQNAgent(drl_config, trading_config, logger)
    elif agent_type.upper() == "A2C":
        return A2CAgent(drl_config, trading_config, logger)
    else:
        raise ValueError(f"Неподдерживаемый тип агента: {agent_type}")


def main():
    """Главная функция обучения."""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Обучение DRL агентов для торговли криптовалютами")
    
    # Основные параметры
    parser.add_argument("--agent_type", type=str, default="PPO", choices=["PPO", "SAC", "DQN", "A2C"],
                       help="Тип агента для обучения")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", 
                       help="Торговая пара")
    parser.add_argument("--timeframe", type=str, default="1d", choices=["1h", "4h", "1d", "1w"],
                       help="Таймфрейм данных")
    parser.add_argument("--exchange", type=str, default="binance",
                       help="Биржа")
    
    # Параметры обучения
    parser.add_argument("--total_timesteps", type=int, default=500000,
                       help="Общее количество шагов обучения")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Скорость обучения")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Размер батча")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Дисконт фактор")
    
    # Архитектура сети
    parser.add_argument("--net_arch", type=str, default="64,64",
                       help="Архитектура сети (разделенная запятыми)")
    parser.add_argument("--activation_fn", type=str, default="relu", 
                       choices=["relu", "tanh", "elu", "leaky_relu", "swish"],
                       help="Функция активации")
    parser.add_argument("--use_lstm", action="store_true",
                       help="Использовать LSTM слои")
    
    # PPO/A2C параметры
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="PPO/A2C: количество шагов на обновление")
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="PPO: количество эпох на обновление")
    parser.add_argument("--clip_range", type=float, default=0.2,
                       help="PPO: диапазон обрезки")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                       help="PPO/A2C: коэффициент энтропии")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                       help="PPO/A2C: коэффициент value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="PPO/A2C: максимальная норма градиента")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="PPO/A2C: лямбда для GAE")
    
    # SAC/DQN параметры
    parser.add_argument("--buffer_size", type=int, default=100000,
                       help="SAC/DQN: размер буфера опыта")
    parser.add_argument("--tau", type=float, default=0.005,
                       help="SAC/DQN: коэффициент мягкого обновления")
    parser.add_argument("--alpha", type=float, default=0.2,
                       help="SAC: коэффициент энтропии")
    parser.add_argument("--target_entropy", type=str, default="auto",
                       help="SAC: целевая энтропия")
    
    # DQN параметры
    parser.add_argument("--exploration_fraction", type=float, default=0.1,
                       help="DQN: доля времени на исследование")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0,
                       help="DQN: начальный epsilon")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05,
                       help="DQN: финальный epsilon")
    parser.add_argument("--target_update_interval", type=int, default=1000,
                       help="DQN: интервал обновления target сети")
    
    # Торговые параметры
    parser.add_argument("--initial_balance", type=float, default=10000.0,
                       help="Начальный баланс")
    parser.add_argument("--commission_rate", type=float, default=0.001,
                       help="Комиссия торговли")
    parser.add_argument("--max_risk", type=float, default=0.02,
                       help="Максимальный риск на сделку")
    parser.add_argument("--reward_scheme", type=str, default="profit_based",
                       choices=["profit_based", "sharpe_based", "risk_adjusted", "differential", "multi_objective"],
                       help="Схема расчета наград")
    parser.add_argument("--action_type", type=str, default="continuous",
                       choices=["continuous", "discrete"],
                       help="Тип пространства действий")
    parser.add_argument("--lookback_window", type=int, default=50,
                       help="Размер окна наблюдения")
    parser.add_argument("--max_episode_steps", type=int, default=1000,
                       help="Максимальное количество шагов в эпизоде")
    
    # Оценка и сохранение
    parser.add_argument("--eval_freq", type=int, default=10000,
                       help="Частота оценки модели")
    parser.add_argument("--save_freq", type=int, default=50000,
                       help="Частота сохранения модели")
    parser.add_argument("--n_eval_episodes", type=int, default=5,
                       help="Количество эпизодов для оценки")
    
    # Система
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="Устройство для вычислений")
    parser.add_argument("--seed", type=int, default=42,
                       help="Сид для воспроизводимости")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                       help="Уровень детализации вывода")
    
    # Директории
    parser.add_argument("--models_dir", type=str, default="CryptoTrade/ai/DRL/models",
                       help="Директория для сохранения моделей")
    parser.add_argument("--logs_dir", type=str, default="CryptoTrade/ai/DRL/logs",
                       help="Директория для логов")
    parser.add_argument("--data_dir", type=str, default="CryptoTrade/data",
                       help="Директория с данными")
    
    # Дополнительные опции
    parser.add_argument("--tensorboard", action="store_true", default=True,
                       help="Использовать TensorBoard для логирования")
    parser.add_argument("--resume", action="store_true",
                       help="Продолжить обучение с последнего checkpoint")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Имя эксперимента (опционально)")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Путь к файлу конфигурации YAML (опционально)")
    
    args = parser.parse_args()
    
    # Загрузка конфигурации из файла если указан
    if args.config_file and os.path.exists(args.config_file):
        import yaml
        with open(args.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Обновляем аргументы значениями из файла
        for key, value in config_data.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Создание конфигураций
    drl_config, trading_config = create_configs_from_args(args)
    
    # Создание логгера
    logger = DRLLogger(f"trainer_{args.agent_type.lower()}", drl_config.log_level, drl_config.logs_dir)
    
    logger.info("="*60)
    logger.info("НАЧАЛО ОБУЧЕНИЯ DRL АГЕНТА")
    logger.info("="*60)
    logger.info(f"Агент: {args.agent_type}")
    logger.info(f"Символ: {args.symbol}")
    logger.info(f"Таймфрейм: {args.timeframe}")
    logger.info(f"Общие шаги: {args.total_timesteps:,}")
    logger.info(f"Устройство: {args.device}")
    logger.info("="*60)
    
    try:
        # Создание менеджера экспериментов
        experiment_manager = ExperimentManager(
            base_dir=drl_config.models_dir,
            experiment_name=args.experiment_name
        )
        
        # Создание среды
        logger.info("Создание торговой среды...")
        env = TradingEnv(trading_config, logger=logger)
        
        # Создание агента
        logger.info(f"Создание {args.agent_type} агента...")
        agent = create_agent(args.agent_type, drl_config, trading_config, logger)
        
        # Создание тренера
        trainer = Trainer(
            agent=agent,
            env=env,
            drl_config=drl_config,
            trading_config=trading_config,
            experiment_manager=experiment_manager,
            logger=logger
        )
        
        # Проверка на продолжение обучения
        if args.resume:
            logger.info("Поиск checkpoint для продолжения обучения...")
            if trainer.load_latest_checkpoint():
                logger.info("Checkpoint найден, продолжаем обучение")
            else:
                logger.info("Checkpoint не найден, начинаем новое обучение")
        
        # Запуск обучения
        start_time = time.time()
        
        trained_agent = trainer.train(
            total_timesteps=drl_config.total_timesteps,
            eval_freq=drl_config.eval_freq,
            save_freq=drl_config.save_freq,
            n_eval_episodes=args.n_eval_episodes
        )
        
        training_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info(f"Время обучения: {training_time:.2f} секунд ({training_time/60:.1f} минут)")
        logger.info(f"Модель сохранена в: {experiment_manager.experiment_dir}")
        
        # Финальная оценка
        logger.info("Проведение финальной оценки...")
        final_metrics = trainer.evaluate(n_episodes=10)
        
        logger.info("Финальные метрики:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Сохранение финальной модели
        final_model_path = trainer.save_final_model()
        logger.info(f"Финальная модель сохранена: {final_model_path}")
        
        logger.info("="*60)
        
        return trained_agent
    
    except KeyboardInterrupt:
        logger.info("Обучение прервано пользователем")
        return None
    
    except Exception as e:
        logger.error(f"Ошибка во время обучения: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def quick_train(
    symbol: str = "BTCUSDT",
    timeframe: str = "1d", 
    agent_type: str = "PPO",
    total_timesteps: int = 100000,
    **kwargs
):
    """
    Быстрый запуск обучения с минимальными настройками.
    
    Args:
        symbol: торговая пара
        timeframe: таймфрейм
        agent_type: тип агента
        total_timesteps: количество шагов
        **kwargs: дополнительные параметры
    """
    
    # Создание конфигураций
    trading_config = TradingConfig(
        symbol=symbol,
        timeframe=timeframe,
        **{k: v for k, v in kwargs.items() if hasattr(TradingConfig, k)}
    )
    
    drl_config = DRLConfig(
        agent_type=agent_type,
        total_timesteps=total_timesteps,
        **{k: v for k, v in kwargs.items() if hasattr(DRLConfig, k)}
    )
    
    # Создание компонентов
    logger = DRLLogger(f"quick_train_{agent_type.lower()}")
    env = TradingEnv(trading_config, logger=logger)
    agent = create_agent(agent_type, drl_config, trading_config, logger)
    
    experiment_manager = ExperimentManager()
    
    trainer = Trainer(
        agent=agent,
        env=env, 
        drl_config=drl_config,
        trading_config=trading_config,
        experiment_manager=experiment_manager,
        logger=logger
    )
    
    logger.info(f"Быстрое обучение: {agent_type} на {symbol} {timeframe}")
    
    return trainer.train(total_timesteps=total_timesteps)


if __name__ == "__main__":
    main()