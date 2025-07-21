"""Базовый пример обучения DRL агента."""

import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from CryptoTrade.ai.DRL.config import DRLConfig, TradingConfig
from CryptoTrade.ai.DRL.environments import TradingEnv
from CryptoTrade.ai.DRL.agents import PPOAgent
from CryptoTrade.ai.DRL.training import Trainer, ExperimentManager
from CryptoTrade.ai.DRL.utils import DRLLogger


def basic_ppo_training():
    """Базовый пример обучения PPO агента."""
    
    print("=" * 60)
    print("БАЗОВЫЙ ПРИМЕР ОБУЧЕНИЯ PPO АГЕНТА")
    print("=" * 60)
    
    # Конфигурация торговли
    trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1d",
        exchange="binance",
        initial_balance=10000.0,
        commission_rate=0.001,
        reward_scheme="profit_based",
        action_type="continuous",
        lookback_window=20,
        max_episode_steps=500
    )
    
    # Конфигурация DRL
    drl_config = DRLConfig(
        agent_type="PPO",
        total_timesteps=50000,  # Небольшое количество для демонстрации
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,
        net_arch=[32, 32],  # Простая архитектура
        activation_fn="relu",
        eval_freq=5000,
        save_freq=10000,
        verbose=1,
        tensorboard_log=True
    )
    
    # Создание логгера
    logger = DRLLogger("basic_example")
    
    # Создание менеджера экспериментов
    experiment_manager = ExperimentManager(experiment_name="basic_ppo_example")
    
    try:
        # Создание среды
        logger.info("Создание торговой среды...")
        env = TradingEnv(trading_config, logger=logger)
        
        # Создание агента
        logger.info("Создание PPO агента...")
        agent = PPOAgent(drl_config, trading_config, logger)
        
        # Создание тренера
        trainer = Trainer(
            agent=agent,
            env=env,
            drl_config=drl_config,
            trading_config=trading_config,
            experiment_manager=experiment_manager,
            logger=logger
        )
        
        # Обучение
        logger.info("Начинаем обучение...")
        trained_agent = trainer.train(
            total_timesteps=drl_config.total_timesteps,
            eval_freq=drl_config.eval_freq,
            save_freq=drl_config.save_freq,
            n_eval_episodes=3
        )
        
        # Финальная оценка
        logger.info("Проведение финальной оценки...")
        final_metrics = trainer.evaluate(n_episodes=5)
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("=" * 60)
        print(f"Эксперимент: {experiment_manager.experiment_name}")
        
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Сохранение финальной модели
        final_model_path = trainer.save_final_model()
        print(f"\nМодель сохранена: {final_model_path}")
        
        print("=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)
        
        return trained_agent
        
    except Exception as e:
        logger.error(f"Ошибка в базовом примере: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_test_all_agents():
    """Быстрое тестирование всех типов агентов."""
    
    print("=" * 60)
    print("БЫСТРОЕ ТЕСТИРОВАНИЕ ВСЕХ АГЕНТОВ")
    print("=" * 60)
    
    # Общие конфигурации
    trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        max_episode_steps=200
    )
    
    base_drl_config = {
        'total_timesteps': 5000,  # Очень мало для быстрого теста
        'learning_rate': 3e-4,
        'batch_size': 32,
        'net_arch': [16, 16],  # Маленькая сеть
        'eval_freq': 2000,
        'save_freq': 5000,
        'verbose': 1
    }
    
    agents_to_test = [
        ("PPO", "continuous"),
        ("SAC", "continuous"), 
        ("DQN", "discrete")
    ]
    
    results = {}
    logger = DRLLogger("quick_test")
    
    for agent_type, action_type in agents_to_test:
        try:
            print(f"\nТестирование {agent_type} агента...")
            
            # Обновляем конфигурации
            trading_config.action_type = action_type
            drl_config = DRLConfig(agent_type=agent_type, **base_drl_config)
            
            # Создание компонентов
            env = TradingEnv(trading_config, logger=logger)
            
            if agent_type == "PPO":
                from CryptoTrade.ai.DRL.agents import PPOAgent
                agent = PPOAgent(drl_config, trading_config, logger)
            elif agent_type == "SAC":
                from CryptoTrade.ai.DRL.agents import SACAgent
                agent = SACAgent(drl_config, trading_config, logger)
            elif agent_type == "DQN":
                from CryptoTrade.ai.DRL.agents import DQNAgent
                agent = DQNAgent(drl_config, trading_config, logger)
            
            experiment_manager = ExperimentManager(experiment_name=f"quick_test_{agent_type.lower()}")
            
            trainer = Trainer(
                agent=agent,
                env=env,
                drl_config=drl_config,
                trading_config=trading_config,
                experiment_manager=experiment_manager,
                logger=logger
            )
            
            # Быстрое обучение
            trainer.train(
                total_timesteps=drl_config.total_timesteps,
                eval_freq=drl_config.eval_freq,
                n_eval_episodes=2
            )
            
            # Оценка
            metrics = trainer.evaluate(n_episodes=3)
            results[agent_type] = {
                'success': True,
                'mean_reward': metrics.get('mean_reward', 0),
                'total_return': metrics.get('total_return', 0)
            }
            
            print(f"✅ {agent_type} успешно обучен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка с {agent_type}: {e}")
            results[agent_type] = {
                'success': False,
                'error': str(e)
            }
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    for agent_type, result in results.items():
        if result['success']:
            print(f"✅ {agent_type}: Награда={result['mean_reward']:.4f}, "
                  f"Доходность={result['total_return']*100:.2f}%")
        else:
            print(f"❌ {agent_type}: {result['error']}")
    
    return results


def demonstrate_model_loading():
    """Демонстрация загрузки и использования сохраненной модели."""
    
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ЗАГРУЗКИ МОДЕЛИ")
    print("=" * 60)
    
    # Сначала обучаем простую модель
    logger = DRLLogger("model_demo")
    
    trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1d",
        max_episode_steps=100
    )
    
    drl_config = DRLConfig(
        agent_type="PPO",
        total_timesteps=3000,
        net_arch=[16, 16],
        verbose=1
    )
    
    try:
        # Обучение
        env = TradingEnv(trading_config, logger=logger)
        agent = PPOAgent(drl_config, trading_config, logger)
        
        experiment_manager = ExperimentManager(experiment_name="model_demo")
        trainer = Trainer(agent, env, drl_config, trading_config, experiment_manager, logger)
        
        logger.info("Обучение модели для демонстрации...")
        trainer.train(total_timesteps=drl_config.total_timesteps)
        
        # Сохранение
        model_path = trainer.save_final_model("demo_model")
        logger.info(f"Модель сохранена: {model_path}")
        
        # Создание нового агента и загрузка
        logger.info("Создание нового агента и загрузка модели...")
        new_agent = PPOAgent(drl_config, trading_config, logger)
        new_agent.load(model_path, env)
        
        # Тестирование загруженной модели
        logger.info("Тестирование загруженной модели...")
        metrics = new_agent.evaluate(env, n_episodes=3)
        
        print(f"\nРезультаты загруженной модели:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("✅ Демонстрация загрузки модели завершена успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации загрузки: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Выберите пример для запуска:")
    print("1. Базовое обучение PPO")
    print("2. Быстрое тестирование всех агентов")
    print("3. Демонстрация загрузки модели")
    print("4. Запустить все примеры")
    
    choice = input("\nВведите номер (1-4): ").strip()
    
    if choice == "1":
        basic_ppo_training()
    elif choice == "2":
        quick_test_all_agents()
    elif choice == "3":
        demonstrate_model_loading()
    elif choice == "4":
        print("Запуск всех примеров...\n")
        basic_ppo_training()
        print("\n" + "="*60 + "\n")
        quick_test_all_agents()
        print("\n" + "="*60 + "\n")
        demonstrate_model_loading()
    else:
        print("Запуск базового примера по умолчанию...")
        basic_ppo_training()