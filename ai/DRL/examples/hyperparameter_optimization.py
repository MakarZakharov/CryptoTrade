"""Пример автоматической оптимизации гиперпараметров для DRL агентов."""

import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from CryptoTrade.ai.DRL.config import DRLConfig, TradingConfig
from CryptoTrade.ai.DRL.utils import HyperparameterTuner, DRLLogger


def optimize_ppo_agent():
    """Пример оптимизации PPO агента."""
    
    print("=" * 60)
    print("ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ PPO АГЕНТА")
    print("=" * 60)
    
    # Базовые конфигурации
    base_trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        reward_scheme="profit_based",
        action_type="continuous",
        lookback_window=20,
        max_episode_steps=300  # Короткие эпизоды для быстрой оптимизации
    )
    
    base_drl_config = DRLConfig(
        agent_type="PPO",
        total_timesteps=20000,  # Небольшое количество для демонстрации
        verbose=0,
        tensorboard_log=False,  # Отключаем для ускорения
        eval_freq=10000,
        save_freq=20000
    )
    
    # Создание тюнера
    logger = DRLLogger("ppo_optimization")
    tuner = HyperparameterTuner(
        base_drl_config=base_drl_config,
        base_trading_config=base_trading_config,
        study_name="ppo_trading_optimization",
        direction="maximize"
    )
    
    # Пользовательские диапазоны параметров (опционально)
    custom_ranges = {
        "reward_scaling": {
            "type": "float",
            "low": 0.1,
            "high": 10.0,
            "log": True
        },
        "lookback_window": {
            "type": "categorical",
            "choices": [10, 20, 30, 50]
        }
    }
    
    try:
        # Запуск оптимизации
        results = tuner.optimize(
            agent_type="PPO",
            n_trials=10,  # Небольшое количество для демонстрации
            training_timesteps=20000,
            evaluation_episodes=3,
            optimization_metric="mean_reward",
            parameter_ranges=custom_ranges,
            timeout=1800,  # 30 минут максимум
            n_jobs=1  # Последовательное выполнение
        )
        
        # Вывод результатов
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print("=" * 60)
        
        best_trial = results["best_trial"]
        print(f"Лучший trial: #{best_trial['number']}")
        print(f"Лучшее значение: {best_trial['value']:.4f}")
        
        print("\nЛучшие параметры:")
        for param, value in best_trial["params"].items():
            print(f"  {param}: {value}")
        
        # Статистика по всем trials
        stats = results["all_trials_stats"]
        print(f"\nСтатистика по всем trials:")
        print(f"  Среднее значение: {stats['mean_value']:.4f}")
        print(f"  Стандартное отклонение: {stats['std_value']:.4f}")
        print(f"  Лучший результат: {stats['max_value']:.4f}")
        
        # Важность параметров
        if results["parameter_importance"]:
            print(f"\nВажность параметров:")
            importance_sorted = sorted(
                results["parameter_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for param, importance in importance_sorted[:5]:  # Топ 5
                print(f"  {param}: {importance:.4f}")
        
        # Обучение финальной модели с лучшими параметрами
        print(f"\n" + "=" * 60)
        print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
        print("=" * 60)
        
        final_agent = tuner.train_best_model(
            agent_type="PPO",
            training_timesteps=50000,  # Больше шагов для финальной модели
            experiment_name="optimized_ppo_final"
        )
        
        print("✅ Оптимизация и обучение финальной модели завершены!")
        
        # Генерация отчета
        report = tuner.generate_optimization_report()
        print(f"\n{report}")
        
        return tuner, results
        
    except Exception as e:
        logger.error(f"Ошибка в оптимизации: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_agents_optimization():
    """Сравнение оптимизации разных типов агентов."""
    
    print("=" * 60)
    print("СРАВНЕНИЕ ОПТИМИЗАЦИИ РАЗНЫХ АГЕНТОВ")
    print("=" * 60)
    
    # Общие конфигурации
    trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        max_episode_steps=200
    )
    
    base_drl_config = DRLConfig(
        total_timesteps=15000,
        verbose=0
    )
    
    agents_to_compare = ["PPO", "SAC", "A2C"]
    results_comparison = {}
    
    logger = DRLLogger("agents_comparison")
    
    for agent_type in agents_to_compare:
        try:
            print(f"\n--- Оптимизация {agent_type} агента ---")
            
            # Настройка для continuous/discrete в зависимости от агента
            if agent_type in ["PPO", "SAC", "A2C"]:
                trading_config.action_type = "continuous"
            else:
                trading_config.action_type = "discrete"
            
            # Создание тюнера для каждого агента
            tuner = HyperparameterTuner(
                base_drl_config=base_drl_config,
                base_trading_config=trading_config,
                study_name=f"{agent_type.lower()}_comparison",
                direction="maximize"
            )
            
            # Быстрая оптимизация
            results = tuner.optimize(
                agent_type=agent_type,
                n_trials=5,  # Мало trials для быстрого сравнения
                training_timesteps=15000,
                evaluation_episodes=3,
                optimization_metric="mean_reward",
                timeout=600  # 10 минут на агента
            )
            
            results_comparison[agent_type] = {
                "best_value": results["best_trial"]["value"],
                "best_params": results["best_trial"]["params"],
                "mean_value": results["all_trials_stats"]["mean_value"],
                "std_value": results["all_trials_stats"]["std_value"]
            }
            
            print(f"✅ {agent_type}: лучший результат = {results['best_trial']['value']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка с {agent_type}: {e}")
            results_comparison[agent_type] = {"error": str(e)}
    
    # Вывод сравнения
    print(f"\n" + "=" * 60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    successful_results = {k: v for k, v in results_comparison.items() if "error" not in v}
    
    if successful_results:
        # Сортируем по лучшему результату
        sorted_results = sorted(
            successful_results.items(),
            key=lambda x: x[1]["best_value"],
            reverse=True
        )
        
        print("Рейтинг агентов по лучшему результату:")
        for i, (agent, result) in enumerate(sorted_results, 1):
            print(f"{i}. {agent}: {result['best_value']:.4f} "
                  f"(среднее: {result['mean_value']:.4f}±{result['std_value']:.4f})")
        
        # Рекомендация
        best_agent = sorted_results[0][0]
        print(f"\n🏆 Рекомендуемый агент: {best_agent}")
    
    else:
        print("Не удалось успешно оптимизировать ни одного агента")
    
    return results_comparison


def advanced_optimization_example():
    """Продвинутый пример оптимизации с пользовательскими параметрами."""
    
    print("=" * 60)
    print("ПРОДВИНУТАЯ ОПТИМИЗАЦИЯ С ПОЛЬЗОВАТЕЛЬСКИМИ ПАРАМЕТРАМИ")
    print("=" * 60)
    
    # Торговая конфигурация с дополнительными параметрами для оптимизации
    trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        reward_scheme="risk_adjusted",  # Более сложная схема наград
        action_type="continuous",
        lookback_window=30,
        max_episode_steps=400
    )
    
    drl_config = DRLConfig(
        agent_type="SAC",
        total_timesteps=25000,
        verbose=0
    )
    
    # Расширенные диапазоны параметров
    advanced_ranges = {
        # Торговые параметры
        "commission_rate": {
            "type": "float",
            "low": 0.001,
            "high": 0.01
        },
        "max_risk_per_trade": {
            "type": "float",
            "low": 0.01,
            "high": 0.05
        },
        "lookback_window": {
            "type": "categorical",
            "choices": [20, 30, 50, 80]
        },
        
        # Параметры наград
        "reward_scaling": {
            "type": "float",
            "low": 0.1,
            "high": 5.0,
            "log": True
        },
        
        # Архитектурные параметры
        "use_lstm": {
            "type": "categorical", 
            "choices": [True, False]
        },
        
        # SAC специфичные параметры
        "train_freq": {
            "type": "categorical",
            "choices": [1, 4, 8, 16]
        },
        "gradient_steps": {
            "type": "categorical",
            "choices": [1, 2, 4]
        }
    }
    
    logger = DRLLogger("advanced_optimization")
    
    tuner = HyperparameterTuner(
        base_drl_config=drl_config,
        base_trading_config=trading_config,
        study_name="advanced_sac_optimization",
        direction="maximize"
    )
    
    try:
        # Запуск продвинутой оптимизации
        results = tuner.optimize(
            agent_type="SAC",
            n_trials=15,
            training_timesteps=25000,
            evaluation_episodes=5,
            optimization_metric="total_return",  # Оптимизируем по общей доходности
            parameter_ranges=advanced_ranges,
            timeout=2400,  # 40 минут
            n_jobs=1
        )
        
        print("\n🎯 Продвинутая оптимизация завершена!")
        
        # Детальный анализ
        best_trial = results["best_trial"]
        print(f"\nЛучший результат: {best_trial['value']:.4f}")
        
        print(f"\nОптимальные параметры:")
        
        # Группируем параметры по категориям
        ml_params = {}
        trading_params = {}
        architecture_params = {}
        
        for param, value in best_trial["params"].items():
            if param in ["learning_rate", "batch_size", "gamma", "tau", "alpha"]:
                ml_params[param] = value
            elif param in ["commission_rate", "max_risk_per_trade", "lookback_window"]:
                trading_params[param] = value
            elif param in ["net_arch_size", "net_arch_layers", "activation_fn", "use_lstm"]:
                architecture_params[param] = value
            else:
                ml_params[param] = value
        
        if ml_params:
            print("\n  ML параметры:")
            for param, value in ml_params.items():
                print(f"    {param}: {value}")
        
        if trading_params:
            print("\n  Торговые параметры:")
            for param, value in trading_params.items():
                print(f"    {param}: {value}")
        
        if architecture_params:
            print("\n  Архитектурные параметры:")
            for param, value in architecture_params.items():
                print(f"    {param}: {value}")
        
        # Важность параметров
        if results["parameter_importance"]:
            print(f"\n📊 Важность параметров (топ-10):")
            importance_sorted = sorted(
                results["parameter_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for param, importance in importance_sorted[:10]:
                print(f"  {param}: {importance:.3f}")
        
        return tuner, results
        
    except Exception as e:
        logger.error(f"Ошибка в продвинутой оптимизации: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("Выберите тип оптимизации:")
    print("1. Базовая оптимизация PPO агента")
    print("2. Сравнение оптимизации разных агентов")
    print("3. Продвинутая оптимизация с пользовательскими параметрами")
    print("4. Запустить все примеры")
    
    choice = input("\nВведите номер (1-4): ").strip()
    
    if choice == "1":
        optimize_ppo_agent()
    elif choice == "2":
        compare_agents_optimization()
    elif choice == "3":
        advanced_optimization_example()
    elif choice == "4":
        print("Запуск всех примеров оптимизации...\n")
        optimize_ppo_agent()
        print("\n" + "="*80 + "\n")
        compare_agents_optimization()
        print("\n" + "="*80 + "\n")
        advanced_optimization_example()
    else:
        print("Запуск базовой оптимизации по умолчанию...")
        optimize_ppo_agent()