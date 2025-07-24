"""
DRL система для торговли криптовалютами - Enterprise Edition.

Полностью реорганизованная архитектура с описательными именами и логической структурой:

deep_learning_trading_agents - DRL агенты для торговли
cryptocurrency_trading_environments - Торговые среды и симуляторы  
market_data_processing_pipeline - Обработка рыночных данных
configuration_and_parameter_management - Управление конфигурациями
training_and_optimization_system - Система обучения и оптимизации
backtesting_and_performance_evaluation - Бэктестинг и оценка производительности
utilities_and_shared_components - Общие утилиты и компоненты
production_deployment_tools - Инструменты для продакшена
example_implementations_and_tutorials - Примеры и обучающие материалы
research_and_experimental_features - Исследовательские функции
saved_models_and_artifacts - Сохраненные модели и артефакты
experimental_results_and_logs - Результаты экспериментов и логи
"""

# Основные импорты для удобства
try:
    from .deep_learning_trading_agents.reinforcement_learning_algorithms import (
        BaseReinforcementLearningAgent,
        ProximalPolicyOptimizationAgent
    )
    from .cryptocurrency_trading_environments.market_simulation_and_modeling import (
        RealisticMarketSimulator
    )
    from .configuration_and_parameter_management import DRLConfig, TradingConfig
    from .utilities_and_shared_components import DRLLogger
    
    # Backward compatibility aliases
    BaseAgent = BaseReinforcementLearningAgent
    PPOAgent = ProximalPolicyOptimizationAgent
    MarketSimulator = RealisticMarketSimulator
    
    __all__ = [
        # Новые названия
        'BaseReinforcementLearningAgent',
        'ProximalPolicyOptimizationAgent', 
        'RealisticMarketSimulator',
        'DRLConfig',
        'TradingConfig',
        'DRLLogger',
        
        # Backward compatibility
        'BaseAgent',
        'PPOAgent',
        'MarketSimulator'
    ]
    
except ImportError as e:
    print(f"Предупреждение: Не удалось импортировать некоторые компоненты: {e}")
    __all__ = []

# Версия системы
__version__ = "2.0.0-enterprise"