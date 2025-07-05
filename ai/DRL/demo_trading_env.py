"""
Демо-скрипт для тестирования реалистичной торговой среды DRL.
Показывает все возможности конфигурации и использования среды.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.DRL.config.trading_config import (
    TradingConfig, DataManager, interactive_config_creator, 
    get_popular_configs, create_multiple_configs
)
from CryptoTrade.ai.DRL.environment.trading_env import TradingEnv
from CryptoTrade.ai.DRL.environment.reward_schemes import TradingMetrics


def demo_simple_config():
    """Демо простой конфигурации."""
    print("=== Демо простой конфигурации ===")
    
    # Создаем простую конфигурацию
    config = TradingConfig(
        exchange='binance',
        symbol='BTCUSDT',
        timeframe='1d',
        initial_balance=100.0,
        reward_scheme='default'
    )
    
    print(f"Создана конфигурация: {config.symbol} на {config.timeframe}")
    
    # Проверяем валидность
    if DataManager.validate_config(config):
        print("✅ Конфигурация валидна!")
        return config
    else:
        print("❌ Конфигурация невалидна!")
        return None


def demo_interactive_config():
    """Демо интерактивной конфигурации."""
    print("\n=== Демо интерактивной конфигурации ===")
    
    # Показать доступные данные
    available_pairs = DataManager.get_available_pairs()
    print("Доступные пары:")
    for exchange, pairs in available_pairs.items():
        print(f"{exchange}: {len(pairs)} пар")
    
    # Можно раскомментировать для интерактивного выбора
    # config = interactive_config_creator()
    # return config
    
    return None


def demo_custom_reward_config():
    """Демо кастомной схемы наград."""
    print("\n=== Демо кастомной схемы наград ===")
    
    # Создаем конфигурацию с кастомными весами наград
    custom_weights = {
        'profit': 1.2,           # Больше внимания прибыли
        'drawdown': -0.8,        # Штраф за просадку
        'sharpe': 0.4,           # Коэффициент Шарпа
        'trade_quality': 0.3,    # Качество сделок
        'volatility': -0.2,      # Штраф за волатильность
        'consistency': 0.25      # Консистентность
    }
    
    config = TradingConfig(
        exchange='binance',
        symbol='ETHUSDT',
        timeframe='4h',
        initial_balance=100.0,
        reward_scheme='custom',
        custom_reward_weights=custom_weights
    )
    
    print(f"Создана кастомная конфигурация: {config.symbol}")
    print(f"Веса наград: {custom_weights}")
    
    return config


def demo_environment_usage(config: TradingConfig):
    """Демо использования торговой среды."""
    print(f"\n=== Демо среды для {config.symbol} ===")
    
    try:
        # Создаем среду
        env = TradingEnv(config)
        print(f"Среда создана успешно!")
        print(f"Размер данных: {len(env.data)} записей")
        print(f"Пространство наблюдений: {env.observation_space.shape}")
        print(f"Пространство действий: {env.action_space.shape}")
        
        # Сброс среды
        obs = env.reset()
        print(f"Начальное наблюдение: {obs.shape}")
        
        # Тестируем несколько случайных действий
        total_reward = 0
        for step in range(10):
            # Случайное действие от -0.5 до 0.5 (покупка/продажа 50% капитала максимум)
            action = np.array([np.random.uniform(-0.5, 0.5)])
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Шаг {step+1}: действие={action[0]:.3f}, награда={reward:.4f}, "
                  f"портфель={info['portfolio_value']:.2f} USDT")
            
            if done:
                break
        
        print(f"Общая награда: {total_reward:.4f}")
        print(f"Финальная стоимость портфеля: {info['portfolio_value']:.2f} USDT")
        print(f"Общая доходность: {info['total_return']:.2%}")
        print(f"Максимальная просадка: {info['max_drawdown']:.2%}")
        print(f"Количество сделок: {info['total_trades']}")
        print(f"Доля прибыльных: {info['win_rate']:.2%}")
        
        return env, info
        
    except Exception as e:
        print(f"Ошибка при создании среды: {e}")
        return None, None


def demo_multiple_configs():
    """Демо множественных конфигураций."""
    print("\n=== Демо множественных конфигураций ===")
    
    # Популярные конфигурации
    popular_configs = get_popular_configs()
    print(f"Найдено {len(popular_configs)} популярных конфигураций")
    
    for i, config in enumerate(popular_configs[:3]):  # Показываем первые 3
        print(f"{i+1}. {config.exchange}-{config.symbol}-{config.timeframe}")
    
    # Создание множественных конфигураций
    multi_configs = create_multiple_configs(
        pairs=['BTCUSDT', 'ETHUSDT'], 
        timeframes=['1d', '4h']
    )
    print(f"Создано {len(multi_configs)} конфигураций для тестирования")
    
    return popular_configs


def demo_reward_breakdown(env: TradingEnv):
    """Демо разбивки наград по компонентам."""
    if not env or not hasattr(env, 'reward_scheme'):
        return
    
    print("\n=== Разбивка наград по компонентам ===")
    
    # Получаем разбивку наград
    if hasattr(env.reward_scheme, 'get_component_breakdown'):
        breakdown = env.reward_scheme.get_component_breakdown()
        
        print("Компоненты наград:")
        for component, stats in breakdown.items():
            print(f"{component}:")
            print(f"  Последнее значение: {stats.get('last', 0):.4f}")
            print(f"  Среднее: {stats.get('mean', 0):.4f}")
            print(f"  Общее: {stats.get('total', 0):.4f}")


def demo_trading_metrics(info: dict):
    """Демо расчета торговых метрик."""
    if not info or 'portfolio_history' not in info:
        return
    
    print("\n=== Торговые метрики ===")
    
    # Используем TradingMetrics для расчета всех метрик
    metrics = TradingMetrics.calculate_all_metrics(
        portfolio_history=info['portfolio_history'],
        trade_history=[],  # В демо нет истории сделок
        initial_balance=100.0
    )
    
    print(f"Общая доходность: {metrics.get('total_return', 0):.2%}")
    print(f"Годовая доходность: {metrics.get('annual_return', 0):.2%}")
    print(f"Коэффициент Шарпа: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Коэффициент Сортино: {metrics.get('sortino_ratio', 0):.3f}")
    print(f"Коэффициент Кальмара: {metrics.get('calmar_ratio', 0):.3f}")
    print(f"Годовая волатильность: {metrics.get('annual_volatility', 0):.2%}")


def visualize_performance(info: dict):
    """Визуализация производительности."""
    if not info or 'portfolio_history' not in info:
        return
    
    print("\n=== Визуализация производительности ===")
    
    try:
        portfolio_history = info['portfolio_history']
        
        plt.figure(figsize=(12, 6))
        
        # График стоимости портфеля
        plt.subplot(1, 2, 1)
        plt.plot(portfolio_history)
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='Начальный капитал')
        plt.title('Стоимость портфеля')
        plt.xlabel('Шаги')
        plt.ylabel('USDT')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График доходности
        plt.subplot(1, 2, 2)
        returns = [(v/100 - 1) * 100 for v in portfolio_history]
        plt.plot(returns)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.title('Доходность (%)')
        plt.xlabel('Шаги')
        plt.ylabel('Доходность %')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('portfolio_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("График сохранен как 'portfolio_performance.png'")
        
    except Exception as e:
        print(f"Ошибка при создании графика: {e}")


def main():
    """Главная функция демо."""
    print("🚀 Демо реалистичной торговой среды DRL")
    print("=" * 50)
    
    # 1. Простая конфигурация
    simple_config = demo_simple_config()
    
    # 2. Интерактивная конфигурация (закомментирована)
    # interactive_config = demo_interactive_config()
    
    # 3. Кастомная схема наград
    custom_config = demo_custom_reward_config()
    
    # 4. Множественные конфигурации
    popular_configs = demo_multiple_configs()
    
    # 5. Тестирование среды
    if simple_config:
        env, info = demo_environment_usage(simple_config)
        
        if env and info:
            # 6. Разбивка наград
            demo_reward_breakdown(env)
            
            # 7. Торговые метрики
            demo_trading_metrics(info)
            
            # 8. Визуализация
            visualize_performance(info)
    
    print("\n✅ Демо завершено!")
    print("\nДля интерактивного выбора конфигурации раскомментируйте строку:")
    print("# config = interactive_config_creator()")


if __name__ == "__main__":
    main()