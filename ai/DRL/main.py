#!/usr/bin/env python3
"""
Главный файл для демонстрации DRL системы торговли криптовалютой.
Показывает полный цикл работы от сбора данных до оценки производительности.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Настройка путей
sys.path.append(os.path.dirname(__file__))

# Импорты модулей системы
from data_processing.data_collector import CryptoDataCollector, DataConfig
from data_processing.feature_engineering import FeatureEngineer
from environment.trading_env import create_trading_environment, TradingConfig
from agents.ppo_agent import create_ppo_agent
from evaluation.backtester import Backtester, BacktestConfig


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'drl_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger('DRL_Demo')


def create_sample_data() -> pd.DataFrame:
    """Создание демонстрационных данных для тестирования."""
    logger = logging.getLogger('DRL_Demo')
    logger.info("Создание демонстрационных данных...")
    
    # Параметры для реалистичных данных
    np.random.seed(42)
    n_points = 2000  # 2000 часовых свечей (~83 дня)
    
    # Базовая цена и волатильность
    base_price = 45000  # Примерная цена BTC
    volatility = 0.02   # 2% часовая волатильность
    
    # Генерация цен с трендом и случайной компонентой
    price_changes = np.random.normal(0.0001, volatility, n_points)  # Небольшой восходящий тренд
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Создание OHLC данных
    data = []
    for i in range(n_points):
        # Open цена
        if i == 0:
            open_price = base_price
        else:
            open_price = data[i-1]['close']
        
        # Close цена
        close_price = prices[i]
        
        # High и Low с учетом волатильности
        intraday_range = abs(close_price - open_price) + np.random.exponential(close_price * 0.005)
        high_price = max(open_price, close_price) + np.random.uniform(0, intraday_range * 0.3)
        low_price = min(open_price, close_price) - np.random.uniform(0, intraday_range * 0.3)
        
        # Volume с некоторой корреляцией с волатильностью
        volume_base = 1000000  # Базовый объем
        volume_multiplier = 1 + abs(price_changes[i]) * 50  # Больше объема при больших движениях
        volume = int(volume_base * volume_multiplier * np.random.uniform(0.5, 2.0))
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Создание DataFrame с временными индексами
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')
    df = pd.DataFrame(data, index=dates)
    
    logger.info(f"Создано {len(df)} записей демо-данных")
    logger.info(f"Период: {df.index[0]} - {df.index[-1]}")
    logger.info(f"Цена: от ${df['close'].min():.2f} до ${df['close'].max():.2f}")
    
    return df


def demonstrate_data_processing():
    """Демонстрация сбора и обработки данных."""
    logger = logging.getLogger('DRL_Demo')
    logger.info("=" * 60)
    logger.info("ДЕМОНСТРАЦИЯ ОБРАБОТКИ ДАННЫХ")
    logger.info("=" * 60)
    
    # Создание демо-данных (в реальности здесь был бы сбор с API)
    raw_data = create_sample_data()
    
    # Генерация технических индикаторов
    logger.info("Добавление технических индикаторов...")
    feature_engineer = FeatureEngineer()
    enhanced_data = feature_engineer.add_all_features(raw_data)
    
    logger.info(f"Исходных колонок: {len(raw_data.columns)}")
    logger.info(f"После добавления признаков: {len(enhanced_data.columns)}")
    logger.info(f"Добавлено признаков: {len(enhanced_data.columns) - len(raw_data.columns)}")
    
    return enhanced_data


def demonstrate_environment():
    """Демонстрация торговой среды."""
    logger = logging.getLogger('DRL_Demo')
    logger.info("=" * 60)
    logger.info("ДЕМОНСТРАЦИЯ ТОРГОВОЙ СРЕДЫ")
    logger.info("=" * 60)
    
    # Получение данных
    data = create_sample_data()
    
    # Создание торговой среды
    config = TradingConfig(
        initial_balance=10000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        lookback_window=30
    )
    
    env = create_trading_environment(data, config, 'profit_based')
    
    logger.info(f"Пространство наблюдений: {env.observation_space.shape}")
    logger.info(f"Пространство действий: {env.action_space}")
    
    # Тестирование среды
    logger.info("Тестирование среды с случайными действиями...")
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    for i in range(100):  # 100 случайных действий
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    logger.info(f"Выполнено шагов: {steps}")
    logger.info(f"Общая награда: {total_reward:.4f}")
    logger.info(f"Финальная стоимость портфеля: ${info['portfolio_value']:.2f}")
    logger.info(f"Доходность: {info['total_return']:.2%}")
    
    return env


def demonstrate_agent_training():
    """Демонстрация обучения агента."""
    logger = logging.getLogger('DRL_Demo')
    logger.info("=" * 60)
    logger.info("ДЕМОНСТРАЦИЯ ОБУЧЕНИЯ АГЕНТА")
    logger.info("=" * 60)
    
    # Подготовка данных для обучения
    data = create_sample_data()
    train_data = data[:1600]  # 80% для обучения
    
    # Создание среды обучения
    config = TradingConfig(
        initial_balance=10000.0,
        lookback_window=30
    )
    train_env = create_trading_environment(train_data, config)
    
    # Создание PPO агента
    logger.info("Создание PPO агента...")
    agent = create_ppo_agent(train_env)
    agent.create_model()
    
    # Обучение (короткое для демонстрации)
    logger.info("Начало обучения (демо-режим с 20,000 шагов)...")
    agent.train(total_timesteps=20000)
    
    # Оценка обучения
    logger.info("Оценка производительности после обучения...")
    eval_results = agent.evaluate(n_episodes=5)
    
    logger.info(f"Средняя награда: {eval_results['mean_reward']:.4f}")
    logger.info(f"Стандартное отклонение: {eval_results['std_reward']:.4f}")
    logger.info(f"Лучший результат: {eval_results['max_reward']:.4f}")
    
    return agent


def demonstrate_backtesting():
    """Демонстрация бэктестинга."""
    logger = logging.getLogger('DRL_Demo')
    logger.info("=" * 60)
    logger.info("ДЕМОНСТРАЦИЯ БЭКТЕСТИНГА")
    logger.info("=" * 60)
    
    # Подготовка данных
    data = create_sample_data()
    train_data = data[:1600]
    test_data = data[1600:]  # 20% для тестирования
    
    # Обучение агента
    logger.info("Быстрое обучение агента для бэктестинга...")
    config = TradingConfig(initial_balance=10000.0, lookback_window=30)
    train_env = create_trading_environment(train_data, config)
    
    agent = create_ppo_agent(train_env)
    agent.create_model()
    agent.train(total_timesteps=15000)  # Быстрое обучение
    
    # Бэктестинг
    logger.info("Запуск бэктестинга...")
    backtest_config = BacktestConfig(
        initial_capital=10000.0,
        commission=0.001,
        benchmark='buy_and_hold'
    )
    
    backtester = Backtester(backtest_config)
    results = backtester.run_backtest(agent, test_data)
    
    # Вывод результатов
    backtester.print_results()
    
    return results


def main():
    """Главная демонстрационная функция."""
    # Настройка логирования
    logger = setup_logging()
    
    logger.info("🚀 ЗАПУСК ДЕМОНСТРАЦИИ DRL СИСТЕМЫ ТОРГОВЛИ КРИПТОВАЛЮТОЙ")
    logger.info("=" * 80)
    
    try:
        # 1. Демонстрация обработки данных
        processed_data = demonstrate_data_processing()
        
        # 2. Демонстрация торговой среды
        env = demonstrate_environment()
        
        # 3. Демонстрация обучения агента
        agent = demonstrate_agent_training()
        
        # 4. Демонстрация бэктестинга
        backtest_results = demonstrate_backtesting()
        
        # Финальные результаты
        logger.info("=" * 80)
        logger.info("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        logger.info("=" * 80)
        
        logger.info("Основные результаты:")
        logger.info(f"✅ Обработано данных: {len(processed_data)} записей")
        logger.info(f"✅ Создана торговая среда с {env.observation_space.shape} наблюдениями")
        logger.info(f"✅ Обучен PPO агент")
        logger.info(f"✅ Выполнен бэктестинг")
        
        if backtest_results:
            logger.info(f"📊 Итоговая доходность: {backtest_results['total_return']:.2%}")
            logger.info(f"📊 Коэффициент Шарпа: {backtest_results['sharpe_ratio']:.2f}")
            logger.info(f"📊 Максимальная просадка: {backtest_results['max_drawdown']:.2%}")
        
        logger.info("\nДля продакшн использования:")
        logger.info("1. Подключите реальные API для сбора данных")
        logger.info("2. Увеличьте количество шагов обучения (500k-1M)")
        logger.info("3. Настройте гиперпараметры")
        logger.info("4. Добавьте более сложные функции наград")
        logger.info("5. Реализуйте систему мониторинга и развертывания")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в демонстрации: {e}")
        raise


if __name__ == "__main__":
    main()