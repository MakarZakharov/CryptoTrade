"""
Примеры использования торгового окружения.
Демонстрирует различные сценарии работы с системой.
"""

import numpy as np
from env import CryptoTradingEnv, ActionSpace, RewardType
from simulator import SlippageModel
from visualization import TradingVisualizer


def example_1_basic_usage():
    """
    Пример 1: Базовое использование окружения.
    Создание, reset, несколько шагов.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    # Создаем окружение
    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        action_type=ActionSpace.DISCRETE
    )

    print(f"Environment created!")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Data loaded: {len(env.data_loader)} candles")

    # Reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:.2f}")

    # Несколько случайных шагов
    print("\nExecuting 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        print(f"Step {i+1}: Action={action_names[action]}, "
              f"Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")

        if terminated or truncated:
            break

    # Получаем метрики
    metrics = env.get_metrics()
    print(f"\nFinal Metrics:")
    print(f"  Total Return: {metrics.total_return_pct:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    env.close()


def example_2_manual_trading():
    """
    Пример 2: Ручная торговая стратегия.
    Реализуем простую стратегию buy & hold.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Manual Trading Strategy (Buy & Hold)")
    print("=" * 70)

    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0
    )

    obs, info = env.reset()
    print(f"Starting balance: ${info['balance']:.2f}")

    # Стратегия: купить в начале и держать
    bought = False

    for step in range(100):
        if not bought:
            action = 1  # BUY
            bought = True
        else:
            action = 0  # HOLD

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            print(f"Step {step}: Portfolio=${info['portfolio_value']:.2f}, "
                  f"Crypto={info['crypto_held']:.6f}")

        if terminated or truncated:
            break

    # Финальная метрика
    metrics = env.get_metrics()
    print(f"\nBuy & Hold Results:")
    print(f"  Final Portfolio: ${info['portfolio_value']:.2f}")
    print(f"  Total Return: {metrics.total_return_pct:.2f}%")
    print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")

    env.close()


def example_3_momentum_strategy():
    """
    Пример 3: Простая momentum стратегия.
    Покупаем при росте, продаем при падении.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Momentum Strategy")
    print("=" * 70)

    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0
    )

    obs, info = env.reset()

    # История цен для расчета momentum
    price_history = []
    momentum_period = 5

    for step in range(150):
        current_price = info['current_price']
        price_history.append(current_price)

        # Рассчитываем momentum
        if len(price_history) >= momentum_period:
            momentum = (price_history[-1] - price_history[-momentum_period]) / price_history[-momentum_period]

            # Торговые сигналы
            if momentum > 0.02 and env.crypto_held == 0:
                # Сильный рост - покупаем
                action = 1  # BUY
            elif momentum < -0.02 and env.crypto_held > 0:
                # Сильное падение - продаем
                action = 2  # SELL
            else:
                # Держим
                action = 0  # HOLD
        else:
            action = 0  # HOLD

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Результаты
    metrics = env.get_metrics()
    print(f"\nMomentum Strategy Results:")
    print(f"  Total Return: {metrics.total_return_pct:.2f}%")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    env.close()


def example_4_different_configurations():
    """
    Пример 4: Разные конфигурации окружения.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Different Environment Configurations")
    print("=" * 70)

    # Конфигурация 1: Continuous action space
    print("\n1. Continuous Action Space:")
    env1 = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        action_type=ActionSpace.CONTINUOUS,
        initial_balance=10000.0
    )
    print(f"   Action space: {env1.action_space}")
    env1.close()

    # Конфигурация 2: Different reward type
    print("\n2. Different Reward Types:")
    for reward_type in [RewardType.PNL, RewardType.SHARPE, RewardType.RISK_ADJUSTED]:
        env = CryptoTradingEnv(
            symbol="BTCUSDT",
            timeframe="1d",
            reward_type=reward_type,
            initial_balance=10000.0
        )
        print(f"   {reward_type.value}: Created successfully")
        env.close()

    # Конфигурация 3: Different slippage models
    print("\n3. Different Slippage Models:")
    for slippage_model in [SlippageModel.FIXED, SlippageModel.PERCENTAGE, SlippageModel.VOLUME_BASED]:
        env = CryptoTradingEnv(
            symbol="BTCUSDT",
            timeframe="1d",
            slippage_model=slippage_model,
            initial_balance=10000.0
        )
        print(f"   {slippage_model.value}: Created successfully")
        env.close()

    print("\nAll configurations work!")


def example_5_visualization():
    """
    Пример 5: Визуализация результатов торговли.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Trading Visualization")
    print("=" * 70)

    # Создаем окружение и торгуем
    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0
    )

    obs, info = env.reset()

    # Простая стратегия
    for step in range(100):
        if step % 20 == 0:
            action = 1  # BUY
        elif step % 20 == 10:
            action = 2  # SELL
        else:
            action = 0  # HOLD

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Получаем данные для визуализации
    equity_curve = env.equity_curve
    trades = env.trades_history
    metrics = env.get_metrics()
    data = env.data_loader.raw_data

    print(f"Creating visualizations...")
    print(f"  Equity curve points: {len(equity_curve)}")
    print(f"  Trades: {len(trades)}")

    # Создаем визуализатор
    viz = TradingVisualizer()

    # Статический график
    viz.plot_full_analysis(
        data=data.iloc[:100],
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        symbol="BTCUSDT",
        save_path="trading_analysis.png",
        show=False
    )

    print(f"  ✓ Static plot saved to 'trading_analysis.png'")

    # Интерактивный график (Plotly)
    viz.create_interactive_plotly(
        data=data.iloc[:100],
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        symbol="BTCUSDT",
        save_path="trading_analysis.html"
    )

    print(f"  ✓ Interactive plot saved to 'trading_analysis.html'")

    env.close()


def example_6_train_test_split():
    """
    Пример 6: Разделение данных на train/test.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Train/Test Split")
    print("=" * 70)

    # Загружаем данные
    from data_loader import DataLoader

    loader = DataLoader(symbol="BTCUSDT", timeframe="1d")
    loader.load()

    total_length = len(loader)
    print(f"Total data points: {total_length}")

    # Разделяем на train/test
    train_loader, test_loader = loader.split_train_test(train_ratio=0.8)

    print(f"Train data points: {len(train_loader)}")
    print(f"Test data points: {len(test_loader)}")

    # Создаем train окружение
    print("\nTraining environment:")
    train_env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        end_index=int(total_length * 0.8),
        initial_balance=10000.0
    )
    print(f"  Data points: {len(train_env.data_loader)}")

    # Создаем test окружение
    print("\nTest environment:")
    test_env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        start_index=int(total_length * 0.8),
        initial_balance=10000.0
    )
    print(f"  Data points: {len(test_env.data_loader)}")

    train_env.close()
    test_env.close()


def example_7_sb3_compatibility():
    """
    Пример 7: Совместимость с Stable-Baselines3.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Stable-Baselines3 Compatibility")
    print("=" * 70)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        # Создаем окружение
        env = CryptoTradingEnv(
            symbol="BTCUSDT",
            timeframe="1d",
            initial_balance=10000.0
        )

        # Проверяем совместимость
        print("Checking environment compatibility with SB3...")
        check_env(env)
        print("✓ Environment is compatible with Stable-Baselines3!")

        # Создаем модель (не обучаем, просто демонстрация)
        print("\nCreating PPO model...")
        model = PPO("MlpPolicy", env, verbose=0)
        print("✓ PPO model created successfully!")

        # Тестируем предсказание
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"✓ Model prediction works! Action: {action}")

        env.close()

    except ImportError:
        print("⚠ Stable-Baselines3 not installed.")
        print("Install with: pip install stable-baselines3")


def main():
    """Запустить все примеры."""
    print("\n" + "=" * 70)
    print("CRYPTO TRADING DRL ENVIRONMENT - USAGE EXAMPLES")
    print("=" * 70)

    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Manual Trading (Buy & Hold)", example_2_manual_trading),
        ("Momentum Strategy", example_3_momentum_strategy),
        ("Different Configurations", example_4_different_configurations),
        ("Visualization", example_5_visualization),
        ("Train/Test Split", example_6_train_test_split),
        ("Stable-Baselines3 Compatibility", example_7_sb3_compatibility)
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in '{name}': {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
