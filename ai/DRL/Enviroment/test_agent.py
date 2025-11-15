"""
Скрипт для тестирования обученного DRL агента.
Загружает модель и тестирует на отложенных данных с детальной визуализацией.
"""

import os
import numpy as np
from stable_baselines3 import PPO, A2C
from datetime import datetime

from env import CryptoTradingEnv, ActionSpace, RewardType
from visualization import TradingVisualizer
from data_loader import DataLoader


def test_agent(
    model_path,
    symbol="BTCUSDT",
    timeframe="1d",
    initial_balance=10000.0,
    visualize=True,
    save_results=True
):
    """
    Протестировать обученного агента.

    Args:
        model_path: Путь к сохраненной модели
        symbol: Торговая пара
        timeframe: Таймфрейм
        initial_balance: Начальный баланс
        visualize: Создавать ли визуализацию
        save_results: Сохранять ли результаты
    """
    print("\n" + "=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ DRL АГЕНТА")
    print("=" * 70)

    # Проверяем существование модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return

    print(f"\n📦 Загрузка модели: {model_path}")

    # Загружаем модель
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
        algorithm = "PPO"
    elif 'a2c' in model_path.lower():
        model = A2C.load(model_path)
        algorithm = "A2C"
    else:
        # Пробуем PPO по умолчанию
        try:
            model = PPO.load(model_path)
            algorithm = "PPO"
        except:
            model = A2C.load(model_path)
            algorithm = "A2C"

    print(f"✅ Модель загружена: {algorithm}")

    # Загружаем данные
    print(f"\n📊 Загрузка данных: {symbol} {timeframe}")
    loader = DataLoader(symbol=symbol, timeframe=timeframe)
    loader.load()

    total_length = len(loader)
    train_size = int(total_length * 0.8)

    print(f"  Всего данных: {total_length} свечей")
    print(f"  Test period: {total_length - train_size} свечей")

    # Test окружение
    print(f"\n🧪 Создание test окружения...")
    test_env = CryptoTradingEnv(
        symbol=symbol,
        timeframe=timeframe,
        start_index=train_size,  # Используем только test данные
        initial_balance=initial_balance,
        action_type=ActionSpace.DISCRETE,
        reward_type=RewardType.RISK_ADJUSTED
    )

    print(f"  Test data: с {train_size} по {total_length} ({total_length - train_size} свечей)")

    # Тестирование
    print(f"\n🤖 Запуск тестирования...")
    print("=" * 70)

    obs, info = test_env.reset()
    total_reward = 0
    steps = 0
    actions_count = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell

    print(f"{'Step':<8} {'Action':<10} {'Price':<12} {'Balance':<12} {'Crypto':<12} {'Portfolio':<12} {'Reward':<10}")
    print("-" * 90)

    while True:
        # Предсказание агента
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        total_reward += reward
        steps += 1
        actions_count[int(action)] += 1

        # Выводим прогресс каждые 10 шагов
        if steps % 10 == 0 or terminated or truncated:
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            print(f"{steps:<8} {action_names[int(action)]:<10} "
                  f"${info['current_price']:<11.2f} "
                  f"${info['balance']:<11.2f} "
                  f"{info['crypto_held']:<11.6f} "
                  f"${info['portfolio_value']:<11.2f} "
                  f"{reward:<9.4f}")

        if terminated or truncated:
            break

    # Результаты
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)

    metrics = test_env.get_metrics()

    print(f"\n⏱️ Статистика эпизода:")
    print(f"  Общее количество шагов: {steps}")
    print(f"  Общая награда: {total_reward:.2f}")
    print(f"  Средняя награда: {total_reward/steps:.4f}")

    print(f"\n🎯 Действия агента:")
    print(f"  HOLD: {actions_count[0]} ({actions_count[0]/steps*100:.1f}%)")
    print(f"  BUY:  {actions_count[1]} ({actions_count[1]/steps*100:.1f}%)")
    print(f"  SELL: {actions_count[2]} ({actions_count[2]/steps*100:.1f}%)")

    print(f"\n💰 Финансовые результаты:")
    print(f"  Начальный баланс: ${initial_balance:.2f}")
    print(f"  Финальный портфель: ${info['portfolio_value']:.2f}")
    print(f"  Абсолютная прибыль: ${metrics.total_return:.2f}")
    print(f"  Относительная прибыль: {metrics.total_return_pct:.2f}%")

    print(f"\n📈 Метрики производительности:")
    print(f"  Annualized Return: {metrics.annualized_return:.2f}%")
    print(f"  Volatility: {metrics.annualized_volatility:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")

    print(f"\n📉 Риск метрики:")
    print(f"  Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
    print(f"  Average Drawdown: ${metrics.avg_drawdown:.2f}")

    print(f"\n🔄 Торговая активность:")
    print(f"  Всего сделок: {metrics.total_trades}")
    print(f"  Выигрышных: {metrics.winning_trades}")
    print(f"  Проигрышных: {metrics.losing_trades}")
    print(f"  Win Rate: {metrics.win_rate:.2f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Средняя сделка: ${metrics.avg_trade_return:.2f}")

    # Сравнение с Buy & Hold
    print(f"\n📊 Сравнение с Buy & Hold:")
    baseline_env = CryptoTradingEnv(
        symbol=symbol,
        timeframe=timeframe,
        start_index=train_size,
        initial_balance=initial_balance
    )

    baseline_env.reset()
    baseline_env.step(1)  # Buy
    for _ in range(steps - 1):
        baseline_env.step(0)  # Hold

    baseline_metrics = baseline_env.get_metrics()

    print(f"  Buy & Hold Return: {baseline_metrics.total_return_pct:.2f}%")
    print(f"  Agent Return: {metrics.total_return_pct:.2f}%")
    print(f"  Разница: {metrics.total_return_pct - baseline_metrics.total_return_pct:+.2f}%")

    if metrics.total_return_pct > baseline_metrics.total_return_pct:
        print(f"  ✅ Агент превзошел Buy & Hold!")
    else:
        print(f"  ⚠️ Buy & Hold показал лучший результат")

    # Визуализация
    if visualize:
        print(f"\n🎨 Создание визуализации...")

        viz = TradingVisualizer()

        # Статический график
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        static_path = f"{output_dir}/test_{algorithm}_{symbol}_{timestamp}.png"

        viz.plot_full_analysis(
            data=test_env.data_loader.raw_data,
            equity_curve=test_env.equity_curve,
            trades=test_env.trades_history,
            metrics=metrics,
            symbol=f"{symbol} - {algorithm} Test",
            save_path=static_path,
            show=False
        )

        print(f"  ✅ График сохранен: {static_path}")

        # Интерактивный график
        interactive_path = f"{output_dir}/test_{algorithm}_{symbol}_{timestamp}.html"

        viz.create_interactive_plotly(
            data=test_env.data_loader.raw_data,
            equity_curve=test_env.equity_curve,
            trades=test_env.trades_history,
            metrics=metrics,
            symbol=f"{symbol} - {algorithm} Test",
            save_path=interactive_path
        )

        print(f"  ✅ Интерактивный график: {interactive_path}")

    # Сохранение результатов
    if save_results:
        import json

        results = {
            'model_path': model_path,
            'algorithm': algorithm,
            'symbol': symbol,
            'timeframe': timeframe,
            'test_date': datetime.now().isoformat(),
            'steps': steps,
            'total_reward': float(total_reward),
            'actions': {k: int(v) for k, v in actions_count.items()},
            'metrics': {
                'total_return': float(metrics.total_return),
                'total_return_pct': float(metrics.total_return_pct),
                'sharpe_ratio': float(metrics.sharpe_ratio),
                'max_drawdown_pct': float(metrics.max_drawdown_pct),
                'total_trades': int(metrics.total_trades),
                'win_rate': float(metrics.win_rate),
                'profit_factor': float(metrics.profit_factor)
            },
            'baseline_return_pct': float(baseline_metrics.total_return_pct)
        }

        results_path = f"{output_dir}/results_{algorithm}_{symbol}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Результаты сохранены: {results_path}")

    print("\n" + "=" * 70)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70 + "\n")

    test_env.close()
    baseline_env.close()

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Тестирование DRL агента')
    parser.add_argument('--model', type=str, required=True, help='Путь к модели')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', type=str, default='1d', help='Таймфрейм')
    parser.add_argument('--balance', type=float, default=10000.0, help='Начальный баланс')
    parser.add_argument('--no-viz', action='store_true', help='Отключить визуализацию')

    args = parser.parse_args()

    test_agent(
        model_path=args.model,
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        visualize=not args.no_viz
    )
