"""
Ручная торговля в окружении.
Позволяет вам самостоятельно торговать и сравнивать результаты с агентом.
"""

import os
import numpy as np
from datetime import datetime

from env import CryptoTradingEnv, ActionSpace, RewardType
from visualization import TradingVisualizer


def manual_trading(
    symbol="BTCUSDT",
    timeframe="1d",
    initial_balance=10000.0,
    start_from_train_end=True
):
    """
    Ручная торговля в окружении.

    Args:
        symbol: Торговая пара
        timeframe: Таймфрейм
        initial_balance: Начальный баланс
        start_from_train_end: Начать с конца train данных (test период)
    """
    print("\n" + "=" * 70)
    print("🎮 РУЧНАЯ ТОРГОВЛЯ")
    print("=" * 70)

    # Загружаем данные
    from data_loader import DataLoader
    loader = DataLoader(symbol=symbol, timeframe=timeframe)
    loader.load()

    total_length = len(loader)

    if start_from_train_end:
        start_index = int(total_length * 0.8)
        print(f"\n📊 Используем test период (последние 20% данных)")
    else:
        start_index = 0
        print(f"\n📊 Используем все данные")

    print(f"  Доступно свечей: {total_length - start_index}")

    # Создаем окружение
    env = CryptoTradingEnv(
        symbol=symbol,
        timeframe=timeframe,
        start_index=start_index,
        initial_balance=initial_balance,
        action_type=ActionSpace.DISCRETE,
        reward_type=RewardType.RISK_ADJUSTED
    )

    print(f"\n💰 Начальный баланс: ${initial_balance:.2f}")
    print(f"  Торговая пара: {symbol}")
    print(f"  Таймфрейм: {timeframe}")

    # Инструкции
    print("\n" + "=" * 70)
    print("📖 ИНСТРУКЦИИ")
    print("=" * 70)
    print("  0 или H - HOLD (держать)")
    print("  1 или B - BUY (купить)")
    print("  2 или S - SELL (продать)")
    print("  Q - выход и сохранение результатов")
    print("  I - показать текущую информацию")
    print("=" * 70)

    # Начинаем эпизод
    obs, info = env.reset()

    print(f"\n🏁 НАЧАЛО ТОРГОВЛИ")
    print(f"  Текущая цена: ${info['current_price']:.2f}")
    print(f"  Баланс: ${info['balance']:.2f}")

    step = 0
    running = True

    while running:
        step += 1

        # Показываем текущее состояние
        print("\n" + "-" * 70)
        print(f"📍 Шаг {step}")
        print(f"  Цена: ${info['current_price']:.2f}")
        print(f"  Баланс: ${info['balance']:.2f}")
        print(f"  Крипта: {info['crypto_held']:.6f} ({info['crypto_held'] * info['current_price']:.2f} USD)")
        print(f"  Портфель: ${info['portfolio_value']:.2f}")
        print(f"  P&L: ${info['portfolio_value'] - initial_balance:.2f} "
              f"({(info['portfolio_value'] / initial_balance - 1) * 100:+.2f}%)")

        # Получаем действие от пользователя
        action_input = input("\n👉 Ваше действие (0=Hold, 1=Buy, 2=Sell, Q=Quit, I=Info): ").strip().upper()

        # Обработка команд
        if action_input == 'Q':
            print("\n🛑 Выход из торговли...")
            break

        elif action_input == 'I':
            # Детальная информация
            print("\n" + "=" * 70)
            print("📊 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ")
            print("=" * 70)
            print(f"  Шаг: {step}")
            print(f"  Цена: ${info['current_price']:.2f}")
            print(f"  Баланс (USD): ${info['balance']:.2f}")
            print(f"  Крипта: {info['crypto_held']:.6f}")
            print(f"  Стоимость крипты: ${info['crypto_held'] * info['current_price']:.2f}")
            print(f"  Портфель: ${info['portfolio_value']:.2f}")
            print(f"  P&L: ${info['portfolio_value'] - initial_balance:.2f}")
            print(f"  P&L%: {(info['portfolio_value'] / initial_balance - 1) * 100:+.2f}%")
            print(f"  Всего сделок: {env.total_trades}")

            # Показываем последние сделки
            if env.trades_history:
                print(f"\n  Последние сделки:")
                for i, trade in enumerate(env.trades_history[-5:], 1):
                    print(f"    {i}. {trade.side}: Entry=${trade.entry_price:.2f}, "
                          f"Exit=${trade.exit_price if trade.exit_price else 'N/A'}, "
                          f"PnL=${trade.pnl:.2f}")

            continue

        # Преобразуем ввод в действие
        if action_input in ['0', 'H', 'HOLD']:
            action = 0
            action_name = "HOLD"
        elif action_input in ['1', 'B', 'BUY']:
            action = 1
            action_name = "BUY"
        elif action_input in ['2', 'S', 'SELL']:
            action = 2
            action_name = "SELL"
        else:
            print("❌ Неверный ввод! Попробуйте снова.")
            continue

        # Выполняем действие
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n✅ Действие: {action_name}")
        print(f"   Награда: {reward:.4f}")

        if info.get('trade_executed'):
            print(f"   🔔 Сделка исполнена!")

        # Проверяем завершение
        if terminated or truncated:
            print("\n🏁 Эпизод завершен!")
            if terminated:
                print("   Причина: достигнуто условие завершения")
            break

    # Финальные результаты
    print("\n" + "=" * 70)
    print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    metrics = env.get_metrics()

    print(f"\n⏱️ Статистика:")
    print(f"  Всего шагов: {step}")

    print(f"\n💰 Финансовые результаты:")
    print(f"  Начальный баланс: ${initial_balance:.2f}")
    print(f"  Финальный портфель: ${info['portfolio_value']:.2f}")
    print(f"  Абсолютная прибыль: ${metrics.total_return:.2f}")
    print(f"  Относительная прибыль: {metrics.total_return_pct:.2f}%")

    print(f"\n📈 Метрики производительности:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")

    print(f"\n🔄 Торговая активность:")
    print(f"  Всего сделок: {metrics.total_trades}")
    print(f"  Выигрышных: {metrics.winning_trades}")
    print(f"  Проигрышных: {metrics.losing_trades}")
    print(f"  Win Rate: {metrics.win_rate:.2f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")

    # Сохранение результатов
    save = input("\n💾 Сохранить результаты? (Y/n): ").strip().upper()

    if save != 'N':
        print("\n📁 Сохранение результатов...")

        # Создаем директорию
        output_dir = "manual_trading_results"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Визуализация
        viz = TradingVisualizer()

        # Статический график
        static_path = f"{output_dir}/manual_{symbol}_{timestamp}.png"
        viz.plot_full_analysis(
            data=env.data_loader.raw_data,
            equity_curve=env.equity_curve,
            trades=env.trades_history,
            metrics=metrics,
            symbol=f"{symbol} - Manual Trading",
            save_path=static_path,
            show=False
        )
        print(f"  ✅ График: {static_path}")

        # Интерактивный график
        interactive_path = f"{output_dir}/manual_{symbol}_{timestamp}.html"
        viz.create_interactive_plotly(
            data=env.data_loader.raw_data,
            equity_curve=env.equity_curve,
            trades=env.trades_history,
            metrics=metrics,
            symbol=f"{symbol} - Manual Trading",
            save_path=interactive_path
        )
        print(f"  ✅ Интерактивный: {interactive_path}")

        # Метрики в JSON
        import json
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'date': datetime.now().isoformat(),
            'steps': step,
            'metrics': {
                'total_return': float(metrics.total_return),
                'total_return_pct': float(metrics.total_return_pct),
                'sharpe_ratio': float(metrics.sharpe_ratio),
                'max_drawdown_pct': float(metrics.max_drawdown_pct),
                'total_trades': int(metrics.total_trades),
                'win_rate': float(metrics.win_rate)
            }
        }

        json_path = f"{output_dir}/manual_{symbol}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ✅ Метрики: {json_path}")

        print("\n✅ Результаты сохранены!")

    print("\n" + "=" * 70)
    print("🎉 СПАСИБО ЗА ИГРУ!")
    print("=" * 70 + "\n")

    env.close()

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ручная торговля')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', type=str, default='1d', help='Таймфрейм')
    parser.add_argument('--balance', type=float, default=10000.0, help='Начальный баланс')
    parser.add_argument('--full', action='store_true', help='Использовать все данные (не только test)')

    args = parser.parse_args()

    manual_trading(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        start_from_train_end=not args.full
    )
