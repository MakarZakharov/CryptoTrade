#!/usr/bin/env python3
"""
MVP скрипт для швидкого запуску навчання STAS_ML агента.
Простий інтерфейс для початку навчання з мінімальними налаштуваннями.
"""

import os
import sys
import argparse
from datetime import datetime

# Додаємо шлях до модулів проекту
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.STAS_ML.config.trading_config import (
    TradingConfig, DataManager, interactive_config_creator
)
from CryptoTrade.ai.STAS_ML.training.train import DRLTrainer, quick_train
from CryptoTrade.ai.STAS_ML.evaluation.evaluate import quick_evaluate


def print_banner():
    """Вивести банер програми."""
    print("🚀" + "="*60 + "🚀")
    print("   MVP НАВЧАННЯ STAS_ML АГЕНТА ДЛЯ ТОРГІВЛІ КРИПТОВАЛЮТАМИ")
    print("🚀" + "="*60 + "🚀")
    print()


def check_dependencies():
    """Перевірити наявність необхідних залежностей."""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import stable_baselines3
    except ImportError:
        missing_deps.append("stable-baselines3")
    
    try:
        import tensorboard
    except ImportError:
        missing_deps.append("tensorboard")
    
    if missing_deps:
        print("❌ Відсутні залежності:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Встановіть залежності:")
        print("   pip install -r CryptoTrade/requirements.txt")
        return False
    
    print("✅ Всі залежності встановлені")
    return True


def show_available_data():
    """Показати доступні дані."""
    print("📊 Доступні дані:")
    available_pairs = DataManager.get_available_pairs()
    
    total_pairs = 0
    for exchange, pairs in available_pairs.items():
        print(f"   {exchange}: {len(pairs)} пар")
        total_pairs += len(pairs)
    
    print(f"   Всього: {total_pairs} торгових пар")
    print()


def create_quick_config():
    """Створити швидку конфігурацію."""
    print("⚡ Швидке налаштування (рекомендується для початківців):")
    print("   1. BTCUSDT на денному таймфреймі")
    print("   2. PPO агент")
    print("   3. Оптимізована схема винагород")
    print("   4. 100,000 кроків навчання")
    
    choice = input("\nВикористовувати швидке налаштування? (y/n): ").lower()
    
    if choice in ['y', 'yes', 'так', '']:
        return TradingConfig(
            symbol='BTCUSDT',
            timeframe='1d',
            reward_scheme='optimized',
            initial_balance=10000.0
        ), "PPO", 100000
    
    return None, None, None


def custom_config_menu():
    """Меню кастомної конфігурації."""
    print("\n🛠️ Кастомне налаштування:")
    
    # Вибір пари
    available_pairs = DataManager.get_available_pairs()
    print("\nДоступні біржі:")
    exchanges = list(available_pairs.keys())
    for i, exchange in enumerate(exchanges, 1):
        print(f"   {i}. {exchange}")
    
    while True:
        try:
            choice = int(input(f"Оберіть біржу (1-{len(exchanges)}): ")) - 1
            if 0 <= choice < len(exchanges):
                selected_exchange = exchanges[choice]
                break
        except ValueError:
            pass
        print("❌ Невірний вибір!")
    
    # Выбор пары
    pairs = available_pairs[selected_exchange]
    print(f"\nДоступные пары на {selected_exchange}:")
    for i, pair in enumerate(pairs[:10], 1):  # Показываем первые 10
        print(f"   {i}. {pair}")
    if len(pairs) > 10:
        print(f"   ... и еще {len(pairs) - 10} пар")
    
    symbol = input("Введите символ пары (например, BTCUSDT): ").upper()
    if symbol not in pairs:
        print(f"⚠️ Пара {symbol} не найдена, используем BTCUSDT")
        symbol = "BTCUSDT"
    
    # Выбор таймфрейма
    timeframes = DataManager.get_available_timeframes(selected_exchange, symbol)
    print(f"\nДоступные таймфреймы для {symbol}:")
    for i, tf in enumerate(timeframes, 1):
        print(f"   {i}. {tf}")
    
    while True:
        try:
            choice = int(input(f"Выберите таймфрейм (1-{len(timeframes)}): ")) - 1
            if 0 <= choice < len(timeframes):
                selected_timeframe = timeframes[choice]
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Выбор агента
    print("\nТип агента:")
    print("   1. PPO (рекомендуется)")
    print("   2. DQN")
    
    while True:
        try:
            choice = int(input("Выберите агента (1-2): "))
            if choice == 1:
                agent_type = "PPO"
                break
            elif choice == 2:
                agent_type = "DQN"
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Количество шагов
    print("\nКоличество шагов обучения:")
    print("   1. Быстро (100,000 шагов, ~15 минут)")
    print("   2. Средне (500,000 шагов, ~1 час)")
    print("   3. Долго (1,000,000 шагов, ~2 часа)")
    print("   4. Пользовательское")
    
    while True:
        try:
            choice = int(input("Выберите (1-4): "))
            if choice == 1:
                timesteps = 100000
                break
            elif choice == 2:
                timesteps = 500000
                break
            elif choice == 3:
                timesteps = 1000000
                break
            elif choice == 4:
                timesteps = int(input("Введите количество шагов: "))
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    config = TradingConfig(
        exchange=selected_exchange,
        symbol=symbol,
        timeframe=selected_timeframe,
        reward_scheme='optimized',
        initial_balance=10000.0
    )
    
    return config, agent_type, timesteps


def get_model_storage_path(config: TradingConfig, model_type: str = "final") -> str:
    """
    Визначити шлях збереження моделі на основі конфігурації.
    
    Args:
        config: Конфігурація торгового середовища
        model_type: Тип моделі ("final", "best", "checkpoint")
        
    Returns:
        Повний шлях до файлу моделі
    """
    experiment_name = f"{config.symbol}_{config.timeframe}_{config.reward_scheme}"
    model_dir = os.path.join("models", experiment_name)
    
    if model_type == "final":
        return os.path.join(model_dir, "final_model.zip")
    elif model_type == "best":
        return os.path.join(model_dir, "best_model.zip")
    elif model_type == "checkpoint":
        return os.path.join(model_dir, "checkpoints")
    else:
        return os.path.join(model_dir, f"{model_type}.zip")


def show_model_locations(config: TradingConfig):
    """Показати інформацію про розташування моделей."""
    experiment_name = f"{config.symbol}_{config.timeframe}_{config.reward_scheme}"
    model_dir = os.path.join("models", experiment_name)
    
    print(f"\n📁 Розташування моделей для експерименту: {experiment_name}")
    print(f"   Базова директорія: {os.path.abspath(model_dir)}")
    
    # Перевіряємо які моделі існують
    model_files = {
        "Фінальна модель": os.path.join(model_dir, "final_model.zip"),
        "Найкраща модель": os.path.join(model_dir, "best_model.zip"),
        "Директорія чекпоінтів": os.path.join(model_dir, "checkpoints")
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                # Підрахунок файлів у директорії чекпоінтів
                try:
                    checkpoint_files = [f for f in os.listdir(model_path) if f.endswith('.zip')]
                    print(f"   ✅ {model_name}: {model_path} ({len(checkpoint_files)} файлів)")
                except:
                    print(f"   ✅ {model_name}: {model_path}")
            else:
                # Розмір файлу моделі
                try:
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"   ✅ {model_name}: {model_path} ({size_mb:.1f} MB)")
                except:
                    print(f"   ✅ {model_name}: {model_path}")
        else:
            print(f"   ❌ {model_name}: {model_path} (не існує)")
    
    # Логи
    log_dir = os.path.join("logs", experiment_name)
    if os.path.exists(log_dir):
        print(f"   📊 Логи: {os.path.abspath(log_dir)}")
    else:
        print(f"   📊 Логи: {os.path.abspath(log_dir)} (не існує)")


def main():
    """Головна функція MVP - автоматичний запуск."""
    print_banner()
    
    # Перевіряємо залежності
    if not check_dependencies():
        return
    
    # Автоматична конфігурація (BTCUSDT, 1d, PPO, optimized)
    config = TradingConfig(
        symbol='BTCUSDT',
        timeframe='1d',
        reward_scheme='optimized',
        initial_balance=10000.0
    )
    agent_type = "PPO"
    timesteps = 1000000  # Збільшено до 1 мільйона кроків (~3-4 години навчання)
    
    # Показуємо налаштування
    print(f"🚀 Автоматичний запуск навчання:")
    print(f"   Пара: {config.symbol}")
    print(f"   Таймфрейм: {config.timeframe}")
    print(f"   Агент: {agent_type}")
    print(f"   Кроків: {timesteps:,}")
    print(f"   Схема винагород: {config.reward_scheme}")
    
    # Показуємо інформацію про розташування моделей
    show_model_locations(config)
    
    print(f"💡 Моніторинг: tensorboard --logdir logs")
    print(f"💡 Для зупинки: Ctrl+C")
    print("-" * 60)
    
    try:
        trainer = DRLTrainer(config, resume_training=True)
        agent = trainer.train(
            total_timesteps=timesteps,
            agent_type=agent_type
        )
        
        print(f"\n✅ Навчання завершено успішно!")
        print(f"📁 Модель збережена в: models/{trainer.experiment_name}")
        print(f"📊 Логи в: logs/{trainer.experiment_name}")
        
        # Показуємо детальну інформацію про збережені моделі
        show_model_locations(config)
        
        # Автоматична оцінка
        print("\n🔍 Запуск автоматичної оцінки...")
        model_path = f"models/{trainer.experiment_name}/final_model"
        try:
            evaluator, results, report = quick_evaluate(
                model_path=model_path,
                symbol=config.symbol,
                timeframe=config.timeframe,
                agent_type=agent_type,
                episodes=5
            )
            print(f"✅ Оцінка завершена!")
        except Exception as e:
            print(f"❌ Помилка при оцінці: {e}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Навчання зупинено користувачем")
    except Exception as e:
        print(f"\n❌ Помилка під час навчання: {e}")
        print(f"💡 Перевірте логи в: logs/")


if __name__ == "__main__":
    # Підтримка аргументів командного рядка для досвідчених користувачів
    parser = argparse.ArgumentParser(description='MVP навчання STAS_ML агента', add_help=False)
    parser.add_argument('--interactive', action='store_true', 
                       help='Інтерактивний режим налаштування')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торгова пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--agent', default='PPO', choices=['PPO', 'DQN'], help='Тип агента')
    parser.add_argument('--timesteps', type=int, default=200000, help='Кількість кроків')
    parser.add_argument('--help', '-h', action='store_true', help='Показати довідку')
    
    args = parser.parse_args()
    
    if args.help:
        print("🚀 MVP Навчання STAS_ML Агента")
        print("\nВикористання:")
        print("  python mvp_train.py                    # Швидкий запуск (за замовчуванням)")
        print("  python mvp_train.py --interactive      # Інтерактивний режим")
        print("  python mvp_train.py --symbol ETHUSDT --timesteps 300000")
        print("\nОпції:")
        parser.print_help()
        sys.exit(0)
    
    if args.interactive:
        # Інтерактивний режим
        main()
    else:
        # Швидкий запуск без інтерактивності (за замовчуванням)
        print_banner()
        print(f"⚡ Швидкий запуск навчання:")
        print(f"   Пара: {args.symbol}")
        print(f"   Таймфрейм: {args.timeframe}")
        print(f"   Агент: {args.agent}")
        print(f"   Кроків: {args.timesteps:,}")
        
        try:
            agent = quick_train(
                symbol=args.symbol,
                timeframe=args.timeframe,
                agent_type=args.agent,
                timesteps=args.timesteps,
                reward_scheme='optimized'
            )
            print("✅ Швидке навчання завершено!")
        except Exception as e:
            print(f"❌ Помилка: {e}")