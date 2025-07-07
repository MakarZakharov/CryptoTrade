#!/usr/bin/env python3
"""
MVP скрипт для быстрого запуска обучения STAS_ML агента.
Простой интерфейс для начала обучения с минимальными настройками.
"""

import os
import sys
import argparse
from datetime import datetime

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.STAS_ML.config.trading_config import (
    TradingConfig, DataManager, interactive_config_creator
)
from CryptoTrade.ai.STAS_ML.training.train import DRLTrainer, quick_train
from CryptoTrade.ai.STAS_ML.evaluation.evaluate import quick_evaluate


def print_banner():
    """Вывести баннер программы."""
    print("🚀" + "="*60 + "🚀")
    print("   MVP ОБУЧЕНИЕ STAS_ML АГЕНТА ДЛЯ ТОРГОВЛИ КРИПТОВАЛЮТАМИ")
    print("🚀" + "="*60 + "🚀")
    print()


def check_dependencies():
    """Проверить наличие необходимых зависимостей."""
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
        print("❌ Отсутствуют зависимости:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Установите зависимости:")
        print("   pip install -r CryptoTrade/requirements.txt")
        return False
    
    print("✅ Все зависимости установлены")
    return True


def show_available_data():
    """Показать доступные данные."""
    print("📊 Доступные данные:")
    available_pairs = DataManager.get_available_pairs()
    
    total_pairs = 0
    for exchange, pairs in available_pairs.items():
        print(f"   {exchange}: {len(pairs)} пар")
        total_pairs += len(pairs)
    
    print(f"   Всего: {total_pairs} торговых пар")
    print()


def create_quick_config():
    """Создать быструю конфигурацию."""
    print("⚡ Быстрая настройка (рекомендуется для начинающих):")
    print("   1. BTCUSDT на дневном таймфрейме")
    print("   2. PPO агент")
    print("   3. Оптимизированная схема наград")
    print("   4. 100,000 шагов обучения")
    
    choice = input("\nИспользовать быструю настройку? (y/n): ").lower()
    
    if choice in ['y', 'yes', 'да', '']:
        return TradingConfig(
            symbol='BTCUSDT',
            timeframe='1d',
            reward_scheme='optimized',
            initial_balance=100.0
        ), "PPO", 100000
    
    return None, None, None


def custom_config_menu():
    """Меню кастомной конфигурации."""
    print("\n🛠️ Кастомная настройка:")
    
    # Выбор пары
    available_pairs = DataManager.get_available_pairs()
    print("\nДоступные биржи:")
    exchanges = list(available_pairs.keys())
    for i, exchange in enumerate(exchanges, 1):
        print(f"   {i}. {exchange}")
    
    while True:
        try:
            choice = int(input(f"Выберите биржу (1-{len(exchanges)}): ")) - 1
            if 0 <= choice < len(exchanges):
                selected_exchange = exchanges[choice]
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
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
        initial_balance=100.0
    )
    
    return config, agent_type, timesteps


def main():
    """Главная функция MVP - автоматический запуск."""
    print_banner()
    
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    # Спрашиваем у пользователя название модели
    print("🏷️ Настройка имени модели:")
    print("   1. Автоматическое имя (BTCUSDT_1d_optimized)")
    print("   2. Пользовательское имя")
    
    custom_name = None
    choice = input("Выберите (1-2) или Enter для автоматического: ").strip()
    
    if choice == "2":
        custom_name = input("Введите имя модели (например, my_best_model): ").strip()
        if not custom_name:
            print("⚠️ Пустое имя, используем автоматическое")
            custom_name = None
    
    # ОПТИМІЗОВАНА конфігурація для роботи з падаючими ринками
    config = TradingConfig(
        symbol='BTCUSDT',
        timeframe='1d',
        reward_scheme='bear_market_optimized',  # Спеціальна схема для падаючих ринків
        initial_balance=10000.0,
        lookback_window=20,  # ЗМЕНШЕНО для швидшої реакції
        
        # ОПТИМІЗОВАНІ ПАРАМЕТРИ для стабільної торгівлі
        enable_position_sizing=True,
        max_risk_per_trade=0.05,  # ЗМЕНШЕНО до 5% для зниження ризику
        position_size_method='fixed_ratio',  # Фіксований ratio для стабільності
        
        # РОЗШИРЕНИЙ Stop-Loss для падаючих ринків
        enable_stop_loss=False,  # ВИМКНЕНО для максимальної активності
        stop_loss_type='percentage',
        stop_loss_percentage=0.30,  # Високий поріг якщо включено
        
        # РОЗШИРЕНІ межі для падаючих ринків
        max_drawdown_limit=0.30,  # 30% для роботи з волатильними падіннями
        reduce_position_on_drawdown=False,  # Не зменшуємо позиції - беремо можливості
        
        # ОПТИМАЛЬНІ торгові параметри
        min_trade_amount=5.0,  # ЗМЕНШЕНО до $5 для дуже частих входів
        commission_rate=0.0001,  # ЗМЕНШЕНО комісію у 10 разів для активності
        slippage_rate=0.0001,   # ЗМЕНШЕНО проскальзування у 5 разів
        spread_rate=0.00005,    # ЗМЕНШЕНО спред у 4 рази
        
        # РОЗШИРЕНІ технічні індикатори для падаючих ринків
        include_technical_indicators=True,
        indicator_periods={
            'sma': [10, 20, 50],  # Додано короткий SMA для швидкої реакції
            'ema': [8, 21, 55],   # EMA для швидшого виявлення змін тренду
            'rsi': [14, 21],      # Два RSI для різних періодів
            'macd': [12, 26, 9],
            'bollinger': [20],
            'atr': [14, 28],      # Два ATR для волатільності
            'stoch': [14],        # Stochastic для oversold/overbought
            'williams_r': [14],   # Williams %R для momentum
        }
    )
    agent_type = "PPO"
    timesteps = 250000  # Збільшено для кращого навчання з новими параметрами
    
    # Показываем настройки
    print(f"🚀 Автоматический запуск обучения:")
    print(f"   Пара: {config.symbol}")
    print(f"   Таймфрейм: {config.timeframe}")
    print(f"   Агент: {agent_type}")
    print(f"   Шагов: {timesteps:,}")
    print(f"   Схема наград: {config.reward_scheme}")
    if custom_name:
        print(f"   Имя модели: {custom_name}")
    print(f"💡 Мониторинг: tensorboard --logdir logs")
    print(f"💡 Для остановки: Ctrl+C")
    print("-" * 60)
    
    try:
        trainer = DRLTrainer(config, resume_training=True, custom_model_name=custom_name)
        agent = trainer.train(
            total_timesteps=timesteps,
            agent_type=agent_type
        )
        
        print(f"\n✅ Обучение завершено успешно!")
        print(f"📁 Модель сохранена в: {trainer.save_dir}/{trainer.experiment_name}")
        print(f"📊 Логи в: logs/{trainer.experiment_name}")
        
        # Автоматическая оценка
        print("\n🔍 Запуск автоматической оценки...")
        model_path = f"{trainer.save_dir}/{trainer.experiment_name}/final_model"
        try:
            evaluator, results, report = quick_evaluate(
                model_path=model_path,
                symbol=config.symbol,
                timeframe=config.timeframe,
                agent_type=agent_type,
                episodes=5
            )
            print(f"✅ Оценка завершена!")
        except Exception as e:
            print(f"❌ Ошибка при оценке: {e}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Обучение остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        print(f"💡 Проверьте логи в: logs/")


if __name__ == "__main__":
    # Поддержка аргументов командной строки для продвинутых пользователей
    parser = argparse.ArgumentParser(description='MVP обучение STAS_ML агента', add_help=False)
    parser.add_argument('--quick', action='store_true', 
                       help='Быстрый запуск с настройками по умолчанию')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--agent', default='PPO', choices=['PPO', 'DQN'], help='Тип агента')
    parser.add_argument('--timesteps', type=int, default=100000, help='Количество шагов')
    parser.add_argument('--help', '-h', action='store_true', help='Показать помощь')
    
    args = parser.parse_args()
    
    if args.help:
        print("🚀 MVP Обучение STAS_ML Агента")
        print("\nИспользование:")
        print("  python mvp_train.py                    # Интерактивный режим")
        print("  python mvp_train.py --quick            # Быстрый запуск")
        print("  python mvp_train.py --quick --symbol ETHUSDT --timesteps 200000")
        print("\nОпции:")
        parser.print_help()
        sys.exit(0)
    
    if args.quick:
        # Быстрый запуск без интерактивности
        print_banner()
        print(f"⚡ Быстрый запуск обучения:")
        print(f"   Пара: {args.symbol}")
        print(f"   Таймфрейм: {args.timeframe}")
        print(f"   Агент: {args.agent}")
        print(f"   Шагов: {args.timesteps:,}")
        
        try:
            agent = quick_train(
                symbol=args.symbol,
                timeframe=args.timeframe,
                agent_type=args.agent,
                timesteps=args.timesteps,
                reward_scheme='optimized'
            )
            print("✅ Быстрое обучение завершено!")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    else:
        # Интерактивный режим
        main()