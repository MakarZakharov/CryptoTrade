#!/usr/bin/env python3
"""
Специализированный скрипт для прибыльной торговли на 15-минутных таймфреймах.
Оптимизирован для реальной торговли с высоким win rate и контролем рисков.
"""

import os
import sys
import argparse
from datetime import datetime

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.DRL.config.trading_config_15m import (
    TradingConfig15m, DataManager15m, interactive_15m_config_creator,
    create_15m_config, get_popular_15m_pairs
)
from CryptoTrade.ai.DRL.training.train import DRLTrainer
from CryptoTrade.ai.DRL.evaluation.evaluate import quick_evaluate


def print_banner():
    """Вывести баннер программы."""
    print("💰" + "="*70 + "💰")
    print("   ПРИБЫЛЬНАЯ ТОРГОВЛЯ DRL НА 15-МИНУТНЫХ ТАЙМФРЕЙМАХ")
    print("💰" + "="*70 + "💰")
    print()


def check_15m_dependencies():
    """Проверить зависимости для 15мин торговли."""
    missing_deps = []
    
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 GPU доступен для ускоренного обучения")
        else:
            print("🔧 Используется CPU")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import stable_baselines3
    except ImportError:
        missing_deps.append("stable-baselines3")
    
    if missing_deps:
        print("❌ Отсутствуют зависимости:")
        for dep in missing_deps:
            print(f"   - {dep}")
        return False
    
    print("✅ Все зависимости установлены")
    return True


def show_15m_data_status():
    """Показать статус 15мин данных."""
    print("📊 Статус 15-минутных данных:")
    
    stats = DataManager15m.get_15m_data_stats()
    good_pairs = []
    
    for pair_key, data in stats.items():
        if data['quality'] == 'good':
            exchange, symbol = pair_key.split('_', 1)
            good_pairs.append((exchange, symbol, data['records']))
    
    if not good_pairs:
        print("❌ Нет качественных 15мин данных!")
        print("💡 Рекомендации:")
        print("   1. Запустите обновление данных")
        print("   2. Проверьте интернет соединение")
        print("   3. Попробуйте другие торговые пары")
        return False
    
    print(f"✅ Найдено {len(good_pairs)} пар с качественными данными")
    
    # Показываем топ-5
    good_pairs.sort(key=lambda x: x[2], reverse=True)
    print("🔝 Топ-5 пар по объему данных:")
    for i, (exchange, symbol, records) in enumerate(good_pairs[:5], 1):
        print(f"   {i}. {exchange}:{symbol} ({records:,} записей)")
    
    return True


def create_profitable_config(symbol: str = None, balance: float = 1000.0):
    """Создать конфигурацию для прибыльной торговли."""
    if symbol:
        # Автоматическая конфигурация
        config = create_15m_config(symbol, 'binance')
        config.initial_balance = balance
        
        if DataManager15m.validate_15m_config(config):
            return config
        else:
            print(f"❌ Пара {symbol} не подходит для 15мин торговли")
            return None
    else:
        # Интерактивная конфигурация
        return interactive_15m_config_creator()


def optimize_training_params(config: TradingConfig15m, fast_mode: bool = False):
    """Получить оптимизированные параметры обучения."""
    if fast_mode:
        # Быстрое обучение для тестирования
        return {
            'total_timesteps': 50000,  # 50k шагов ~2-3 часа
            'eval_freq': 5000,
            'save_freq': 10000,
            'model_config': {
                'learning_rate': 2e-4,  # Чуть быстрее для тестов
                'n_steps': 512,
                'batch_size': 64
            }
        }
    else:
        # Полное обучение для продакшна
        return {
            'total_timesteps': 500000,  # 500k шагов ~1-2 дня
            'eval_freq': 10000,
            'save_freq': 25000,
            'model_config': {
                'learning_rate': 1e-4,  # Стабильное обучение
                'n_steps': 1024,
                'batch_size': 128,
                'n_epochs': 4
            }
        }


def train_profitable_agent(config: TradingConfig15m, model_name: str = None, 
                          fast_mode: bool = False):
    """Обучить агента для прибыльной 15мин торговли."""
    print(f"🎯 Цель обучения: Win Rate >55%, Просадка <15%")
    print(f"⚡ Режим: {'Быстрый тест' if fast_mode else 'Полное обучение'}")
    print(f"💱 Пара: {config.symbol} на 15мин")
    print(f"💰 Капитал: {config.initial_balance} USDT")
    print("-" * 60)
    
    # Получаем оптимизированные параметры
    params = optimize_training_params(config, fast_mode)
    
    # Создаем trainer с кастомным именем модели
    if not model_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = f"profitable_15m_{config.symbol}_{timestamp}"
    
    trainer = DRLTrainer(
        config=config, 
        resume_training=True,
        custom_model_name=model_name
    )
    
    # Запускаем обучение
    try:
        print(f"🚀 Начинаем обучение для прибыльной торговли...")
        print(f"📈 Ожидаемое время: {params['total_timesteps']//1000}k шагов")
        
        agent = trainer.train(
            total_timesteps=params['total_timesteps'],
            eval_freq=params['eval_freq'],
            save_freq=params['save_freq'],
            agent_type="PPO",
            model_config=params['model_config']
        )
        
        print(f"\n🎉 Обучение завершено успешно!")
        print(f"📁 Модель: {trainer.save_dir}/{trainer.experiment_name}")
        
        # Быстрая оценка
        print(f"\n🔍 Оценка производительности...")
        model_path = f"{trainer.save_dir}/{trainer.experiment_name}/final_model"
        
        try:
            evaluator, results, report = quick_evaluate(
                model_path=model_path,
                symbol=config.symbol,
                timeframe=config.timeframe,
                agent_type="PPO",
                episodes=5
            )
            
            # Анализ результатов
            win_rate = results['mean_win_rate']
            max_drawdown = results['mean_drawdown']
            total_return = results['mean_return']
            
            print(f"\n📊 РЕЗУЛЬТАТЫ ОЦЕНКИ:")
            print(f"   Win Rate: {win_rate:.1%} {'✅' if win_rate > 0.55 else '❌'}")
            print(f"   Просадка: {max_drawdown:.1%} {'✅' if max_drawdown < 0.15 else '❌'}")
            print(f"   Доходность: {total_return:.1%}")
            
            # Рекомендации
            if win_rate > 0.55 and max_drawdown < 0.15:
                print(f"\n🎯 МОДЕЛЬ ГОТОВА ДЛЯ ТОРГОВЛИ!")
                print(f"   Агент показывает прибыльные результаты")
                print(f"   Риски контролируются в допустимых пределах")
            else:
                print(f"\n⚠️ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНОЕ ОБУЧЕНИЕ:")
                if win_rate <= 0.55:
                    print(f"   • Увеличьте время обучения для лучшего Win Rate")
                if max_drawdown >= 0.15:
                    print(f"   • Настройте управление рисками")
                    
        except Exception as e:
            print(f"❌ Ошибка при оценке: {e}")
        
        return trainer, agent
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Обучение остановлено пользователем")
        return trainer, None
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        return None, None


def quick_profitable_setup():
    """Быстрая настройка для прибыльной торговли."""
    print("⚡ Быстрая настройка прибыльной торговли на 15мин")
    
    # Выбираем лучшие пары
    popular_pairs = get_popular_15m_pairs()
    print(f"\n📈 Рекомендуемые пары для 15мин торговли:")
    
    for i, pair in enumerate(popular_pairs[:5], 1):
        print(f"   {i}. {pair}")
    
    while True:
        try:
            choice = input(f"\nВыберите пару (1-5) или Enter для BTCUSDT: ").strip()
            if not choice:
                selected_pair = 'BTCUSDT'
                break
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < 5:
                selected_pair = popular_pairs[choice_idx]
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Выбираем режим обучения
    print(f"\n🎯 Режим обучения:")
    print(f"   1. Быстрый тест (50k шагов, ~2 часа)")
    print(f"   2. Полное обучение (500k шагов, ~1-2 дня)")
    
    while True:
        try:
            mode_choice = input("Выберите режим (1-2): ").strip()
            fast_mode = mode_choice == "1"
            break
        except:
            pass
        print("❌ Неверный выбор!")
    
    return selected_pair, fast_mode


def main():
    """Главная функция."""
    print_banner()
    
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    # Проверяем данные
    if not show_15m_data_status():
        return
    
    # Парсим аргументы
    parser = argparse.ArgumentParser(description='Прибыльная 15мин торговля DRL')
    parser.add_argument('--symbol', help='Торговая пара (например, BTCUSDT)')
    parser.add_argument('--balance', type=float, default=1000.0, help='Начальный капитал')
    parser.add_argument('--fast', action='store_true', help='Быстрое обучение')
    parser.add_argument('--model-name', help='Имя модели')
    parser.add_argument('--quick', action='store_true', help='Быстрая настройка')
    
    args = parser.parse_args()
    
    if args.quick:
        # Быстрая настройка
        symbol, fast_mode = quick_profitable_setup()
        config = create_profitable_config(symbol, args.balance)
        
        if config:
            train_profitable_agent(config, args.model_name, fast_mode)
    
    elif args.symbol:
        # Автоматическая настройка
        config = create_profitable_config(args.symbol, args.balance)
        
        if config:
            train_profitable_agent(config, args.model_name, args.fast)
        else:
            print(f"❌ Не удалось создать конфигурацию для {args.symbol}")
    
    else:
        # Интерактивная настройка
        config = create_profitable_config()
        
        if config:
            # Спрашиваем про режим обучения
            print(f"\n🎯 Режим обучения:")
            print(f"   1. Быстрый тест (50k шагов)")
            print(f"   2. Полное обучение (500k шагов)")
            
            mode_choice = input("Выберите режим (1-2) или Enter для полного: ").strip()
            fast_mode = mode_choice == "1"
            
            train_profitable_agent(config, args.model_name, fast_mode)
        else:
            print("❌ Не удалось создать конфигурацию")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n👋 Программа завершена пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()