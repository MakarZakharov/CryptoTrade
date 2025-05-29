import os
import sys
import warnings

# Додаємо шлях до оптимізованої стратегії
sys.path.append(os.path.join(os.path.dirname(__file__), '../../strategies/TestStrategies'))

warnings.filterwarnings('ignore')


def main():
    """Запуск оптимізованої BTC стратегії"""

    print("🎯 ЗАПУСК ОПТИМІЗОВАНОЇ СТРАТЕГІЇ")
    print("=" * 50)

    try:
        from test_strategy import main as strategy_main
        strategy_main()
        print("\n✅ Стратегія виконана успішно!")

    except ImportError as e:
        print(f"❌ Помилка імпорту: {e}")
    except Exception as e:
        print(f"❌ Помилка виконання: {e}")


if __name__ == '__main__':
    main()