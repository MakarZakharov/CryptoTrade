import os
import sys
import warnings

# Додаємо шлях до test_strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../strategies/TestStrategies'))
try:
    from test_strategy import main as test_strategy_main
    TEST_STRATEGY_AVAILABLE = True
except ImportError:
    TEST_STRATEGY_AVAILABLE = False
    print("⚠️ test_strategy.py не знайдено або має помилки імпорту")

warnings.filterwarnings('ignore')


def run_test_strategy():
    """Запуск тестової стратегії з test_strategy.py"""
    if not TEST_STRATEGY_AVAILABLE:
        print("❌ test_strategy.py недоступний")
        return

    print("🎯 ЗАПУСК TEST_STRATEGY.PY")
    print("=" * 50)

    try:
        # Викликаємо main функцію з test_strategy.py
        test_strategy_main()
        print("✅ test_strategy.py виконано успішно!")
    except Exception as e:
        print(f"❌ Помилка при запуску test_strategy: {e}")


def main():
    """Головна функція - активація test_strategy.py"""
    print("🎯 АКТИВАЦІЯ TEST_STRATEGY.PY")
    print("=" * 40)

    run_test_strategy()


if __name__ == '__main__':
    main()