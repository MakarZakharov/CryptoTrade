import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

load_dotenv()


def test_binance_connection():
    """Тестує підключення до Binance API"""

    print("🔍 Тестування Binance API...")
    print("-" * 60)

    # Завантажуємо ключі
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    # Перевірка наявності ключів
    if not api_key or not api_secret:
        print("❌ API ключі не знайдені в .env файлі!")
        return False

    print(f"✅ API Key знайдено: {api_key[:10]}...")
    print(f"✅ API Secret знайдено: {api_secret[:10]}...")
    print()

    try:
        # Створюємо клієнта
        client = Client(api_key, api_secret)

        # Тест 1: Перевірка статусу системи (не потребує авторизації)
        print("📡 Тест 1: Перевірка статусу Binance...")
        status = client.get_system_status()
        print(f"   Статус системи: {status['status']}")
        print()

        # Тест 2: Отримання часу сервера
        print("⏰ Тест 2: Синхронізація часу...")
        server_time = client.get_server_time()
        print(f"   Час сервера: {server_time['serverTime']}")
        print()

        # Тест 3: Тест API ключа (потребує авторизації)
        print("🔑 Тест 3: Перевірка API ключа...")
        account = client.get_account()
        print(f"   ✅ API ключ валідний!")
        print(f"   Тип акаунту: {account['accountType']}")
        print(f"   Можливість торгувати: {account['canTrade']}")
        print(f"   Можливість виводити: {account['canWithdraw']}")
        print()

        # Тест 4: Отримання балансів
        print("💰 Тест 4: Отримання балансів...")
        balances = client.get_account()['balances']
        non_zero = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]

        if non_zero:
            print(f"   Знайдено {len(non_zero)} активів:")
            for balance in non_zero[:5]:  # Показуємо перші 5
                total = float(balance['free']) + float(balance['locked'])
                print(f"   - {balance['asset']}: {total:.8f}")
        else:
            print("   ⚠️ Баланси порожні")
        print()

        # Тест 5: Перевірка торгових прав
        print("🔐 Тест 5: Перевірка торгових прав...")
        permissions = account.get('permissions', [])
        print(f"   Дозволи: {', '.join(permissions)}")

        if 'SPOT' in permissions:
            print("   ✅ Spot торгівля дозволена")
        else:
            print("   ❌ Spot торгівля заборонена")

        print()
        print("=" * 60)
        print("✅ Всі тести пройдено успішно!")
        print("=" * 60)
        return True

    except BinanceAPIException as e:
        print("\n" + "=" * 60)
        print("❌ ПОМИЛКА BINANCE API:")
        print("=" * 60)
        print(f"Код помилки: {e.code}")
        print(f"Повідомлення: {e.message}")
        print()

        # Пояснення помилок
        if e.code == -2015:
            print("🔍 Рішення для помилки -2015:")
            print("1. Перевірте права API ключа на Binance:")
            print("   - Enable Reading ✅")
            print("   - Enable Spot & Margin Trading ✅")
            print()
            print("2. Перевірте IP обмеження:")
            print("   - Додайте вашу IP адресу до білого списку")
            print("   - Або вимкніть IP обмеження")
            print()
            print("3. Створіть новий API ключ, якщо проблема не вирішується")

        elif e.code == -1021:
            print("🔍 Рішення для помилки -1021:")
            print("Час на вашому комп'ютері не синхронізований з Binance")
            print("Синхронізуйте час в Windows або додайте recvWindow параметр")

        return False

    except Exception as e:
        print(f"\n❌ Невідома помилка: {e}")
        return False


if __name__ == "__main__":
    test_binance_connection()