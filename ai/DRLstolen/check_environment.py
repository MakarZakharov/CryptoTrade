"""
Проверка виртуального окружения и зависимостей
"""

import sys
import platform

def check_environment():
    print("🔍 ПРОВЕРКА ВИРТУАЛЬНОГО ОКРУЖЕНИЯ")
    print("=" * 50)
    
    # Информация о системе
    print(f"🐍 Python версия: {sys.version}")
    print(f"💻 Платформа: {platform.system()} {platform.release()}")
    print(f"📁 Python путь: {sys.executable}")
    print(f"📦 Virtual env: {'Да' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'Нет'}")
    
    print("\n📚 ПРОВЕРКА БИБЛИОТЕК:")
    print("-" * 30)
    
    # Список библиотек для проверки
    libraries = [
        ('pandas', 'pd'),
        ('numpy', 'np'), 
        ('matplotlib', 'plt'),
        ('stable_baselines3', 'sb3'),
        ('gymnasium', 'gym'),
        ('torch', None),
        ('yfinance', 'yf'),
        ('ccxt', None),
    ]
    
    installed = []
    missing = []
    
    for lib_name, alias in libraries:
        try:
            if alias:
                exec(f"import {lib_name} as {alias}")
            else:
                exec(f"import {lib_name}")
            
            # Получаем версию если возможно
            try:
                version = eval(f"{alias or lib_name}.__version__")
                print(f"✅ {lib_name}: {version}")
            except:
                print(f"✅ {lib_name}: установлена")
            installed.append(lib_name)
            
        except ImportError:
            print(f"❌ {lib_name}: НЕ установлена")
            missing.append(lib_name)
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"✅ Установлено: {len(installed)}")
    print(f"❌ Отсутствует: {len(missing)}")
    
    if missing:
        print(f"\n📝 ДЛЯ УСТАНОВКИ ОТСУТСТВУЮЩИХ:")
        print(f"pip install {' '.join(missing)}")
    
    print(f"\n🎯 ГОТОВНОСТЬ: {'🟢 ГОТОВО' if len(missing) < 3 else '🟡 ЧАСТИЧНО' if len(missing) < 6 else '🔴 НЕ ГОТОВО'}")
    
    return len(missing) < 3

if __name__ == "__main__":
    check_environment()