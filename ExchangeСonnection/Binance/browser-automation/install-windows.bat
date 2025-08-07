@echo off
echo 🤖 Встановлення Binance Browser Bot на Windows
echo ===============================================

REM Перевірка Node.js
echo ℹ️  Перевірка Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js не знайдено!
    echo 📥 Завантажте з https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js знайдено
node --version

REM Встановлення npm залежностей
echo.
echo 📦 Встановлення залежностей...
call npm install
if %errorlevel% neq 0 (
    echo ❌ Помилка встановлення npm пакетів
    pause
    exit /b 1
)

REM Встановлення Playwright
echo.
echo 🎭 Встановлення Playwright браузерів...
call npx playwright install chromium
if %errorlevel% neq 0 (
    echo ❌ Помилка встановлення браузерів
    pause
    exit /b 1
)

REM Створення .env файлу
echo.
echo ⚙️  Налаштування конфігурації...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo ✅ Файл .env створено
        echo ⚠️  ВАЖЛИВО: Відредагуйте файл .env з вашими налаштуваннями!
    ) else (
        echo ❌ Файл .env.example не знайдено
    )
) else (
    echo ℹ️  Файл .env вже існує
)

REM Створення директорій
echo.
echo 📁 Створення робочих директорій...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "screenshots" mkdir screenshots
echo ✅ Директорії створено

REM Тестування
echo.
echo 🧪 Запуск тестів...
call node -e "console.log('✅ Node.js працює'); console.log('Node версія:', process.version);"
if %errorlevel% neq 0 (
    echo ❌ Помилка тестування
) else (
    echo ✅ Тести пройшли
)

echo.
echo 🎉 Встановлення завершено!
echo ===============================================
echo.
echo 📋 Наступні кроки:
echo 1. Відредагуйте файл .env з вашими налаштуваннями
echo 2. Додайте проксі та облікові дані Binance
echo 3. Запустіть: node start.js
echo.
echo 💡 Для демо режиму запустіть: node start.js (оберіть опцію 5)
echo.
echo ⚠️  ВАЖЛИВО: Ніколи не комітьте файл .env в Git!
echo.
pause